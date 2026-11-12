"""How PGD feeds each detector family and reads its plate scores (milestone 2b).

The attack must see each model the way evaluation does, and must push on the scores evaluation
keeps. Both differ by family, in the installed Ultralytics (8.4.178):

- YOLOv8 (`DetectView`): predict() pads the image to a multiple of 32; the eval output is
  (y, raw) with y of shape (1, 4 + classes, anchors), already sigmoid, row 4 the plate score.
- YOLOv10: our fine-tuned checkpoints load with `end2end = False`, so predict() fuses the model,
  drops the one-to-one head and runs the one-to-many head with NMS, exactly like YOLOv8; the
  unfused network's eval output is that same one-to-many output. So YOLOv10 gets `DetectView`.
  A YOLOv10 with `end2end = True` would return a top-300 table first instead, and is refused.
- RT-DETR (`RTDETRView`): predict() stretches the image to an imgsz x imgsz square
  (scale_fill); the eval output is (top-300 table, (dec_bboxes, dec_scores, enc_bboxes,
  enc_scores, dn_meta)) with dec_scores the last decoder layer's logits, (1, batch, queries,
  classes).

Reading element 0 of RT-DETR's (or an end-to-end YOLOv10's) output as if it were YOLOv8's layout
does not fail: it quietly optimizes a few numbers of the top-k table, and the model looks robust.
Each view therefore reads its own family's raw scores, before any top-k, which is the set the
evaluation's detections are taken from (checked against predict() in tests/test_views_real.py).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn

PLATE = 0  # the only class in every fine-tuned model
STRIDE = 32  # YOLO's largest stride: its input sides must be multiples of this
PAD_VALUE = 114 / 255  # the gray Ultralytics pads with at prediction time


def predict_padding(h: int, w: int) -> tuple[int, int, int, int]:
    """(left, right, top, bottom) padding that YOLO's predict() adds to an h x w image.

    It pads each side up to the next multiple of 32 and centers the image (LetterBox with
    auto=True). The surrogate must see the same layout during the attack as at evaluation.
    """
    dw, dh = (-w) % STRIDE, (-h) % STRIDE
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    return left, dw - left, top, dh - top


class View(Protocol):
    supports_hinge: bool
    layout: str  # views with equal layouts build the same input from the same image

    def prepare(self, x: Tensor) -> Tensor:
        """Eval-size image (1, 3, H, W) in [0, 1] -> the network's input, as predict() builds it."""

    def plates_to_input(self, boxes: np.ndarray, h: int, w: int) -> np.ndarray:
        """Plate boxes (N, 4) x1, y1, x2, y2 on the h x w image -> the same boxes on the input."""

    def scores(self, output) -> Tensor:
        """The network's eval-mode output -> (A,) plate confidence per candidate, differentiable."""


class DetectView:
    """YOLOv8's Detect head (milestone 1 and 2a's only case)."""

    supports_hinge = True
    layout = "pad32"

    def prepare(self, x: Tensor) -> Tensor:
        return F.pad(x, predict_padding(*x.shape[-2:]), value=PAD_VALUE)

    def plates_to_input(self, boxes: np.ndarray, h: int, w: int) -> np.ndarray:
        left, _, top, _ = predict_padding(h, w)
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) + [left, top, left, top]

    def scores(self, output) -> Tensor:
        return raw_detect(output)[0, 4]


class RTDETRView:
    """RT-DETR: the image stretched to an imgsz x imgsz square, scores from the decoder."""

    supports_hinge = False

    def __init__(self, imgsz: int):
        self.imgsz = imgsz
        self.layout = f"square{imgsz}"

    def prepare(self, x: Tensor) -> Tensor:
        # predict() resizes the 8-bit image with cv2's bilinear filter; this is the same filter on
        # floats (half-pixel centers, no antialiasing), within a level of 255 per pixel.
        return F.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)

    def plates_to_input(self, boxes: np.ndarray, h: int, w: int) -> np.ndarray:
        scale = np.array([self.imgsz / w, self.imgsz / h] * 2)
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * scale

    def scores(self, output) -> Tensor:
        dec_scores = output[1][1]
        return dec_scores[-1, 0, :, PLATE].sigmoid()


def view_for(name: str, net: nn.Module, imgsz: int) -> View:
    """The view for a zoo model; refuses a YOLO head in end-to-end mode (see the module notes)."""
    from plate_cloak.models.zoo import spec

    if spec(name).family == "rtdetr":
        return RTDETRView(imgsz)
    head = net.model[-1] if hasattr(net, "model") else None
    if getattr(head, "end2end", False):
        raise NotImplementedError(
            f"{name} has an end-to-end head; its eval output is a top-k table, not per-anchor "
            "scores, and no view reads it yet"
        )
    return DetectView()


def raw_detect(output) -> Tensor:
    """The (1, 5, A) prediction tensor from a YOLOv8 Detect head's eval-mode output."""
    return output[0] if isinstance(output, tuple | list) else output
