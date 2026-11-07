"""Suppression losses on a YOLOv8 detector's raw output (before NMS, which is not differentiable).

`raw` is the Detect head's output in eval mode, shape (1, 5, A): one column per candidate box
("anchor"), rows 0-3 the box as center x, center y, width, height in input pixels, row 4 the
plate score, already passed through a sigmoid. PGD minimizes these losses: a lower loss means
the detector is less sure there is a plate anywhere.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def topk_loss(raw: Tensor, k: int) -> Tensor:
    """Sum of the k highest plate scores (Spike 01's loss).

    Pushing down only the top candidates spends the attack where NMS would find a detection,
    instead of on thousands of background candidates that are already near 0.
    """
    scores = raw[0, 4]
    return scores.topk(min(k, scores.numel())).values.sum()


def hinge_loss(raw: Tensor, floor: float, plate_mask_anchors: Tensor | None) -> Tensor:
    """Sum of how far each candidate's score sits above `floor`, over the selected candidates.

    Candidates already below `floor` contribute 0 (and no gradient), so the attack stops spending
    effort on them. `plate_mask_anchors` is a (A,) bool tensor that limits the loss to candidates
    on a plate (see anchors_on_plates); None uses every candidate.
    """
    excess = (raw[0, 4] - floor).clamp(min=0)
    if plate_mask_anchors is not None:
        excess = excess[plate_mask_anchors]
    return excess.sum()


def anchors_on_plates(raw: Tensor, boxes: np.ndarray) -> Tensor:
    """(A,) bool: True for candidates whose box overlaps at least one plate box.

    boxes: (N, 4) plate boxes as x1, y1, x2, y2 in the same pixel frame as `raw` (the padded
    input). Overlap means any intersection with positive area.
    """
    cx, cy, w, h = raw[0, :4].detach()
    cand = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)  # (A, 4)
    plates = torch.as_tensor(boxes, dtype=cand.dtype, device=cand.device).reshape(-1, 4)
    ix = torch.minimum(cand[:, None, 2], plates[None, :, 2]) - torch.maximum(
        cand[:, None, 0], plates[None, :, 0]
    )
    iy = torch.minimum(cand[:, None, 3], plates[None, :, 3]) - torch.maximum(
        cand[:, None, 1], plates[None, :, 1]
    )
    return ((ix > 0) & (iy > 0)).any(dim=1)
