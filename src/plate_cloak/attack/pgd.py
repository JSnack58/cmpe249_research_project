"""PGD on the plates, the random-noise control, and the budget audit.

Images are float tensors (1, 3, H, W) in [0, 1], already at the evaluation size (long side 640),
and `boxes` are every annotated plate in that image as (N, 4) x1, y1, x2, y2 pixels.

The noise `delta` lives in the attack's own space: RGB units for space="rgb", Y/Cb/Cr units for
space="ycbcr". YCbCr noise is applied to the image as the RGB change it causes,
`x + YCBCR_TO_RGB @ delta`. That is the same as converting to YCbCr, adding delta and converting
back (the conversion is linear), but it leaves pixels with zero noise bit-for-bit unchanged
instead of picking up float round-off from the round trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plate_cloak.attack.colorspace import RGB_TO_YCBCR, YCBCR_TO_RGB, fit_to_gamut
from plate_cloak.attack.losses import anchors_on_plates, hinge_loss, topk_loss

STRIDE = 32  # YOLOv8's largest stride: its input sides must be multiples of this
PAD_VALUE = 114 / 255  # the gray Ultralytics pads with at prediction time


@dataclass(frozen=True)
class LossSettings:
    type: Literal["topk", "hinge"]
    k: int  # topk: how many of the highest scores to push down
    floor: float  # hinge: scores below this are left alone


@dataclass(frozen=True)
class AttackSettings:
    method: Literal["pgd", "random"]
    space: Literal["rgb", "ycbcr"]
    eps: Tensor  # (3,) per-channel L_inf budget in [0, 1] units (R, G, B or Y, Cb, Cr)
    steps: int
    step_factor: float  # per-step size = step_factor * eps / steps
    margin: float  # noise region = plate box grown by this fraction on each side
    mask: Literal["uniform", "sobel"]
    loss: LossSettings
    quant_margin: float  # YCbCr only, in 1/255 units: reserved for 8-bit rounding

    @classmethod
    def from_cfg(cls, attack) -> AttackSettings:
        """Build from the config's `attack:` block (budgets there are in 1/255 units)."""
        if attack.space == "rgb":
            eps_255 = [attack.eps.rgb] * 3
        elif attack.space == "ycbcr":
            eps_255 = [attack.eps.y, attack.eps.cb, attack.eps.cr]
        else:
            raise ValueError(f"unknown attack.space {attack.space!r}; options: rgb, ycbcr")
        if attack.mask != "uniform":
            raise NotImplementedError(f"attack.mask={attack.mask!r} arrives in milestone 2")
        return cls(
            method=attack.method,
            space=attack.space,
            eps=torch.tensor(eps_255, dtype=torch.float32) / 255,
            steps=attack.steps,
            step_factor=attack.step_factor,
            margin=attack.margin,
            mask=attack.mask,
            loss=LossSettings(attack.loss.type, attack.loss.k, attack.loss.floor),
            quant_margin=attack.quant_margin,
        )


def plate_mask(h: int, w: int, boxes: np.ndarray, margin: float) -> Tensor:
    """(1, 1, H, W) mask: 1 on every plate box grown by `margin` per side, 0 elsewhere.

    Pass every annotated plate, including ones too small to count in evaluation, so no plate in
    the image is left clean.
    """
    mask = torch.zeros(1, 1, h, w)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
        top, bottom = max(0, int(np.floor(y1 - dy))), min(h, int(np.ceil(y2 + dy)))
        left, right = max(0, int(np.floor(x1 - dx))), min(w, int(np.ceil(x2 + dx)))
        mask[..., top:bottom, left:right] = 1.0
    return mask


def predict_padding(h: int, w: int) -> tuple[int, int, int, int]:
    """(left, right, top, bottom) padding that Ultralytics' predict() adds to an h x w image.

    It pads each side up to the next multiple of 32 and centers the image (LetterBox with
    auto=True). The surrogate must see the same layout during the attack as at evaluation.
    """
    dw, dh = (-w) % STRIDE, (-h) % STRIDE
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    return left, dw - left, top, dh - top


def pgd(net: nn.Module, x: Tensor, boxes: np.ndarray, s: AttackSettings) -> Tensor:
    """Run PGD against `net` (the surrogate) and return the adversarial image in [0, 1].

    Each step moves the noise against the sign of the loss gradient, clips it to the budget
    (which is 0 off the plates), then keeps the result inside the valid color range.
    The start is zero noise, so the attack is deterministic.
    """
    if s.method != "pgd":
        raise ValueError(f"pgd() got method={s.method!r}; use random_sign_noise for the control")
    if net.training:
        raise ValueError("put the surrogate in eval mode first (BatchNorm would update)")

    eps = s.eps.to(x).view(1, 3, 1, 1)
    bound = _bound(x, boxes, s)
    step = s.step_factor * eps / s.steps
    left, right, top, bottom = predict_padding(*x.shape[-2:])
    boxes_padded = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) + [left, top, left, top]

    delta = torch.zeros_like(x)
    for _ in range(s.steps):
        with torch.enable_grad():  # works even if the caller is inside torch.no_grad()
            delta.requires_grad_(True)
            adv = _apply(x, delta, s.space)
            raw = _raw(net(F.pad(adv, (left, right, top, bottom), value=PAD_VALUE)))
            (grad,) = torch.autograd.grad(_loss(raw, boxes_padded, s.loss), delta)
        with torch.no_grad():
            delta = _project(x, delta - step * grad.sign(), bound, s.space)
    with torch.no_grad():
        return _apply(x, delta, s.space)


def random_sign_noise(
    x: Tensor, boxes: np.ndarray, s: AttackSettings, generator: torch.Generator
) -> Tensor:
    """The control: noise of the full budget with a random sign per pixel and channel.

    Same budget, region and color-range handling as PGD, so any gap in recall between the two
    comes from PGD's choice of direction, not from the amount of noise. `generator` must be on
    x's device.
    """
    sign = torch.randint(0, 2, x.shape, generator=generator, device=x.device).to(x) * 2 - 1
    bound = _bound(x, boxes, s)
    with torch.no_grad():
        return _apply(x, _project(x, sign * bound, bound, s.space), s.space)


def to_uint8(x: Tensor) -> np.ndarray:
    """(1, 3, H, W) float image in [0, 1] -> (H, W, 3) uint8, rounded to the nearest level."""
    return (x[0].detach().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def audit(clean_u8: np.ndarray, adv_u8: np.ndarray) -> dict[str, float]:
    """Largest absolute change per channel over the whole image, in 1/255 units.

    Computed on the 8-bit images that evaluation actually uses, so it includes rounding.
    Inputs are (H, W, 3) uint8 RGB.
    """
    if clean_u8.shape != adv_u8.shape:
        raise ValueError(f"shape mismatch: {clean_u8.shape} vs {adv_u8.shape}")
    diff_rgb = adv_u8.astype(np.float64) - clean_u8.astype(np.float64)  # already in 1/255 units
    diff_ycc = diff_rgb @ RGB_TO_YCBCR.numpy().T  # the chroma offset cancels in a diff
    rgb, ycc = np.abs(diff_rgb).max(axis=(0, 1)), np.abs(diff_ycc).max(axis=(0, 1))
    return {
        "R": float(rgb[0]),
        "G": float(rgb[1]),
        "B": float(rgb[2]),
        "Y": float(ycc[0]),
        "Cb": float(ycc[1]),
        "Cr": float(ycc[2]),
    }


def _bound(x: Tensor, boxes: np.ndarray, s: AttackSettings) -> Tensor:
    """(1, 3, H, W) per-pixel, per-channel noise limit: the budget on plates, 0 elsewhere."""
    eps = s.eps.to(x).view(1, 3, 1, 1)
    if s.space == "ycbcr":
        # Rounding to 8-bit moves each of Y, Cb, Cr by at most 0.5/255 (each row of the matrix
        # has absolute weights summing to 1), so this reserve keeps the budget on the saved image.
        eps = eps - s.quant_margin / 255
    return eps * plate_mask(*x.shape[-2:], boxes, s.margin).to(x)


def _apply(x: Tensor, delta: Tensor, space: str) -> Tensor:
    """The image with the noise added. The clamp only removes float round-off (~1e-7)."""
    if space == "ycbcr":
        delta = torch.einsum("ij,bjhw->bihw", YCBCR_TO_RGB.to(delta), delta)
    return (x + delta).clamp(0, 1)


def _project(x: Tensor, delta: Tensor, bound: Tensor, space: str) -> Tensor:
    """Clip the noise to the budget, then keep x + noise inside the valid color range."""
    delta = torch.maximum(torch.minimum(delta, bound), -bound)
    if space == "rgb":
        # Each RGB channel has its own budget, so clipping a channel at 0 or 1 can't break it.
        return (x + delta).clamp(0, 1) - x
    # In YCbCr, clipping RGB would leak budget across channels; shrink each pixel's noise
    # along its own direction instead (fit_to_gamut has no useful gradient, and needs none).
    return delta * fit_to_gamut(x, delta)


def _raw(output) -> Tensor:
    """The (1, 5, A) prediction tensor from a Detect head's eval-mode output."""
    return output[0] if isinstance(output, tuple | list) else output


def _loss(raw: Tensor, boxes_padded: np.ndarray, loss: LossSettings) -> Tensor:
    if loss.type == "topk":
        return topk_loss(raw, loss.k)
    if loss.type == "hinge":
        return hinge_loss(raw, loss.floor, anchors_on_plates(raw, boxes_padded))
    raise ValueError(f"unknown attack.loss.type {loss.type!r}; options: topk, hinge")
