"""RGB <-> YCbCr conversion and the gamut fit that keeps a YCbCr budget exact.

Tensors are float images in [0, 1] with shape (B, 3, H, W).
YCbCr follows ITU-R BT.601 full range (the JPEG convention); Cb and Cr are offset by 0.5 so all
three channels also live in [0, 1]. The conversion is a fixed 3x3 matrix, so it is differentiable
and gradients pass straight through it.
"""

from __future__ import annotations

import torch

# Rows: how much R, G, B go into Y, Cb, Cr (see the spike notebook for the derivation).
RGB_TO_YCBCR = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)
YCBCR_TO_RGB = torch.linalg.inv(RGB_TO_YCBCR)
CHROMA_OFFSET = torch.tensor([0.0, 0.5, 0.5]).view(1, 3, 1, 1)


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", RGB_TO_YCBCR.to(x), x) + CHROMA_OFFSET.to(x)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", YCBCR_TO_RGB.to(x), x - CHROMA_OFFSET.to(x))


def fit_to_gamut(x_rgb: torch.Tensor, delta_ycc: torch.Tensor) -> torch.Tensor:
    """Per-pixel scale t in [0, 1] so that adding t * delta_ycc keeps every pixel valid RGB.

    x_rgb:     (B, 3, H, W) clean image in [0, 1]
    delta_ycc: (B, 3, H, W) noise in YCbCr units
    returns:   (B, 1, H, W) scale t, with t = 1 wherever no shrinking is needed, such that
               ycbcr_to_rgb(rgb_to_ycbcr(x_rgb) + t * delta_ycc) lies in [0, 1].

    Idea: each pixel's noise is a 3-D arrow. Shrinking it along its own direction never makes
    any channel bigger, so the Y/Cb/Cr budget still holds. The clean pixel is valid and the set
    of valid RGB colors is a cube (convex), so a fitting t always exists.
    """
    # The amount of noise applied to each pixel.
    delta_rgb = torch.einsum("ij,bjhw->bihw", YCBCR_TO_RGB.to(delta_ycc), delta_ycc)

    # For each channel, the fraction of its noise that fits before it leaves [0,1]:
    # Positive noise(pushing up) has 1 - pixel_value of room limit,
    # negative noise(pushing down) pixel_value of room limit,
    # and zero noise(no push) has inf room so there's no limit.
    # We shrink the pixel's RGB channels to smallest value, a 3 to 1 dimensional reduction.
    # Finally we cap noise_scale at 1 so the noise is only ever shrunk, never grown.
    noise_scale = (
        torch.where(delta_rgb > 0, (1 - x_rgb) / delta_rgb.abs(), x_rgb / delta_rgb.abs())
        .nan_to_num(nan=torch.inf)
        .amin(dim=1, keepdim=True)
        .clamp(max=1)
    )
    return noise_scale
