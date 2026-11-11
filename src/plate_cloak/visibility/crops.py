"""Clean and attacked crops of each counted plate, the same region in both images.

A crop is the plate box grown by the attack margin (the region the noise may touch), plus
CONTEXT_PX of untouched surroundings on each side, so the edge of the noise sits inside the crop
instead of on its border. Crops are clipped at the image edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from plate_cloak.data.eval_images import load_by_id

CONTEXT_PX = 16


@dataclass
class PlateCrop:
    image_id: str
    plate_idx: int  # the plate's row in the label file, as in per_plate.csv
    clean: Image.Image
    adv: Image.Image


def crop_box(
    box: np.ndarray, margin: float, context: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) in whole pixels: the box grown like plate_mask, plus context."""
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, math.floor(x1 - dx) - context),
        max(0, math.floor(y1 - dy) - context),
        min(w, math.ceil(x2 + dx) + context),
        min(h, math.ceil(y2 + dy) + context),
    )


def plate_crops(cfg: DictConfig, adv_dir: Path, image_id: str) -> list[PlateCrop]:
    """Crops of every counted (not ignored) plate in one image, clean and attacked."""
    ev = load_by_id(cfg, image_id)
    adv = Image.open(Path(adv_dir) / f"{image_id}.png").convert("RGB")
    if adv.size != ev.image.size:
        raise RuntimeError(f"{image_id}: attacked image is {adv.size}, clean {ev.image.size}")
    w, h = ev.image.size
    crops = []
    for idx in np.flatnonzero(~ev.ignore):
        region = crop_box(ev.boxes[idx], cfg.attack.margin, CONTEXT_PX, w, h)
        crops.append(PlateCrop(image_id, int(idx), ev.image.crop(region), adv.crop(region)))
    return crops
