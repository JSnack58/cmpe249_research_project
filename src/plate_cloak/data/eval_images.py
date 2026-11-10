"""Evaluation images: the exact pictures and plate boxes every model is scored on.

Each image is resized so its long side is `imgsz` (640). That resized image is what gets attacked,
and both its clean and attacked versions are what every target model is shown. Images come from
`data.split`: the test split for reported results, the validation split for attack development.
Plates smaller than min_w x min_h at that size are kept as "ignore" regions: too small to read,
so a detection on one counts as neither a hit nor a false positive.
An image is eligible for evaluation if it has at least one plate that is not ignored.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageOps

from plate_cloak.config import resolve_path


@dataclass
class EvalImage:
    image_id: str
    image: Image.Image  # RGB, long side = imgsz
    boxes: np.ndarray  # (M, 4) float, x1 y1 x2 y2 in pixels of `image`
    ignore: np.ndarray  # (M,) bool, True = too small to count


def load_eval_image(
    image_path: Path, label_path: Path, imgsz: int, min_w: float, min_h: float
) -> EvalImage:
    """Load one image and its YOLO label, upright, RGB, resized to long side imgsz."""
    with Image.open(image_path) as raw:
        # exif_transpose applies any camera rotation tag so pixels match the labels.
        img = ImageOps.exif_transpose(raw).convert("RGB")
    scale = imgsz / max(img.size)
    width, height = round(img.width * scale), round(img.height * scale)
    img = img.resize((width, height), Image.Resampling.BILINEAR)

    rows = np.loadtxt(label_path, ndmin=2)  # class cx cy w h, normalized to [0, 1]
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes = np.clip(boxes, 0, [width, height, width, height])

    box_w, box_h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    ignore = (box_w < min_w) | (box_h < min_h)
    return EvalImage(Path(image_path).stem, img, boxes, ignore)


def is_eligible(ev: EvalImage) -> bool:
    return bool((~ev.ignore).any())


SPLITS = ("test", "val")  # val: attack development; test: the reported results


def write_eligible_list(
    data_root: Path,
    out_path: Path,
    imgsz: int,
    min_w: float,
    min_h: float,
    split: str = "test",
) -> int:
    """Freeze the sorted IDs of a split's eligible images to out_path; return how many there are."""
    _check_split(split)
    data_root = Path(data_root)
    eligible = []
    for image_path in sorted((data_root / "images" / split).glob("*.jpg")):
        label_path = data_root / "labels" / split / f"{image_path.stem}.txt"
        if is_eligible(load_eval_image(image_path, label_path, imgsz, min_w, min_h)):
            eligible.append(image_path.stem)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(eligible) + "\n")
    return len(eligible)


def eval_image_ids(cfg: DictConfig) -> list[str]:
    """IDs to evaluate: the frozen eligible list, optionally subsampled (seeded) to data.n_eval."""
    ids = resolve_path(cfg.data.eligible_list).read_text().split()
    if cfg.data.n_eval is not None and cfg.data.n_eval < len(ids):
        ids = sorted(random.Random(cfg.seed).sample(ids, cfg.data.n_eval))
    return ids


def load_by_id(cfg: DictConfig, image_id: str) -> EvalImage:
    """Load one image of the configured split (data.split) by ID."""
    split = _check_split(cfg.data.split)
    root = resolve_path(cfg.data.root)
    return load_eval_image(
        root / "images" / split / f"{image_id}.jpg",
        root / "labels" / split / f"{image_id}.txt",
        cfg.data.imgsz,
        cfg.data.min_plate_w,
        cfg.data.min_plate_h,
    )


def _check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown data split {split!r}; options: {', '.join(SPLITS)}")
    return split
