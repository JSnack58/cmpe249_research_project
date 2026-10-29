"""The detectors in this study and where their weights live.

Fine-tuned weights:  weights/<name>_s<seed>.pt  (+ <name>_s<seed>.json summary)
COCO starting weights: weights/init/<file>      (downloaded by Ultralytics on first use)
Both folders are git-ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from plate_cloak.config import resolve_path

WEIGHTS_DIR = resolve_path("weights")


@dataclass(frozen=True)
class ModelSpec:
    family: Literal["yolo", "rtdetr"]  # which Ultralytics class loads it
    init: str  # COCO-pretrained checkpoint that fine-tuning starts from


ZOO: dict[str, ModelSpec] = {
    "yolov8n": ModelSpec("yolo", "yolov8n.pt"),  # A3: small surrogate
    "yolov8s": ModelSpec("yolo", "yolov8s.pt"),  # main surrogate (white-box)
    "yolov8l": ModelSpec("yolo", "yolov8l.pt"),  # A3: large surrogate
    "yolov10l": ModelSpec("yolo", "yolov10l.pt"),  # CNN transfer control, NMS-free
    "rtdetr-l": ModelSpec("rtdetr", "rtdetr-l.pt"),  # Transformer transfer target
}


def spec(name: str) -> ModelSpec:
    if name not in ZOO:
        raise KeyError(f"unknown model {name!r}; options: {sorted(ZOO)}")
    return ZOO[name]


def weights_path(name: str, seed: int) -> Path:
    spec(name)  # validate the name
    return WEIGHTS_DIR / f"{name}_s{seed}.pt"


def init_weights_path(name: str) -> Path:
    return WEIGHTS_DIR / "init" / spec(name).init


def model_class(family: str):
    """The Ultralytics class for a family (imported lazily: ultralytics is slow to import)."""
    from ultralytics import RTDETR, YOLO

    return RTDETR if family == "rtdetr" else YOLO


def load_detector(name: str, seed: int):
    """Load a fine-tuned detector as an Ultralytics model object."""
    path = weights_path(name, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run: python -m plate_cloak finetune --model {name} --seed {seed}"
        )
    return model_class(spec(name).family)(str(path))
