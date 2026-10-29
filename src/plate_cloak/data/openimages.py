"""Open Images V7 "Vehicle registration plate" boxes, downloaded into a YOLO-format dataset.

Layout written under `out`:
  annotations/{train,validation,test}_plates.csv   raw Open Images rows for the plate class
  images/{train,val,test}/<ImageID>.jpg
  labels/{train,val,test}/<ImageID>.txt            YOLO lines: class cx cy w h (normalized)
  data.yaml                                        dataset description for Ultralytics training

Idempotent: files that already exist are kept, so an interrupted run can simply be restarted.
Checked 2026-10-28: none of the 8,154 images carries an EXIF rotation tag and Open Images lists
rotation 0 for all val/test images, so images are stored as downloaded.
Adapted from spikes/02_openimages/download_openimages_plates.py.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

PLATE_LABEL = "/m/01jfm_"  # Open Images class ID for "Vehicle registration plate"
BBOX_CSV = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
YOLO_SPLIT = {"train": "train", "validation": "val", "test": "test"}
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"

Box = tuple[float, float, float, float]  # normalized x1, y1, x2, y2


def prepare_openimages(out: Path, workers: int = 32) -> Path:
    """Download plate images and labels for every split; return the path of data.yaml."""
    out = Path(out)
    for split, yolo_split in YOLO_SPLIT.items():
        boxes = _plate_boxes(split, out / "annotations" / f"{split}_plates.csv")
        img_dir = out / "images" / yolo_split
        lbl_dir = out / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        ids = sorted(boxes)
        fetch = partial(_download, split, img_dir)
        with ThreadPoolExecutor(workers) as pool:
            ok = list(pool.map(fetch, ids))
        for image_id, got in zip(ids, ok, strict=True):
            if got:  # images that are gone from the server get no label either
                lines = yolo_label_lines(boxes[image_id])
                (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
        print(f"{split}: {sum(ok)}/{len(ids)} images")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: license_plate\n"
    )
    return data_yaml


def yolo_label_lines(boxes: Iterable[Box]) -> list[str]:
    """Normalized (x1, y1, x2, y2) boxes to YOLO label lines 'class cx cy w h'."""
    return [
        f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
        for x1, y1, x2, y2 in boxes
    ]


def _plate_boxes(split: str, cache: Path) -> dict[str, list[Box]]:
    """Plate boxes per image ID; streams the split's CSV once (train is ~2 GB) and caches rows."""
    if not cache.exists():
        rows = []
        with urllib.request.urlopen(BBOX_CSV[split]) as resp:
            for row in csv.reader(io.TextIOWrapper(resp, encoding="utf-8")):
                if len(row) > 2 and row[2] == PLATE_LABEL:
                    rows.append(row)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

    boxes: dict[str, list[Box]] = defaultdict(list)
    for row in csv.reader(cache.open()):
        x1, x2, y1, y2 = map(float, row[4:8])  # Open Images column order: XMin, XMax, YMin, YMax
        boxes[row[0]].append((x1, y1, x2, y2))
    return boxes


def _download(split: str, img_dir: Path, image_id: str) -> bool:
    dest = img_dir / f"{image_id}.jpg"
    if dest.exists():
        return True
    try:
        url = IMAGE_URL.format(split=split, image_id=image_id)
        data = urllib.request.urlopen(url, timeout=30).read()
    except OSError:
        return False
    dest.write_bytes(data)
    return True
