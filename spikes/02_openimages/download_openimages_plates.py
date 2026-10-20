"""Download Open Images V7 "Vehicle registration plate" images and write a YOLO-format dataset.

Usage:  python spikes/02_openimages/download_openimages_plates.py [--out data/openimages_plates]

Layout written:
  <out>/annotations/{train,validation,test}_plates.csv   raw Open Images boxes for the plate class
  <out>/images/{train,val,test}/<ImageID>.jpg
  <out>/labels/{train,val,test}/<ImageID>.txt            YOLO: class cx cy w h (normalized)
  <out>/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PLATE_MID = "/m/01jfm_"
BBOX_CSV = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
YOLO_SPLIT = {"train": "train", "validation": "val", "test": "test"}
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"


def fetch_plate_rows(split: str, dest: Path) -> list[list[str]]:
    """Stream the split's bbox CSV and keep only plate rows (the train CSV is ~2 GB)."""
    if dest.exists():
        return list(csv.reader(dest.open()))
    rows = []
    with urllib.request.urlopen(BBOX_CSV[split]) as resp:
        for row in csv.reader(io.TextIOWrapper(resp, encoding="utf-8")):
            if len(row) > 2 and row[2] == PLATE_MID:
                rows.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return rows


def download(split: str, image_id: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        data = urllib.request.urlopen(IMAGE_URL.format(split=split, image_id=image_id), timeout=30).read()
    except OSError:
        return False
    dest.write_bytes(data)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/openimages_plates"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    for split, yolo_split in YOLO_SPLIT.items():
        rows = fetch_plate_rows(split, args.out / "annotations" / f"{split}_plates.csv")
        boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for r in rows:
            x1, x2, y1, y2 = map(float, r[4:8])
            boxes[r[0]].append((x1, y1, x2, y2))

        img_dir = args.out / "images" / yolo_split
        lbl_dir = args.out / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        ids = sorted(boxes)
        with ThreadPoolExecutor(args.workers) as pool:
            ok = list(pool.map(lambda i: download(split, i, img_dir / f"{i}.jpg"), ids))

        for image_id, got in zip(ids, ok, strict=True):
            if not got:
                continue
            lines = [
                f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                for x1, y1, x2, y2 in boxes[image_id]
            ]
            (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
        print(f"{split}: {sum(ok)}/{len(ids)} images, {len(rows)} boxes")

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: license_plate\n"
    )


if __name__ == "__main__":
    main()
