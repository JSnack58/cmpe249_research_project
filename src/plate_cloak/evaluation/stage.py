"""Evaluate stage: show every target model the clean and the attacked images, and score them.

    python -m plate_cloak evaluate --config configs/base.yaml [--variant V] [--name N] [k=v ...]
    python -m plate_cloak run      (same options: attack, then evaluate)

Output: results/runs/<name>/ holding
    config.yaml        the resolved settings (`--config` on it reproduces the run)
    per_detection.csv.gz  image_id, model, seed, condition, score, is_tp: every kept detection
                       (gzipped: RT-DETR keeps all 300 queries above the floor, so a full run
                       has millions of rows; pandas.read_csv reads it directly)
    per_plate.csv      image_id, plate_idx, model, seed, condition, found: every counted plate
    metrics.json       per target: clean and attacked mAP50 and recall50, and their drops,
                       each as a bootstrap CI; plus the attack's fingerprint and audit maxima

Predictions keep everything above a 0.001 floor so AP sees the full ranking; recall counts a
plate as found only at `eval.conf`. Rows are in the same image order for every model and
condition, and every CI uses the same resamples (same seed), so differences between models or
conditions are paired: each resample compares them on the same images.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from plate_cloak.attack.stage import fingerprint
from plate_cloak.data.eval_images import eval_image_ids, load_by_id
from plate_cloak.evaluation.detection import Pooled
from plate_cloak.evaluation.matching import ImageMatch, match_image
from plate_cloak.evaluation.stats import bootstrap_ci
from plate_cloak.runs import check_run_name, create_run_dir, provenance, write_resolved_config

CONF_FLOOR = 0.001  # keep near-misses: AP ranks every detection, recall applies eval.conf later
CONDITIONS = ("clean", "attacked")
DETECTION_COLUMNS = ["image_id", "model", "seed", "condition", "score", "is_tp"]
PLATE_COLUMNS = ["image_id", "plate_idx", "model", "seed", "condition", "found"]


def predict_boxes(
    model, image: Image.Image, imgsz: int = 640, conf_floor: float = CONF_FLOOR, device=None
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 4) x1 y1 x2 y2 boxes and (N,) scores from an Ultralytics model.

    `image` must be a PIL RGB image: Ultralytics reads a NumPy array as BGR, so an RGB array would
    silently swap the red and blue channels. Every detector here has one class (plate).
    """
    result = model.predict(source=image, imgsz=imgsz, conf=conf_floor, verbose=False, device=device)
    boxes = result[0].boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4)
    return xyxy, boxes.conf.cpu().numpy().astype(np.float64).reshape(-1)


def load_target(name: str, seed: int):
    """A fine-tuned target model (a separate function so tests can swap in stand-ins)."""
    from plate_cloak.models.zoo import load_detector

    return load_detector(name, seed)


def evaluate(cfg: DictConfig, adv_dir: Path, run_dir: Path | None = None) -> dict:
    """Score every target on clean and attacked images; return metrics.json's content.

    If `run_dir` is given it must not exist yet. It is created, and every output written into it,
    only after all the work succeeds, so a crash never leaves a half-written run behind.
    """
    adv_dir = Path(adv_dir)
    if run_dir is not None:
        run_dir = Path(run_dir)
        check_run_name(run_dir.name)
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} already exists; pick another --name")
    ids = eval_image_ids(cfg)
    done = _check_cache(cfg, adv_dir, ids)

    targets = [(t.name, t.seed) for t in cfg.models.targets]
    models = {t: load_target(*t) for t in targets}
    matches = {(t, c): [] for t in targets for c in CONDITIONS}
    det_rows, plate_rows = [], []
    for n, image_id in enumerate(ids, start=1):
        ev = load_by_id(cfg, image_id)
        adv = Image.open(adv_dir / f"{image_id}.png").convert("RGB")
        if adv.size != ev.image.size:
            raise RuntimeError(f"{image_id}: attacked image is {adv.size}, clean {ev.image.size}")
        counted_idx = np.flatnonzero(~ev.ignore)  # label row of each counted plate
        for (name, seed), model in models.items():
            for condition, image in zip(CONDITIONS, (ev.image, adv), strict=True):
                boxes, scores = predict_boxes(model, image, cfg.data.imgsz, device=cfg.device)
                m = match_image(boxes, scores, ev.boxes, ev.ignore, cfg.eval.iou, cfg.eval.conf)
                matches[((name, seed), condition)].append(m)
                key = {"image_id": image_id, "model": name, "seed": seed, "condition": condition}
                det_rows += [
                    {**key, "score": f"{s:.6f}", "is_tp": int(tp)}
                    for s, tp in zip(m.scores, m.is_tp, strict=True)
                ]
                plate_rows += [
                    {**key, "plate_idx": int(idx), "found": int(f)}
                    for idx, f in zip(counted_idx, m.gt_found, strict=True)
                ]
        if n % 50 == 0 or n == len(ids):
            print(f"evaluate: {n}/{len(ids)} images")

    metrics = {
        "n_images": len(ids),
        "attack": {"fingerprint": adv_dir.name, "max_change": done["max_change"]},
        "eval": OmegaConf.to_container(cfg.eval, resolve=True),
        "targets": {
            f"{name}_s{seed}": {
                "name": name,
                "seed": seed,
                **_target_metrics(
                    matches[((name, seed), "clean")], matches[((name, seed), "attacked")], cfg
                ),
            }
            for name, seed in targets
        },
        "provenance": provenance(),
    }
    if run_dir is not None:
        create_run_dir(run_dir.parent, run_dir.name)
        write_resolved_config(cfg, run_dir)
        _write_csv(run_dir / "per_detection.csv.gz", DETECTION_COLUMNS, det_rows)
        _write_csv(run_dir / "per_plate.csv", PLATE_COLUMNS, plate_rows)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"evaluate: wrote {run_dir}")
    return metrics


def _target_metrics(clean: list[ImageMatch], attacked: list[ImageMatch], cfg: DictConfig) -> dict:
    """mAP50 and recall50 per condition, and clean minus attacked, as bootstrap CIs."""
    c, a = Pooled(clean), Pooled(attacked)

    def ci(statistic):
        b = cfg.eval.bootstrap
        return bootstrap_ci(len(clean), statistic, b.n, b.ci, cfg.seed).as_dict()

    return {
        "clean": {"map50": ci(c.ap), "recall50": ci(c.recall)},
        "attacked": {"map50": ci(a.ap), "recall50": ci(a.recall)},
        "drop_map50": ci(lambda w: c.ap(w) - a.ap(w)),
        "drop_recall50": ci(lambda w: c.recall(w) - a.recall(w)),
    }


def _check_cache(cfg: DictConfig, adv_dir: Path, ids: list[str]) -> dict:
    """Refuse an attack cache that is unfinished or was made with other settings."""
    expected = fingerprint(cfg, ids)
    if adv_dir.name != expected:
        raise RuntimeError(
            f"{adv_dir} was made with different attack settings or images "
            f"(fingerprint {adv_dir.name}, this config gives {expected})"
        )
    done = adv_dir / "done.json"
    if not done.exists():
        raise RuntimeError(f"{adv_dir} is not finished (no done.json); rerun the attack stage")
    return json.loads(done.read_text())


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write rows as CSV; gzip it if the name ends in .gz (with no timestamp, so identical
    results give identical bytes and git sees no change)."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    data = buf.getvalue().encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
