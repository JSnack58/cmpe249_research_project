"""Fine-tune one detector on the plate dataset with Ultralytics.

Every model gets the same data, image size and epoch count, early stopping off, and each
family's default Ultralytics optimizer settings (RT-DETR does not train well with YOLO's).
Training output goes to runs/finetune/<name>_s<seed>/ (git-ignored); the best-validation
checkpoint is copied to weights/<name>_s<seed>.pt with a JSON summary next to it.

Crash-safe: an interrupted run resumes from its last checkpoint; a finished run is only exported.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from plate_cloak.config import resolve_path
from plate_cloak.models import zoo
from plate_cloak.models.zoo import model_class

RUNS_DIR = resolve_path("runs/finetune")
IMGSZ = 640
FINISHED = -1  # Ultralytics sets a checkpoint's epoch to -1 once training completes


def finetune(
    name: str,
    seed: int,
    data_yaml: Path,
    epochs: int,
    batch: int,
    workers: int,
    device: str,
    timing: bool = False,
) -> Path:
    """Train (or resume, or just export) `name` with `seed`; return the exported weights path.

    timing=True is a short throwaway run for measuring speed: it trains in runs/timing/ and is
    never exported, so it cannot be mistaken for (or block) the real fine-tune.
    """
    out = zoo.weights_path(name, seed)
    if not timing and out.exists() and out.with_suffix(".json").exists():
        print(f"{out.name} already exists; delete it to retrain")
        return out

    run_name = f"{name}_s{seed}"
    runs_dir = RUNS_DIR.parent / "timing" if timing else RUNS_DIR
    run_dir = runs_dir / run_name
    last = run_dir / "weights" / "last.pt"
    cls = model_class(zoo.spec(name).family)

    if timing:
        shutil.rmtree(run_dir, ignore_errors=True)  # always a fresh measurement
        model = cls(str(_init(name)))
        model.train(
            **_train_args(data_yaml, epochs, batch, workers, device, seed, runs_dir, run_name)
        )
        return _report_timing(run_dir / "results.csv", run_name, epochs)

    if last.exists() and checkpoint_epoch(last) == FINISHED:
        print(f"{run_name}: training already finished; exporting")
        model = cls(str(run_dir / "weights" / "best.pt"))
    elif last.exists():
        print(f"{run_name}: resuming from {last}")
        model = cls(str(last))
        model.train(resume=True)
    else:
        model = cls(str(_init(name)))
        model.train(
            **_train_args(data_yaml, epochs, batch, workers, device, seed, runs_dir, run_name)
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "weights" / "best.pt", out)
    summary = {
        "model": name,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        **_best_epoch(run_dir / "results.csv"),
        "train_dir": str(run_dir),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{run_name}: best val mAP@0.5 {summary['best_val_map50']:.3f} -> {out}")
    return out


def _init(name: str) -> Path:
    """COCO starting weights; Ultralytics downloads them to weights/init/ on first use."""
    init = zoo.init_weights_path(name)
    init.parent.mkdir(parents=True, exist_ok=True)
    return init


def _train_args(data_yaml, epochs, batch, workers, device, seed, runs_dir, run_name) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "device": device,
        "imgsz": IMGSZ,
        "seed": seed,
        "deterministic": True,
        "patience": 0,  # 0 disables early stopping, so every model trains the same epochs
        "project": str(runs_dir),
        "name": run_name,
        "exist_ok": True,
    }


def _report_timing(results_csv: Path, run_name: str, epochs: int) -> Path:
    """Seconds per epoch from Ultralytics' cumulative 'time' column; the first epoch is warm-up."""
    with results_csv.open() as f:
        times = [float(row["time"]) for row in csv.DictReader(f, skipinitialspace=True)]
    per_epoch = times[-1] - times[-2] if len(times) > 1 else times[-1]
    print(
        f"{run_name}: {epochs} epochs in {times[-1]:.0f}s; steady state {per_epoch:.0f}s/epoch "
        f"-> 100 epochs ~ {per_epoch * 100 / 3600:.1f} h"
    )
    return results_csv.parent


def checkpoint_epoch(path: Path) -> int:
    """Epoch stored in an Ultralytics checkpoint (-1 once training has finished)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False).get("epoch", 0)


def _best_epoch(results_csv: Path) -> dict[str, Any]:
    """Best validation mAP@0.5 from Ultralytics' per-epoch results.csv."""
    with results_csv.open() as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return {
        "best_val_map50": float(best["metrics/mAP50(B)"]),
        "best_val_map50_95": float(best["metrics/mAP50-95(B)"]),
        "best_epoch": int(float(best["epoch"])),
        "epochs_run": len(rows),
    }
