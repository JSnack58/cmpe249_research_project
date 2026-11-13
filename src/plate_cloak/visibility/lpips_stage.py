"""Visibility stage: LPIPS between each clean and attacked plate crop.

    python -m plate_cloak visibility --config SWEEP --variant V --name N [--out DIR] [k=v ...]
    python -m plate_cloak visibility --cache DIR --name N [--out DIR] [k=v ...]

Reads the attack cache that the config's settings point to (it must be finished), or with --cache
an explicit cache folder on the test split (milestone 1's caches, whose run configs predate the
current schema; the attack and data settings come from the cache's settings.yaml), and writes
<out>/<name>/ holding
    lpips.csv      image_id, plate_idx, lpips: one row per counted plate (empty if skipped)
    summary.json   n, n_skipped, mean, median, p90, the budget and the cache fingerprint

LPIPS (Zhang et al. 2018, AlexNet features, the paper's default) is a learned distance between
two images that tracks how different people judge them to look; 0 means identical. Crops with a
side under MIN_SIDE pixels are skipped (the network's features are too coarse for them).
"""

from __future__ import annotations

import csv
import functools
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from plate_cloak.attack.stage import attack_settings, fingerprint
from plate_cloak.config import BASE_CONFIG, load_config, resolve_path
from plate_cloak.data.eval_images import eval_image_ids
from plate_cloak.progress import Progress
from plate_cloak.runs import check_run_name
from plate_cloak.visibility.crops import PlateCrop, plate_crops

MIN_SIDE = 32
DEFAULT_OUT = "results/visibility"


@functools.cache
def _model(device: str):
    import lpips  # imported here: it loads torchvision's AlexNet weights on first use

    with warnings.catch_warnings():
        # lpips 0.1.4 (its latest release) still calls torchvision with the old `pretrained=`
        # argument; the weights it gets are the same, so these two notices are only noise.
        warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated", UserWarning)
        warnings.filterwarnings("ignore", "Arguments other than a weight enum", UserWarning)
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def lpips_scores(pairs: list[PlateCrop], device) -> np.ndarray:
    """LPIPS of each (clean, attacked) crop pair; NaN where a side is under MIN_SIDE pixels."""
    scores = np.full(len(pairs), np.nan)
    for i, pair in enumerate(pairs):
        if min(pair.clean.size) < MIN_SIDE:
            continue
        a, b = (_tensor(img, device) for img in (pair.clean, pair.adv))
        with torch.no_grad():
            scores[i] = float(_model(str(device))(a, b))
    return scores


def config_for_cache(adv_dir: Path, overrides: list[str] | tuple[str, ...] = ()) -> DictConfig:
    """base.yaml on the test split, with the cache's own attack and data settings.

    Refuses a cache whose image count differs from the images this config selects, so the crops
    are taken from the images the cache was attacked on.
    """
    settings = OmegaConf.load(Path(adv_dir) / "settings.yaml")
    flat = {"data.split": "test"}
    for section in ("attack", "data"):
        for key, value in _flat(OmegaConf.to_container(settings[section]), section).items():
            flat[key] = value
    pairs = [*overrides, *(f"{k}={_yaml(v)}" for k, v in flat.items())]
    cfg = load_config(BASE_CONFIG, pairs)
    n = len(eval_image_ids(cfg))
    if n != settings.n_images:
        raise ValueError(f"{adv_dir} holds {settings.n_images} images, but the config selects {n}")
    return cfg


def run_visibility(
    cfg: DictConfig, name: str, out_root: Path | None = None, adv_dir: Path | None = None
) -> Path:
    """Score every counted plate of the config's images; return the output folder.

    `adv_dir` names the cache folder; by default it is the one the config's settings point to.
    """
    check_run_name(name)
    out = Path(out_root or resolve_path(DEFAULT_OUT)) / name
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick another --name")
    ids = eval_image_ids(cfg)
    if adv_dir is None:
        adv_dir = resolve_path(cfg.output.cache_root) / fingerprint(cfg, ids)
    adv_dir = Path(adv_dir)
    if not (adv_dir / "done.json").exists():
        raise RuntimeError(f"{adv_dir} is not finished (no done.json); run the attack first")

    rows = []
    bar = Progress(len(ids), "lpips")
    for image_id in ids:
        pairs = plate_crops(cfg, adv_dir, image_id)
        for pair, score in zip(pairs, lpips_scores(pairs, cfg.device), strict=True):
            rows.append((pair.image_id, pair.plate_idx, score))
        bar.update()

    scores = np.array([r[2] for r in rows], dtype=float)
    kept = scores[~np.isnan(scores)]
    summary = {
        "n": int(kept.size),
        "n_skipped": int(np.isnan(scores).sum()),
        "mean": float(kept.mean()) if kept.size else None,
        "median": float(np.median(kept)) if kept.size else None,
        "p90": float(np.percentile(kept, 90)) if kept.size else None,
        "budget": attack_settings(cfg.attack)["eps"],
        "fingerprint": adv_dir.name,
    }
    out.mkdir(parents=True)
    with (out / "lpips.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "plate_idx", "lpips"])
        writer.writerows((i, p, "" if np.isnan(s) else f"{s:.6f}") for i, p, s in rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    mean = "n/a" if summary["mean"] is None else f"{summary['mean']:.4f}"
    print(f"visibility: wrote {out}  (LPIPS mean {mean} over {summary['n']} plates)")
    return out


def _flat(d: dict, prefix: str) -> dict:
    out = {}
    for key, value in d.items():
        path = f"{prefix}.{key}"
        out.update(_flat(value, path) if isinstance(value, dict) else {path: value})
    return out


def _yaml(value) -> str:
    return json.dumps(value)  # a valid YAML scalar for the override parser


def _tensor(img, device) -> torch.Tensor:
    """(1, 3, H, W) in [-1, 1], the range LPIPS expects."""
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
    return (x / 127.5 - 1).to(device)
