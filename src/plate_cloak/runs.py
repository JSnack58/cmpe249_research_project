"""Run folders: one per experiment run, never overwritten.

results/runs/<name>/ holds the resolved config.yaml (enough to reproduce the run) plus the
metrics and per-image results the evaluate stage writes.
Names are generated from the timestamp and the settings that differ from base.yaml, or given
explicitly with --name.
"""

from __future__ import annotations

import platform
import re
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

MAX_NAME_LEN = 80
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def run_name(cfg: DictConfig, base: DictConfig, now: datetime) -> str:
    """'YYYY-MM-DD_HHMM' plus 'key-value' for each setting that differs from base."""
    diffs = _flatten(cfg).items() - _flatten(base).items()
    parts = [now.strftime("%Y-%m-%d_%H%M")]
    parts += [_UNSAFE.sub("", f"{key}-{value}") for key, value in sorted(diffs)]
    return "_".join(parts)[:MAX_NAME_LEN]


def create_run_dir(root: Path, name: str) -> Path:
    """Create root/name; refuse if it exists, so earlier results can never be overwritten."""
    if not name or _UNSAFE.search(name) or name.startswith("."):
        raise ValueError(f"run name {name!r} may only use letters, digits, '.', '_', '-'")
    run_dir = Path(root) / name
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()  # raises FileExistsError if the run already exists
    return run_dir


def write_resolved_config(cfg: DictConfig, run_dir: Path) -> None:
    """Save the exact settings used, so `--config <run_dir>/config.yaml` reproduces the run."""
    OmegaConf.save(cfg, Path(run_dir) / "config.yaml", resolve=True)


def provenance() -> dict[str, Any]:
    """Where a result came from: code version, library versions, hardware, time."""
    import torch  # imported here so loading this module stays fast

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": version("torch"),
        "ultralytics": version("ultralytics"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _flatten(cfg: DictConfig) -> dict[str, str]:
    """{'attack.eps.cb': '6', ...}; values as compact strings so lists compare and print cleanly."""
    flat = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        flat.update(_flatten_value(key, value))
    return flat


def _flatten_value(path: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            out.update(_flatten_value(f"{path}.{key}", sub))
        return out
    return {path: str(value).replace(" ", "")}


def _git(*args: str) -> str:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
