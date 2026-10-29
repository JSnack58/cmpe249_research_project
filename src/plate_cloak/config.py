"""Experiment configs: configs/base.yaml plus sweep variants and command-line overrides.

A config file is one of two kinds:
- a sweep file (has a top-level `variants:` key): applied on top of base.yaml, then one variant;
- a full config (base.yaml itself, or a run folder's resolved config.yaml): loaded as-is,
  and checked to have exactly the same keys as base.yaml so stale configs fail loudly.

Precedence, last wins: base.yaml < sweep `overrides:` < the chosen variant < command-line overrides.
Every override must name a key that already exists; typos raise KeyError instead of being ignored.
The returned config is read-only so no stage can change settings after they are recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"

_MISSING = object()


def load_config(
    config_path: Path,
    overrides: Iterable[str] = (),
    variant: str | None = None,
) -> DictConfig:
    """Load a config, apply a sweep variant and `key=value` overrides, and return it read-only."""
    config_path = Path(config_path)
    raw = OmegaConf.load(config_path)

    if "variants" in raw:
        options = variants(config_path)
        if variant is None:
            raise ValueError(f"{config_path} is a sweep file; pass a variant: {options}")
        if variant not in raw.variants:
            raise KeyError(f"{config_path} has no variant {variant!r}; options: {options}")
        cfg = OmegaConf.load(BASE_CONFIG)
        _apply(cfg, raw.get("overrides") or {}, source=f"{config_path} overrides")
        _apply(cfg, raw.variants[variant] or {}, source=f"{config_path} variant {variant}")
    else:
        if variant is not None:
            raise ValueError(f"{config_path} is not a sweep file, so it has no variant {variant!r}")
        cfg = raw
        _check_same_keys(cfg, OmegaConf.load(BASE_CONFIG), config_path)

    _apply(cfg, _parse_cli(overrides), source="command line")
    OmegaConf.set_readonly(cfg, True)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Config paths are relative to the repository root, so commands work from any directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def variants(config_path: Path) -> list[str]:
    """Variant names in a sweep file, in file order."""
    return list(OmegaConf.load(config_path).get("variants") or {})


def _parse_cli(overrides: Iterable[str]) -> dict[str, Any]:
    """Turn ["attack.eps.cb=6", ...] into {"attack.eps.cb": 6, ...}; values are parsed as YAML."""
    parsed = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} must look like key=value")
        parsed[key.strip()] = yaml.safe_load(value)
    return parsed


def _apply(cfg: DictConfig, updates: Mapping[str, Any], source: str) -> None:
    """Set dotted keys on cfg; every key must already exist in cfg."""
    for key, value in updates.items():
        if OmegaConf.select(cfg, key, default=_MISSING) is _MISSING:
            raise KeyError(f"unknown config key {key!r} (from {source})")
        OmegaConf.update(cfg, key, value, merge=False)


def _check_same_keys(cfg: DictConfig, base: DictConfig, path: Path) -> None:
    missing = _leaf_keys(base) - _leaf_keys(cfg)
    extra = _leaf_keys(cfg) - _leaf_keys(base)
    if missing or extra:
        raise KeyError(
            f"{path} does not match base.yaml: "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}"
        )


def _leaf_keys(cfg: DictConfig, prefix: str = "") -> set[str]:
    """Dotted paths of every leaf; lists count as leaves (their contents are values)."""
    keys = set()
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, DictConfig):
            keys |= _leaf_keys(value, prefix=f"{path}.")
        else:
            keys.add(path)
    return keys
