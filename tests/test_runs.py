import re
from datetime import datetime

import pytest

from plate_cloak.config import BASE_CONFIG, load_config
from plate_cloak.runs import create_run_dir, provenance, run_name, write_resolved_config

NOW = datetime(2026, 11, 2, 14, 32)


def test_run_name_lists_differences():
    cfg = load_config(BASE_CONFIG, overrides=["attack.eps.cb=6"])
    assert run_name(cfg, load_config(BASE_CONFIG), NOW) == "2026-11-02_1432_attack.eps.cb-6"


def test_run_name_without_differences_is_timestamp():
    base = load_config(BASE_CONFIG)
    assert run_name(base, base, NOW) == "2026-11-02_1432"


def test_run_name_is_filesystem_safe_and_bounded():
    cfg = load_config(
        BASE_CONFIG,
        overrides=[
            "models.targets=[{name: rtdetr-l, seed: 0}]",
            "attack.eps.cb=6",
            "attack.steps=7",
        ],
    )
    name = run_name(cfg, load_config(BASE_CONFIG), NOW)
    assert len(name) <= 80
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name)


def test_existing_run_dir_refuses(tmp_path):
    create_run_dir(tmp_path, "r1")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "r1")


def test_run_dir_name_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, "../elsewhere")


def test_resolved_config_round_trips(tmp_path):
    cfg = load_config(BASE_CONFIG, overrides=["attack.eps.cb=6"])
    write_resolved_config(cfg, tmp_path)
    assert load_config(tmp_path / "config.yaml") == cfg


def test_provenance_records_code_and_environment():
    p = provenance()
    assert re.fullmatch(r"[0-9a-f]{40}", p["git_commit"])
    assert isinstance(p["git_dirty"], bool)
    assert {"timestamp", "python", "torch", "ultralytics", "gpu"} <= p.keys()
