import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from plate_cloak import cli
from plate_cloak.analysis.report import h1_table, h1_verdict, report
from plate_cloak.config import BASE_CONFIG, load_config

N_IMAGES = 120
DEFAULT_DROPS = {"yolov8s": 0.9, "yolov10l": 0.5, "rtdetr-l": 0.3}


def write_synthetic_run(
    run_dir,
    drops=None,
    seed_jitter=0.0,
    retrained_drop=0.2,
    targets=None,
    clean_misses=(),
):
    """A run folder in the evaluate stage's format, with known drops.

    Every image has one countable plate. Clean: every model finds every plate at score 0.9
    (except (model, seed, image) in `clean_misses`). Attacked: a model with drop d misses the
    first round(d * N_IMAGES) plates, so missed sets are nested across models (paired) and
    mAP50 and recall50 both fall by about d. yolov8s seeds other than 0 use `retrained_drop`;
    seed s of any other model adds seed_jitter * (s - 1).
    """
    drops = {**DEFAULT_DROPS, **(drops or {})}
    targets = targets or [(name, seed) for name in drops for seed in (0, 1, 2)]
    run_dir.mkdir(parents=True)
    target_cfg = ", ".join(f"{{name: {n}, seed: {s}}}" for n, s in targets)
    cfg = load_config(BASE_CONFIG, [f"models.targets=[{target_cfg}]", "eval.bootstrap.n=200"])
    OmegaConf.save(cfg, run_dir / "config.yaml")
    ids = [f"img{i:03d}" for i in range(N_IMAGES)]
    dets, plates = [], []
    for name, seed in targets:
        if name == "yolov8s":
            d = drops[name] if seed == 0 else retrained_drop
        else:
            d = drops[name] + seed_jitter * (seed - 1)
        n_missed = round(d * N_IMAGES)
        for condition in ("clean", "attacked"):
            for i, image_id in enumerate(ids):
                if condition == "clean":
                    hit = (name, seed, image_id) not in clean_misses
                else:
                    hit = i >= n_missed
                key = {"image_id": image_id, "model": name, "seed": seed, "condition": condition}
                plates.append({**key, "plate_idx": 0, "found": int(hit)})
                if hit:
                    dets.append({**key, "score": 0.9, "is_tp": 1})
    pd.DataFrame(dets).to_csv(run_dir / "per_detection.csv.gz", index=False)
    pd.DataFrame(plates).to_csv(run_dir / "per_plate.csv", index=False)
    return run_dir


# --- verdict ---------------------------------------------------------------------------------


def test_verdict_supported_when_rtdetr_resists(tmp_path):
    run = write_synthetic_run(tmp_path / "r", drops={"yolov10l": 0.6, "rtdetr-l": 0.1})
    v = h1_verdict(run)
    assert v["supported"] and v["difference"]["point"] == pytest.approx(0.5, abs=0.02)
    assert v["difference"]["low"] > 0


def test_verdict_not_supported_when_drops_equal(tmp_path):
    run = write_synthetic_run(tmp_path / "r", drops={"yolov10l": 0.4, "rtdetr-l": 0.4})
    v = h1_verdict(run)
    assert not v["supported"]
    # Same images missed by both models: every paired resample gives a difference of exactly 0.
    assert (v["difference"]["low"], v["difference"]["high"]) == (0.0, 0.0)


def test_verdict_needs_the_drop_to_reach_the_threshold(tmp_path):
    run = write_synthetic_run(tmp_path / "r", drops={"yolov10l": 0.5, "rtdetr-l": 0.3})
    v = h1_verdict(run)  # difference 0.2 is clearly above 0 but below 0.35
    assert v["difference"]["low"] > 0 and not v["supported"]
    assert h1_verdict(run, threshold=0.15)["supported"]


def test_attack_success_counts_only_plates_every_target_found_clean(tmp_path):
    # rtdetr-l seed 0 misses img000 even when clean, so img000 is left out for every model.
    # rtdetr-l misses img000..img035 when attacked (drop 0.3 of 120): 35 of the 119 kept plates.
    run = write_synthetic_run(tmp_path / "r", clean_misses={("rtdetr-l", 0, "img000")})
    v = h1_verdict(run)
    assert v["attack_success"]["n_plates"] == 119
    assert v["attack_success"]["rate"]["rtdetr-l"] == pytest.approx(35 / 119)


def test_verdict_is_none_without_both_architectures(tmp_path):
    run = write_synthetic_run(tmp_path / "r", targets=[("yolov8s", 0)])
    assert h1_verdict(run) is None


# --- table -----------------------------------------------------------------------------------


def test_table_rows_follow_the_roles_of_the_targets(tmp_path):
    t = h1_table(write_synthetic_run(tmp_path / "r"))
    assert list(t.index) == ["yolov8s (white-box)", "yolov8s (retrained)", "yolov10l", "rtdetr-l"]
    assert list(t["seeds"]) == ["0", "1,2", "0,1,2", "0,1,2"]
    assert t.loc["yolov8s (white-box)", "drop_map50"] == pytest.approx(0.9, abs=0.02)
    assert t.loc["yolov8s (retrained)", "drop_map50"] == pytest.approx(0.2, abs=0.02)
    assert t.loc["rtdetr-l", "clean_map50"] == pytest.approx(1.0)
    assert t.loc["rtdetr-l", "drop_recall50"] == pytest.approx(0.3, abs=0.01)
    row = t.loc["yolov10l"]
    assert row["drop_map50_low"] <= row["drop_map50"] <= row["drop_map50_high"]


def test_table_reports_seed_spread(tmp_path):
    run = write_synthetic_run(
        tmp_path / "r", drops={"yolov10l": 0.6, "rtdetr-l": 0.1}, seed_jitter=0.05
    )
    t = h1_table(run)
    assert {"drop_seed_min", "drop_seed_max"} <= set(t.columns)
    assert t.loc["yolov10l", "drop_seed_min"] == pytest.approx(0.55, abs=0.02)
    assert t.loc["yolov10l", "drop_seed_max"] == pytest.approx(0.65, abs=0.02)
    assert t.loc["yolov10l", "drop_map50"] == pytest.approx(0.6, abs=0.02)  # mean of the seeds


def test_table_for_a_sweep_run_with_one_target(tmp_path):
    t = h1_table(write_synthetic_run(tmp_path / "r", targets=[("yolov8s", 0)]))
    assert list(t.index) == ["yolov8s (white-box)"]


# --- report and CLI --------------------------------------------------------------------------


def test_report_writes_table_json_and_chart(tmp_path):
    main = write_synthetic_run(tmp_path / "runs" / "main", drops={"rtdetr-l": 0.1})
    sweep = write_synthetic_run(tmp_path / "runs" / "cb4", targets=[("yolov8s", 0)])
    out = tmp_path / "report"
    report([main, sweep], out)
    data = json.loads((out / "h1.json").read_text())
    assert set(data) == {"main", "cb4"}
    assert data["main"]["verdict"]["cnn_control"] == "yolov10l"
    assert data["cb4"]["verdict"] is None
    assert len(data["main"]["table"]) == 4
    md = (out / "h1_table.md").read_text()
    assert "## main" in md and "## cb4" in md and "rtdetr-l" in md and "—" not in md
    assert (out / "h1_bars.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_report(tmp_path):
    run = write_synthetic_run(tmp_path / "runs" / "main")
    out = tmp_path / "report"
    assert cli.main(["report", "--runs", str(run), "--out", str(out)]) == 0
    assert (out / "h1.json").exists()


def test_seed_mean_uses_the_same_resamples_for_every_seed(tmp_path):
    # All three rtdetr-l seeds identical: the seed mean must equal any single seed's CI exactly.
    run = write_synthetic_run(tmp_path / "r", seed_jitter=0.0)
    t = h1_table(run)
    row = t.loc["rtdetr-l"]
    assert row["drop_seed_min"] == row["drop_seed_max"] == pytest.approx(row["drop_map50"])
    assert np.isfinite(row["drop_map50_low"])
