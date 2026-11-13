"""H2 by its written rule: YCbCr hides plates from YOLOv8s about as well as RGB, at lower LPIPS."""

import json

import numpy as np
import pandas as pd
import pytest
from test_report import N_IMAGES, write_synthetic_run

from plate_cloak import cli
from plate_cloak.analysis.h2 import h2_chart, h2_verdict

SURROGATE = [("yolov8s", 0)]


def run(root, name, drop):
    return write_synthetic_run(root / name, drops={"yolov8s": drop}, targets=SURROGATE)


def lpips_csv(root, name, values, fingerprint):
    """lpips.csv and summary.json, as the visibility stage writes them, in a folder of their own."""
    rows = [{"image_id": f"img{i:03d}", "plate_idx": 0, "lpips": v} for i, v in enumerate(values)]
    (root / name).mkdir()
    path = root / name / "lpips.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (root / name / "summary.json").write_text(json.dumps({"fingerprint": fingerprint}))
    return path


def set_fingerprint(run_dir, fingerprint):
    (run_dir / "metrics.json").write_text(json.dumps({"attack": {"fingerprint": fingerprint}}))


def inputs(tmp_path, ycbcr_drop, rgb_drop, ycbcr_lpips, rgb_lpips):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.002, N_IMAGES)
    y, r = run(tmp_path, "ycbcr", ycbcr_drop), run(tmp_path, "rgb", rgb_drop)
    set_fingerprint(y, "yyyyyyyyyyyy")
    set_fingerprint(r, "rrrrrrrrrrrr")
    return (
        y,
        r,
        lpips_csv(tmp_path, "ycbcr_lpips", ycbcr_lpips + noise, "yyyyyyyyyyyy"),
        lpips_csv(tmp_path, "rgb_lpips", rgb_lpips + noise[::-1], "rrrrrrrrrrrr"),
    )


def test_h2_supported_when_recall_matches_and_lpips_is_lower(tmp_path):
    v = h2_verdict(*inputs(tmp_path, 0.9, 0.9, 0.02, 0.03))
    assert v["recall_difference"]["point"] == pytest.approx(0.0)
    assert v["recall_equivalent"] and v["less_visible"] and v["supported"]
    assert v["ycbcr_recall"]["point"] == pytest.approx(0.1, abs=0.01)
    assert v["lpips_difference"]["point"] == pytest.approx(-0.01, abs=0.001)
    assert v["n_plates_lpips"] == N_IMAGES and v["tolerance"] == 0.05


def test_h2_not_supported_when_lpips_is_higher(tmp_path):
    v = h2_verdict(*inputs(tmp_path, 0.9, 0.9, 0.04, 0.03))
    assert v["recall_equivalent"] and not v["less_visible"] and not v["supported"]


def test_h2_not_supported_when_the_recall_ci_leaves_the_band(tmp_path):
    # YCbCr misses 4% fewer plates: the point is inside 0.05 but the CI crosses it.
    v = h2_verdict(*inputs(tmp_path, 0.86, 0.9, 0.02, 0.03))
    assert abs(v["recall_difference"]["point"]) < 0.05 < v["recall_difference"]["high"]
    assert not v["recall_equivalent"] and v["less_visible"] and not v["supported"]


def test_h2_lpips_is_paired_over_plates_in_both_files(tmp_path):
    y, r, yl, rl = inputs(tmp_path, 0.9, 0.9, 0.02, 0.03)
    table = pd.read_csv(yl)
    table.loc[:9, "lpips"] = np.nan  # skipped crops (too small)
    table.to_csv(yl, index=False)
    assert h2_verdict(y, r, yl, rl)["n_plates_lpips"] == N_IMAGES - 10


def test_h2_refuses_swapped_lpips_files(tmp_path):
    y, r, yl, rl = inputs(tmp_path, 0.9, 0.9, 0.02, 0.03)
    with pytest.raises(ValueError, match="was not computed on"):
        h2_verdict(y, r, rl, yl)


def test_h2_refuses_lpips_that_misses_counted_plates(tmp_path):
    # A subsample (or another cache with the same fingerprint) must not be joined silently.
    y, r, yl, rl = inputs(tmp_path, 0.9, 0.9, 0.02, 0.03)
    pd.read_csv(yl).iloc[:60].to_csv(yl, index=False)
    with pytest.raises(ValueError, match="plates"):
        h2_verdict(y, r, yl, rl)


def test_h2_refuses_runs_over_different_images(tmp_path):
    y, r, yl, rl = inputs(tmp_path, 0.9, 0.9, 0.02, 0.03)
    for name in ("per_plate.csv", "per_detection.csv.gz"):
        table = pd.read_csv(r / name)
        table["image_id"] = table["image_id"].str.replace("img000", "imgXXX")
        table.to_csv(r / name, index=False)
    with pytest.raises(ValueError, match="same images"):
        h2_verdict(y, r, yl, rl)


def write_rung(root, ladder, rung, drop, lpips_mean):
    name = f"{ladder}_{rung}"
    run(root / "runs", name, drop)
    vis = root / "vis" / name
    vis.mkdir(parents=True)
    (vis / "summary.json").write_text(json.dumps({"mean": lpips_mean, "n": 10}))
    return vis


def test_h2_chart_marks_eye_test_results(tmp_path):
    dirs = [
        write_rung(tmp_path, "visibility_rgb", "rgb4", 0.6, 0.008),
        write_rung(tmp_path, "visibility_rgb", "rgb2", 0.3, 0.002),
        write_rung(tmp_path, "visibility_cbcr", "cb4", 0.4, 0.008),
    ]
    scores = tmp_path / "scores.json"
    budgets = {
        "visibility_rgb/rgb2": {"passed": True, "complete": True, "correct": 20, "n": 40},
        "visibility_rgb/rgb4": {"passed": False, "complete": True},
    }
    scores.write_text(json.dumps({"budgets": budgets}))
    out = tmp_path / "h2_chart.png"
    points = h2_chart(dirs, [scores], out, runs_root=tmp_path / "runs")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # Sorted by ladder, then from least visible up.
    assert [(p["ladder"], p["rung"], p["eye"]) for p in points] == [
        ("visibility_cbcr", "cb4", "untested"),
        ("visibility_rgb", "rgb2", "pass"),
        ("visibility_rgb", "rgb4", "fail"),
    ]
    assert points[1]["eye_score"] == "20/40"
    assert points[2]["recall"] == pytest.approx(0.4, abs=0.01)
    assert points[2]["lpips"] == 0.008


def test_cli_h2_writes_json_markdown_and_chart(tmp_path):
    y, r, yl, rl = inputs(tmp_path, 0.9, 0.9, 0.02, 0.03)
    out = tmp_path / "out"
    args = ["h2", "--ycbcr-run", str(y), "--rgb-run", str(r)]
    args += ["--ycbcr-lpips", str(yl), "--rgb-lpips", str(rl), "--out", str(out)]
    vis = write_rung(tmp_path, "visibility_rgb", "rgb2", 0.3, 0.002)
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"budgets": {}}))
    args += ["--chart-rungs", str(vis), "--eyetest-scores", str(scores)]
    args += ["--runs-root", str(tmp_path / "runs")]
    assert cli.main(args) == 0
    assert json.loads((out / "h2.json").read_text())["verdict"]["supported"]
    assert "H2 supported" in (out / "h2.md").read_text()
    assert (out / "h2_chart.png").exists()
