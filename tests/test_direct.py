"""Milestone 2b analysis over several direct-attack runs: the strength check and H1-direct."""

import json

import pandas as pd
import pytest
from test_report import write_synthetic_run

from plate_cloak import cli
from plate_cloak.analysis.direct import h1_direct, strength_check, transfer_matrix, white_box
from plate_cloak.analysis.report import load_run

CNN, TRANSFORMER = "yolov10l", "rtdetr-l"


def direct_run(root, model, seed, drop, targets=None):
    """A run attacking (model, seed) with its own weights; by default shown only to itself."""
    me = (model, seed)
    return write_synthetic_run(
        root / f"{model}_s{seed}",
        drops={model: drop},
        targets=targets or [me],
        surrogates=[me],
    )


def direct_runs(root, cnn_drop, transformer_drop, seeds=(0, 1, 2)):
    return [direct_run(root, CNN, s, cnn_drop) for s in seeds] + [
        direct_run(root, TRANSFORMER, s, transformer_drop) for s in seeds
    ]


# --- white_box -------------------------------------------------------------------------------


def test_white_box_is_the_single_surrogate(tmp_path):
    targets = [(CNN, 1), (TRANSFORMER, 0)]
    run = direct_run(tmp_path, CNN, 1, 0.5, targets=targets)
    assert white_box(load_run(run)) == (CNN, 1)


def test_white_box_refuses_two_surrogates(tmp_path):
    two = [(CNN, 0), (CNN, 1)]
    run = write_synthetic_run(tmp_path / "r", drops={CNN: 0.5}, targets=two, surrogates=two)
    with pytest.raises(ValueError, match="one surrogate"):
        white_box(load_run(run))


def test_white_box_refuses_a_surrogate_that_is_not_a_target(tmp_path):
    run = write_synthetic_run(
        tmp_path / "r", drops={CNN: 0.5}, targets=[(CNN, 1)], surrogates=[(CNN, 0)]
    )
    with pytest.raises(ValueError, match="not one of its targets"):
        white_box(load_run(run))


# --- strength check --------------------------------------------------------------------------


def test_strength_check_passes_at_the_bar_point_estimate(tmp_path):
    cnn, tf = direct_run(tmp_path, CNN, 0, 0.8), direct_run(tmp_path, TRANSFORMER, 0, 0.3)
    point = strength_check([cnn])[0]["drop"]["point"]
    rows = strength_check([cnn, tf], bar=point)
    assert [(r["model"], r["seed"]) for r in rows] == [(CNN, 0), (TRANSFORMER, 0)]
    assert rows[0]["passed"] and not rows[1]["passed"]  # exactly at the bar passes
    assert rows[0]["drop"]["low"] < point < rows[0]["drop"]["high"]  # the CI does not matter
    assert not strength_check([cnn], bar=point + 1e-9)[0]["passed"]


def test_strength_check_default_bar_is_the_owners_0_70(tmp_path):
    run = direct_run(tmp_path, CNN, 0, 0.75)
    assert strength_check([run])[0]["passed"]
    assert not strength_check([run], bar=0.80)[0]["passed"]


# --- H1-direct -------------------------------------------------------------------------------


def test_h1_direct_supported_when_cnn_drops_much_more(tmp_path):
    v = h1_direct(direct_runs(tmp_path, 0.9, 0.1))
    assert v["seeds"] == [0, 1, 2]
    assert v["cnn_drop"]["point"] == pytest.approx(0.9, abs=0.01)
    assert v["transformer_drop"]["point"] == pytest.approx(0.1, abs=0.01)
    assert v["difference"]["point"] == pytest.approx(0.8, abs=0.02)
    assert v["supported"] and v["threshold"] == 0.35
    assert set(v["attack_success"]) == {f"{m}_s{s}" for m in (CNN, TRANSFORMER) for s in (0, 1, 2)}


def test_h1_direct_not_supported_in_the_mirror_case(tmp_path):
    v = h1_direct(direct_runs(tmp_path, 0.1, 0.9))
    assert v["difference"]["point"] == pytest.approx(-0.8, abs=0.02)
    assert not v["supported"]


def test_h1_direct_not_supported_below_the_threshold(tmp_path):
    v = h1_direct(direct_runs(tmp_path, 0.5, 0.4))
    assert v["difference"]["low"] > 0 and not v["supported"]


def test_h1_direct_refuses_runs_over_different_images(tmp_path):
    runs = direct_runs(tmp_path, 0.9, 0.1)
    for name in ("per_plate.csv", "per_detection.csv.gz"):
        table = pd.read_csv(runs[-1] / name)
        table["image_id"] = table["image_id"].str.replace("img000", "imgXXX")
        table.to_csv(runs[-1] / name, index=False)
    with pytest.raises(ValueError, match="same images"):
        h1_direct(runs)


def test_h1_direct_refuses_a_repeated_model_seed(tmp_path):
    runs = direct_runs(tmp_path, 0.9, 0.1)
    again = direct_run(tmp_path / "again", CNN, 0, 0.9)
    with pytest.raises(ValueError, match="twice"):
        h1_direct([*runs, again])


def test_h1_direct_refuses_unequal_seeds(tmp_path):
    runs = [direct_run(tmp_path, CNN, s, 0.9) for s in (0, 1, 2)]
    runs += [direct_run(tmp_path, TRANSFORMER, s, 0.1) for s in (0, 1)]
    with pytest.raises(ValueError, match="same seeds"):
        h1_direct(runs)


def test_h1_direct_bootstrap_is_shared_across_runs(tmp_path):
    # Identical drops on identical images: every resample's difference is exactly 0.
    v = h1_direct(direct_runs(tmp_path, 0.5, 0.5))
    assert (v["difference"]["low"], v["difference"]["high"]) == (0.0, 0.0)
    assert v["cnn_drop"]["low"] < v["cnn_drop"]["high"]


# --- transfer matrix and CLI -----------------------------------------------------------------


def test_transfer_matrix_rows_are_attacked_models_columns_every_target(tmp_path):
    targets = [(CNN, 0), (TRANSFORMER, 0)]
    runs = [
        write_synthetic_run(
            tmp_path / m, drops={CNN: d1, TRANSFORMER: d2}, targets=targets, surrogates=[(m, 0)]
        )
        for m, d1, d2 in ((CNN, 0.6, 0.1), (TRANSFORMER, 0.05, 0.5))
    ]
    t = transfer_matrix(runs)
    assert list(t.index) == [f"{CNN}_s0", f"{TRANSFORMER}_s0"]
    assert list(t.columns) == [f"{CNN}_s0", f"{TRANSFORMER}_s0"]
    assert t.loc[f"{CNN}_s0", f"{TRANSFORMER}_s0"] == pytest.approx(0.1, abs=0.01)
    assert t.loc[f"{TRANSFORMER}_s0", f"{TRANSFORMER}_s0"] == pytest.approx(0.5, abs=0.01)


def test_cli_direct_writes_both_reports(tmp_path):
    runs = [str(r) for r in direct_runs(tmp_path / "runs", 0.9, 0.1)]
    out = tmp_path / "out"
    assert cli.main(["direct", "strength", "--runs", *runs, "--out", str(out)]) == 0
    assert cli.main(["direct", "verdict", "--runs", *runs, "--out", str(out)]) == 0
    assert len(json.loads((out / "strength.json").read_text())) == 6
    assert json.loads((out / "h1_direct.json").read_text())["verdict"]["supported"]
    md = (out / "h1_direct.md").read_text()
    assert "H1-direct supported" in md and "Transfer" in md
    assert "| yolov10l_s0 |" in (out / "strength.md").read_text()
