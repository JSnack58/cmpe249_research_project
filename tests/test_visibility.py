"""Plate crops and per-plate LPIPS (milestone 2a, Task 8)."""

import json

import numpy as np
import pytest
from PIL import Image

import plate_cloak.visibility.lpips_stage as lpips_stage
from plate_cloak import cli
from plate_cloak.attack.stage import fingerprint
from plate_cloak.config import BASE_CONFIG, load_config
from plate_cloak.visibility.crops import CONTEXT_PX, crop_box, plate_crops
from plate_cloak.visibility.lpips_stage import lpips_scores

IDS = ["img_a", "img_b"]
# img_a: one plate in the middle; img_b: a plate in the top-left corner and one too small to count.
LABELS = {"img_a": "0 0.5 0.5 0.5 0.3\n", "img_b": "0 0.12 0.12 0.24 0.2\n0 0.8 0.8 0.02 0.02\n"}


def overrides(tmp_path):
    root = tmp_path / "data"
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    for i, image_id in enumerate(IDS):
        pixels = np.random.default_rng(i).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / "test" / f"{image_id}.jpg")
        (root / "labels" / "test" / f"{image_id}.txt").write_text(LABELS[image_id])
    eligible = tmp_path / "eligible.txt"
    eligible.write_text("\n".join(IDS) + "\n")
    return [
        f"data.root={root}",
        f"data.eligible_list={eligible}",
        f"output.cache_root={tmp_path / 'cache'}",
        "data.imgsz=160",
        "data.min_plate_w=20",
        "data.min_plate_h=10",
    ]


def fake_cache(cfg, noise=8):
    """A finished attack cache: every image with +noise/255 everywhere (crops only see plates)."""
    from plate_cloak.config import resolve_path
    from plate_cloak.data.eval_images import load_by_id

    out = resolve_path(cfg.output.cache_root) / fingerprint(cfg, IDS)
    out.mkdir(parents=True)
    for image_id in IDS:
        clean = np.array(load_by_id(cfg, image_id).image).astype(int)
        Image.fromarray(np.clip(clean + noise, 0, 255).astype(np.uint8)).save(
            out / f"{image_id}.png"
        )
    (out / "done.json").write_text("{}")
    return out


def test_crop_box_grows_and_clips():
    # 40 x 20 box, margin 10% (4 px, 2 px), then 16 px of context on every side.
    assert crop_box(np.array([60.0, 50.0, 100.0, 70.0]), 0.1, CONTEXT_PX, 160, 120) == (
        40,
        32,
        120,
        88,
    )
    assert crop_box(np.array([2.0, 3.0, 40.0, 25.0]), 0.1, CONTEXT_PX, 160, 120)[:2] == (0, 0)
    assert crop_box(np.array([130.0, 100.0, 158.0, 119.0]), 0.1, 16, 160, 120)[2:] == (160, 120)


def test_crops_cover_counted_plates_and_match_in_size(tmp_path):
    cfg = load_config(BASE_CONFIG, overrides(tmp_path))
    adv_dir = fake_cache(cfg)
    crops = plate_crops(cfg, adv_dir, "img_b")
    assert [c.plate_idx for c in crops] == [0]  # the tiny second plate is not counted
    (c,) = crops
    # The corner plate's crop is clipped at the image edge; both crops are still the same region.
    assert c.clean.size == c.adv.size
    diff = np.array(c.adv).astype(int) - np.array(c.clean).astype(int)
    assert diff.max() <= 8 and diff.max() > 0


@pytest.mark.slow
def test_lpips_is_zero_for_identical_crops_and_positive_for_noise(tmp_path):
    cfg = load_config(BASE_CONFIG, overrides(tmp_path))
    adv_dir = fake_cache(cfg)
    noisy = plate_crops(cfg, adv_dir, "img_a")
    same = [type(c)(c.image_id, c.plate_idx, c.clean, c.clean) for c in noisy]
    tiny = Image.new("RGB", (20, 20))
    small = [type(noisy[0])("x", 0, tiny, tiny)]
    scores = lpips_scores(same + noisy + small, "cpu")
    assert scores[0] < 1e-6 and scores[1] > 0 and np.isnan(scores[2])


def test_cli_visibility_writes_csv_and_summary(tmp_path, monkeypatch):
    cfg = load_config(BASE_CONFIG, overrides(tmp_path))
    adv_dir = fake_cache(cfg)
    monkeypatch.setattr(
        lpips_stage, "lpips_scores", lambda pairs, device: np.array([0.1, 0.3][: len(pairs)])
    )
    out = tmp_path / "vis"
    args = ["visibility", "--name", "v1", "--out", str(out), *overrides(tmp_path), "device=cpu"]
    assert cli.main(args) == 0
    rows = (out / "v1" / "lpips.csv").read_text().splitlines()
    assert rows[0] == "image_id,plate_idx,lpips" and len(rows) == 3
    summary = json.loads((out / "v1" / "summary.json").read_text())
    assert summary["n"] == 2 and summary["n_skipped"] == 0
    assert summary["mean"] == pytest.approx(0.1)  # 0.1 per image here (one plate each)
    assert summary["fingerprint"] == adv_dir.name
    assert summary["budget"] == {"y": 2, "cb": 12, "cr": 12}


def test_cli_visibility_refuses_an_unfinished_cache(tmp_path):
    cfg = load_config(BASE_CONFIG, overrides(tmp_path))
    (fake_cache(cfg) / "done.json").unlink()
    with pytest.raises(RuntimeError, match="not finished"):
        cli.main(
            ["visibility", "--name", "v1", "--out", str(tmp_path / "vis"), *overrides(tmp_path)]
        )
