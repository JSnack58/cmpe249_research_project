import json

import numpy as np
import pytest
from detectors import TinyDetector
from PIL import Image

import plate_cloak.attack.stage as stage
from plate_cloak import cli
from plate_cloak.attack.stage import fingerprint, image_seed, run_attack
from plate_cloak.config import BASE_CONFIG, load_config

IDS = ["img_a", "img_b"]


def tiny_cfg(tmp_path, *overrides):
    """Two small fixture images with one plate each, a 3-step attack, on CPU."""
    root = tmp_path / "data"
    for i, image_id in enumerate(IDS):
        (root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(i)
        pixels = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / "test" / f"{image_id}.jpg")
        (root / "labels" / "test" / f"{image_id}.txt").write_text("0 0.5 0.5 0.5 0.3\n")
    eligible = tmp_path / "eligible.txt"
    eligible.write_text("\n".join(IDS) + "\n")
    return load_config(
        BASE_CONFIG,
        overrides=[
            f"data.root={root}",
            f"data.eligible_list={eligible}",
            f"output.cache_root={tmp_path / 'cache'}",
            "data.imgsz=160",
            "device=cpu",
            "attack.steps=3",
            *overrides,
        ],
    )


@pytest.fixture
def fake_surrogate(monkeypatch):
    """Patch in TinyDetector and a fixed weights digest; count how often PGD runs."""
    calls = {"pgd": 0, "digest": "weights-v1"}
    real_pgd = stage.pgd

    def counting_pgd(*args, **kwargs):
        calls["pgd"] += 1
        return real_pgd(*args, **kwargs)

    monkeypatch.setattr(stage, "pgd", counting_pgd)
    monkeypatch.setattr(stage, "load_surrogate", lambda cfg: TinyDetector())
    monkeypatch.setattr(stage, "weights_digest", lambda cfg: calls["digest"])
    return calls


# --- fingerprint ----------------------------------------------------------------------------


def test_fingerprint_changes_with_attack_settings_and_images():
    base = load_config(BASE_CONFIG)
    f = fingerprint(base, ["a", "b"])
    assert fingerprint(load_config(BASE_CONFIG, ["attack.eps.cb=4"]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["attack.loss.type=hinge"]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["models.surrogate.seed=1"]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["data.imgsz=320"]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["seed=7"]), ["a", "b"]) != f
    assert fingerprint(base, ["a"]) != f
    assert len(f) == 12 and int(f, 16) >= 0


def test_fingerprint_ignores_eval_and_output_settings():
    f = fingerprint(load_config(BASE_CONFIG), ["a"])
    for override in ["eval.conf=0.5", "eval.bootstrap.n=10", "output.runs_root=x", "device=cpu"]:
        assert fingerprint(load_config(BASE_CONFIG, [override]), ["a"]) == f, override


def test_image_seed_depends_on_seed_and_image_only():
    assert image_seed(42, "a") == image_seed(42, "a")
    assert len({image_seed(42, "a"), image_seed(42, "b"), image_seed(7, "a")}) == 3


# --- run_attack -----------------------------------------------------------------------------


def test_attack_writes_images_audit_and_done(tmp_path, fake_surrogate):
    cfg = tiny_cfg(tmp_path)
    out = run_attack(cfg)
    assert out.name == fingerprint(cfg, IDS)
    for image_id in IDS:
        adv = Image.open(out / f"{image_id}.png")
        assert adv.mode == "RGB" and adv.size == (160, 120)
    audit_rows = (out / "audit.csv").read_text().splitlines()
    assert audit_rows[0] == "image_id,R,G,B,Y,Cb,Cr" and len(audit_rows) == 3
    done = json.loads((out / "done.json").read_text())
    assert done["n_images"] == 2 and done["max_change"]["Y"] <= 2.0
    assert "weights-v1" in (out / "settings.yaml").read_text()
    assert not list(out.glob("*.tmp"))


def test_completed_cache_is_reused(tmp_path, fake_surrogate):
    cfg = tiny_cfg(tmp_path)
    first = run_attack(cfg)
    assert run_attack(cfg) == first
    assert fake_surrogate["pgd"] == 2  # second call did no work


def test_interrupted_cache_is_finished_without_redoing_saved_images(tmp_path, fake_surrogate):
    cfg = tiny_cfg(tmp_path)
    out = run_attack(cfg)
    saved = (out / "img_a.png").read_bytes()
    (out / "done.json").unlink()
    (out / "img_b.png").unlink()  # as if the machine crashed after the first image
    assert (run_attack(cfg) / "done.json").exists()
    assert fake_surrogate["pgd"] == 3  # only img_b was attacked again
    assert (out / "img_a.png").read_bytes() == saved


def test_cache_from_other_weights_is_refused(tmp_path, fake_surrogate):
    cfg = tiny_cfg(tmp_path)
    run_attack(cfg)
    fake_surrogate["digest"] = "weights-v2"  # the surrogate was retrained
    with pytest.raises(RuntimeError, match="different surrogate weights"):
        run_attack(cfg)


def test_budget_violation_is_never_marked_done(tmp_path, fake_surrogate, monkeypatch):
    cfg = tiny_cfg(tmp_path)

    def bad_pgd(net, x, boxes, s):
        return (x + 5 / 255).clamp(0, 1)  # moves Y by 5/255 everywhere: over the Y budget of 2

    monkeypatch.setattr(stage, "pgd", bad_pgd)
    with pytest.raises(RuntimeError, match="exceed the budget"):
        run_attack(cfg)
    assert not (tmp_path / "cache" / fingerprint(cfg, IDS) / "done.json").exists()


def test_random_control_is_reproducible(tmp_path, fake_surrogate):
    cfg = tiny_cfg(tmp_path, "attack.method=random")
    out = run_attack(cfg)
    first = (out / "img_a.png").read_bytes()
    (out / "done.json").unlink()
    (out / "img_a.png").unlink()
    run_attack(cfg)
    assert (out / "img_a.png").read_bytes() == first
    assert fake_surrogate["pgd"] == 0


def test_cli_attack_runs_a_sweep_variant(tmp_path, fake_surrogate, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text("id: s\nvariants:\n  cb4: {attack.eps.cb: 4, attack.eps.cr: 4}\n")
    overrides = [
        f"{k}={v}"
        for k, v in [
            ("data.root", cfg.data.root),
            ("data.eligible_list", cfg.data.eligible_list),
            ("output.cache_root", cfg.output.cache_root),
            ("data.imgsz", 160),
            ("device", "cpu"),
            ("attack.steps", 3),
        ]
    ]
    assert cli.main(["attack", "--config", str(sweep), "--variant", "cb4", *overrides]) == 0
    expected = load_config(sweep, overrides, variant="cb4")
    assert (tmp_path / "cache" / fingerprint(expected, IDS) / "done.json").exists()
