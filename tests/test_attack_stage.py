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
    monkeypatch.setattr(stage, "load_surrogates", lambda cfg: [TinyDetector()])
    monkeypatch.setattr(stage, "weights_digests", lambda cfg: {"yolov8s_s0": calls["digest"]})
    return calls


# --- fingerprint ----------------------------------------------------------------------------


def test_fingerprint_changes_with_attack_settings_and_images():
    base = load_config(BASE_CONFIG)
    f = fingerprint(base, ["a", "b"])
    assert fingerprint(load_config(BASE_CONFIG, ["attack.eps.cb=8"]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["attack.loss.type=hinge"]), ["a", "b"]) != f
    other = "models.surrogates=[{name: yolov8s, seed: 1}]"
    assert fingerprint(load_config(BASE_CONFIG, [other]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["data.imgsz=320"]), ["a", "b"]) != f
    assert fingerprint(load_config(BASE_CONFIG, ["seed=7"]), ["a", "b"]) != f
    assert fingerprint(base, ["a"]) != f
    assert len(f) == 12 and int(f, 16) >= 0


def test_fingerprint_ignores_eval_and_output_settings():
    f = fingerprint(load_config(BASE_CONFIG), ["a"])
    for override in ["eval.conf=0.5", "eval.bootstrap.n=10", "output.runs_root=x", "device=cpu"]:
        assert fingerprint(load_config(BASE_CONFIG, [override]), ["a"]) == f, override


def fp(*overrides):
    return fingerprint(load_config(BASE_CONFIG, list(overrides)), ["a", "b"])


def test_fingerprint_ignores_budgets_the_space_does_not_use():
    # An RGB attack never reads Cb/Cr, so changing them must not invalidate its cache.
    assert fp("attack.space=rgb") == fp("attack.space=rgb", "attack.eps.cb=8", "attack.eps.y=1")
    assert fp("attack.space=rgb") == fp("attack.space=rgb", "attack.quant_margin=0.3")
    assert fp() == fp("attack.eps.rgb=4")  # base is YCbCr
    assert fp() != fp("attack.quant_margin=0.3")
    assert fp() == fp("attack.loss.floor=0.2")  # top-k has no floor
    assert fp("attack.loss.type=hinge") != fp("attack.loss.type=hinge", "attack.loss.floor=0.2")
    assert fp("attack.loss.type=hinge") == fp("attack.loss.type=hinge", "attack.loss.k=10")


def test_fingerprint_changes_with_surrogates_split_momentum_and_diversity():
    two = "models.surrogates=[{name: yolov8s, seed: 0}, {name: yolov8n, seed: 0}]"
    for override in [two, "data.split=val", "attack.momentum=1.0", "attack.diversity.p=0.5"]:
        assert fp(override) != fp(), override
    assert fp("attack.diversity.min_scale=0.8") == fp()  # DI is off, so its scale is unused
    assert fp("attack.diversity.p=0.5", "attack.diversity.min_scale=0.8") != fp(
        "attack.diversity.p=0.5"
    )


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
    assert done["n_images"] == 2 and done["max_change"]["Y"] <= 3.0
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


def test_cache_from_other_weights_is_refused(tmp_path, fake_surrogate, monkeypatch):
    cfg = tiny_cfg(
        tmp_path, "models.surrogates=[{name: yolov8s, seed: 0}, {name: yolov8n, seed: 0}]"
    )
    monkeypatch.setattr(stage, "load_surrogates", lambda cfg: [TinyDetector(0), TinyDetector(1)])
    digests = {"yolov8s_s0": "weights-v1", "yolov8n_s0": "n-v1"}
    monkeypatch.setattr(stage, "weights_digests", lambda cfg: dict(digests))
    run_attack(cfg)
    assert [s["weights_sha256"] for s in _settings(cfg)["surrogates"]] == ["weights-v1", "n-v1"]
    digests["yolov8n_s0"] = "n-v2"  # one member of the ensemble was retrained
    with pytest.raises(RuntimeError, match="different surrogate weights"):
        run_attack(cfg)


def test_budget_violation_is_never_marked_done(tmp_path, fake_surrogate, monkeypatch):
    cfg = tiny_cfg(tmp_path)

    def bad_pgd(nets, x, boxes, s, generator=None):
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


def _settings(cfg):
    import yaml

    return yaml.safe_load((run_attack(cfg) / "settings.yaml").read_text())


def test_diversity_draws_depend_on_the_image_not_the_order(tmp_path, fake_surrogate):
    # A resumed or subsampled run must redo an image with the same random draws.
    both = run_attack(tiny_cfg(tmp_path, "attack.diversity.p=0.5"))
    alone = run_attack(tiny_cfg(tmp_path, "attack.diversity.p=0.5", "data.n_eval=1"))
    image_id = (alone / "audit.csv").read_text().splitlines()[1].split(",")[0]
    assert alone != both
    assert (alone / f"{image_id}.png").read_bytes() == (both / f"{image_id}.png").read_bytes()
