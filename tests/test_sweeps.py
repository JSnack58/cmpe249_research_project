"""Checks on the committed sweep files: each variant changes one factor and nothing else."""

import yaml

from plate_cloak.config import BASE_CONFIG, CONFIG_DIR, load_config, variants
from plate_cloak.runs import _flatten

CBCR = CONFIG_DIR / "sweeps" / "cbcr.yaml"
SURROGATE_ONLY = [{"name": "yolov8s", "seed": 0}]


def cbcr(variant):
    return load_config(CBCR, variant=variant)


def changed_keys(cfg):
    """Dotted keys whose value differs from base.yaml."""
    base = _flatten(load_config(BASE_CONFIG))
    return {k for k, v in _flatten(cfg).items() if base[k] != v}


def test_cbcr_has_the_five_planned_variants_in_order():
    assert variants(CBCR) == ["rgb8", "cb4", "cb8", "cb12", "random8"]


def test_cbcr_only_shows_the_attacks_to_the_surrogate():
    # The budget rule only needs the white-box model's recall, which keeps the sweep cheap.
    for v in variants(CBCR):
        assert [dict(t) for t in cbcr(v).models.targets] == SURROGATE_ONLY, v


def test_rgb8_is_pgd_in_rgb_at_8():
    a = cbcr("rgb8").attack
    assert (a.method, a.space, a.eps.rgb) == ("pgd", "rgb", 8)


def test_cb_variants_are_pgd_in_ycbcr_with_y2_and_equal_cb_cr():
    for v, value in [("cb4", 4), ("cb8", 8), ("cb12", 12)]:
        a = cbcr(v).attack
        assert (a.method, a.space) == ("pgd", "ycbcr"), v
        assert (a.eps.y, a.eps.cb, a.eps.cr) == (2, value, value), v


def test_random8_is_the_noise_control_at_the_rgb8_budget():
    a = cbcr("random8").attack
    assert (a.method, a.space, a.eps.rgb) == ("random", "rgb", 8)


def test_variants_change_only_the_attack_budget_space_and_method():
    allowed = {
        "models.targets",
        "attack.method",
        "attack.space",
        "attack.eps.rgb",
        "attack.eps.y",
        "attack.eps.cb",
        "attack.eps.cr",
    }
    for v in variants(CBCR):
        assert changed_keys(cbcr(v)) <= allowed, (v, changed_keys(cbcr(v)) - allowed)


def test_one_factor_between_neighbouring_variants():
    def diff(a, b):
        fa, fb = _flatten(cbcr(a)), _flatten(cbcr(b))
        return {k for k in fa if fa[k] != fb[k]}

    assert diff("cb4", "cb8") == diff("cb8", "cb12") == {"attack.eps.cb", "attack.eps.cr"}
    assert diff("rgb8", "random8") == {"attack.method"}


def test_every_variant_sets_its_own_budget_instead_of_inheriting_base():
    # Task 12 later writes the chosen Cb/Cr into base.yaml. A variant that relied on base's value
    # would then silently change, and rerunning the sweep would no longer reproduce it.
    raw = yaml.safe_load(CBCR.read_text())
    shared = raw.get("overrides") or {}
    for v, settings in raw["variants"].items():
        keys = {**shared, **(settings or {})}
        needed = {"attack.method", "attack.space"}
        needed |= {"attack.eps.rgb"} if keys.get("attack.space") == "rgb" else set()
        if keys.get("attack.space") == "ycbcr":
            needed |= {"attack.eps.y", "attack.eps.cb", "attack.eps.cr"}
        assert needed <= keys.keys(), (v, needed - keys.keys())


LOSS = CONFIG_DIR / "sweeps" / "loss.yaml"


def test_loss_sweep_pairs_each_loss_with_the_rgb8_and_cb8_attacks():
    assert variants(LOSS) == ["topk_rgb8", "hinge_rgb8", "topk_cb8", "hinge_cb8"]
    for budget in ("rgb8", "cb8"):
        topk, hinge = (load_config(LOSS, variant=f"{t}_{budget}") for t in ("topk", "hinge"))
        assert (topk.attack.loss.type, hinge.attack.loss.type) == ("topk", "hinge")
        ft, fh = _flatten(topk), _flatten(hinge)
        assert {k for k in ft if ft[k] != fh[k]} == {"attack.loss.type"}  # one factor
        # Same attack as the matching cbcr variant, apart from the loss.
        ref = _flatten(cbcr(budget))
        assert {k for k in ft if ft[k] != ref[k]} <= {"attack.loss.type"}
    for v in variants(LOSS):
        assert [dict(t) for t in load_config(LOSS, variant=v).models.targets] == SURROGATE_ONLY


EPS = CONFIG_DIR / "sweeps" / "eps.yaml"


def test_eps_sweep_only_raises_the_rgb_budget_and_keeps_every_target():
    assert variants(EPS) == ["rgb8", "rgb16", "rgb32"]
    base = load_config(BASE_CONFIG)
    for v, value in [("rgb8", 8), ("rgb16", 16), ("rgb32", 32)]:
        cfg = load_config(EPS, variant=v)
        assert (cfg.attack.method, cfg.attack.space, cfg.attack.eps.rgb) == ("pgd", "rgb", value)
        assert cfg.models.targets == base.models.targets, v  # transfer needs every target
        assert changed_keys(cfg) <= {"attack.space", "attack.eps.rgb"}, v


def test_eps_rgb8_is_the_milestone_1_rgb_attack():
    # Same attack settings as the m1 RGB run (base.yaml as it was then, with space=rgb, which meant
    # RGB 8), so it reuses that cache.
    from plate_cloak.attack.stage import fingerprint

    m1 = load_config(BASE_CONFIG, overrides=["attack.space=rgb", "attack.eps.rgb=8"])
    eps8 = load_config(EPS, variant="rgb8")
    ids = ["a", "b"]  # the fingerprint also covers the image ids; a fixed list compares settings
    assert fingerprint(m1, ids) == fingerprint(eps8, ids)


VIS_CBCR = CONFIG_DIR / "sweeps" / "visibility_cbcr.yaml"
VIS_RGB = CONFIG_DIR / "sweeps" / "visibility_rgb.yaml"
VIS_Y = CONFIG_DIR / "sweeps" / "visibility_y.yaml"


def test_visibility_ladders_are_white_box_only_on_300_test_images():
    for sweep in (VIS_CBCR, VIS_RGB, VIS_Y):
        for v in variants(sweep):
            cfg = load_config(sweep, variant=v)
            assert [dict(t) for t in cfg.models.targets] == SURROGATE_ONLY, v
            assert [dict(m) for m in cfg.models.surrogates] == SURROGATE_ONLY, v
            assert (cfg.data.n_eval, cfg.data.split) == (300, "test"), v
            # Milestone 1's attack: no transfer techniques while the ceiling is chosen.
            assert (cfg.attack.momentum, cfg.attack.diversity.p) == (0.0, 0.0), v


def test_visibility_ladders_change_one_budget_per_rung():
    assert variants(VIS_CBCR) == ["cb4", "cb8", "cb12", "cb16", "cb20", "cb24"]
    for v in variants(VIS_CBCR):
        a = load_config(VIS_CBCR, variant=v).attack
        assert (a.method, a.space, a.eps.y) == ("pgd", "ycbcr", 2)
        assert a.eps.cb == a.eps.cr == int(v[2:])
    assert variants(VIS_RGB) == ["rgb2", "rgb4", "rgb6", "rgb8", "rgb10", "rgb12"]
    for v in variants(VIS_RGB):
        a = load_config(VIS_RGB, variant=v).attack
        assert (a.method, a.space, a.eps.rgb) == ("pgd", "rgb", int(v[3:]))


def test_y_ladder_raises_y_at_the_cbcr_ceiling():
    # The eye test put the Cb/Cr ceiling at 4 (sessions ceiling1 to ceiling3); Y 2 is its base rung.
    assert variants(VIS_Y) == ["y3", "y4"]
    for v in variants(VIS_Y):
        a = load_config(VIS_Y, variant=v).attack
        assert (a.method, a.space, a.eps.cb, a.eps.cr) == ("pgd", "ycbcr", 4, 4), v
        assert a.eps.y == int(v[1:]), v


TRANSFER_DEV = CONFIG_DIR / "sweeps" / "transfer_dev.yaml"
ENSEMBLE = [{"name": m, "seed": 0} for m in ("yolov8s", "yolov8n", "yolov8l")]


def test_transfer_dev_never_shows_rtdetr_and_uses_validation():
    assert variants(TRANSFER_DEV) == [
        "ens",
        "ens_di",
        "ens_mi",
        "ens_di_mi",
        "ens_di_mi_100",
        "ens_di_mi_y2",
    ]
    base = load_config(BASE_CONFIG).attack.eps
    raw = yaml.safe_load(TRANSFER_DEV.read_text())
    for v in variants(TRANSFER_DEV):
        cfg = load_config(TRANSFER_DEV, variant=v)
        assert cfg.data.split == "val", v
        assert [dict(m) for m in cfg.models.surrogates] == ENSEMBLE, v
        targets = [dict(t) for t in cfg.models.targets]
        assert not any(t["name"].startswith("rtdetr") for t in targets), v
        assert targets == [
            *ENSEMBLE,
            *({"name": "yolov8s", "seed": s} for s in (1, 2)),
            *({"name": "yolov10l", "seed": s} for s in (0, 1, 2)),
        ], v
        a = cfg.attack
        assert (a.method, a.space) == ("pgd", "ycbcr"), v
        # The frozen ceiling, except the step-down that the final eye test forced (checked below).
        y = 2 if v == "ens_di_mi_y2" else base.y
        assert (a.eps.y, a.eps.cb, a.eps.cr) == (y, base.cb, base.cr), v
    # The budget is written into the file, so a later base.yaml change cannot move it silently.
    shared = raw["overrides"]
    assert {"attack.eps.y", "attack.eps.cb", "attack.eps.cr", "attack.space"} <= shared.keys()
    techniques = {
        v: (
            load_config(TRANSFER_DEV, variant=v).attack.diversity.p,
            load_config(TRANSFER_DEV, variant=v).attack.momentum,
        )
        for v in variants(TRANSFER_DEV)
    }
    assert techniques == {
        "ens": (0.0, 0.0),
        "ens_di": (0.5, 0.0),
        "ens_mi": (0.0, 1.0),
        "ens_di_mi": (0.5, 1.0),
        "ens_di_mi_100": (0.5, 1.0),
        "ens_di_mi_y2": (0.5, 1.0),
    }


def test_transfer_dev_100_step_rerun_changes_only_the_steps_of_the_winner():
    # The validation winner was ens_di_mi (YOLOv10l drop 0.085, docs/experiment-decisions.md).
    fa = _flatten(load_config(TRANSFER_DEV, variant="ens_di_mi"))
    fb = _flatten(load_config(TRANSFER_DEV, variant="ens_di_mi_100"))
    assert {k for k in fa if fa[k] != fb[k]} == {"attack.steps"}
    pair = ("ens_di_mi", "ens_di_mi_100")
    steps = [load_config(TRANSFER_DEV, variant=v).attack.steps for v in pair]
    assert steps == [40, 100]


def test_transfer_dev_step_down_lowers_only_y_to_the_rung_below_the_ceiling():
    # ens_di_mi failed its own eye test (final_check, 28 of 40); the rung below the frozen ceiling
    # Y 3, Cb/Cr 4 is Y 2, Cb/Cr 4, which passed in ceiling3. The ablation is not redone.
    fa = _flatten(load_config(TRANSFER_DEV, variant="ens_di_mi"))
    fb = _flatten(load_config(TRANSFER_DEV, variant="ens_di_mi_y2"))
    assert {k for k in fa if fa[k] != fb[k]} == {"attack.eps.y"}
    a = load_config(TRANSFER_DEV, variant="ens_di_mi_y2").attack
    assert (a.eps.y, a.eps.cb, a.eps.cr) == (2, 4, 4)


TRANSFER_FINAL = CONFIG_DIR / "sweeps" / "transfer_final.yaml"


def test_transfer_final_is_the_checked_attack_on_the_test_set_with_every_target():
    assert variants(TRANSFER_FINAL) == ["ycbcr", "rgb"]
    base = load_config(BASE_CONFIG)
    checked = load_config(TRANSFER_DEV, variant="ens_di_mi_y2").attack  # passed final_check2
    for v in variants(TRANSFER_FINAL):
        cfg = load_config(TRANSFER_FINAL, variant=v)
        assert cfg.data.split == "test" and cfg.data.n_eval is None, v  # every test image, once
        assert [dict(m) for m in cfg.models.surrogates] == ENSEMBLE, v
        targets = [dict(t) for t in cfg.models.targets]
        assert targets == [*map(dict, base.models.targets), *ENSEMBLE[1:]], v  # with RT-DETR
        a = cfg.attack
        same = (a.method, a.steps, a.momentum, a.diversity.p, a.diversity.min_scale, a.loss.type)
        assert same == (
            checked.method,
            checked.steps,
            checked.momentum,
            checked.diversity.p,
            checked.diversity.min_scale,
            checked.loss.type,
        ), v
    ycbcr = load_config(TRANSFER_FINAL, variant="ycbcr").attack
    assert ycbcr.space == "ycbcr"
    assert (ycbcr.eps.y, ycbcr.eps.cb, ycbcr.eps.cr) == (
        checked.eps.y,
        checked.eps.cb,
        checked.eps.cr,
    )
    rgb = load_config(TRANSFER_FINAL, variant="rgb").attack
    assert (rgb.space, rgb.eps.rgb) == ("rgb", base.attack.eps.rgb)  # the frozen RGB ceiling, 2
    # Budgets are written into the file, not inherited from base.yaml.
    raw = yaml.safe_load(TRANSFER_FINAL.read_text())["variants"]
    assert {"attack.space", "attack.eps.y", "attack.eps.cb", "attack.eps.cr"} <= raw["ycbcr"].keys()
    assert {"attack.space", "attack.eps.rgb"} <= raw["rgb"].keys()


DIRECT_STRENGTH = CONFIG_DIR / "sweeps" / "direct_strength.yaml"
DIRECT_MODELS = {"rtdetr": "rtdetr-l", "v10l": "yolov10l"}


def test_direct_strength_attacks_each_model_with_its_own_weights_at_rgb16():
    names = [f"{m}_s{s}" for m in DIRECT_MODELS for s in (0, 1, 2)]
    assert variants(DIRECT_STRENGTH) == names
    raw = yaml.safe_load(DIRECT_STRENGTH.read_text())
    assert {"attack.space", "attack.eps.rgb"} <= raw["overrides"].keys()  # not inherited
    for v in names:
        cfg = load_config(DIRECT_STRENGTH, variant=v)
        short, seed = v.rsplit("_s", 1)
        model = [{"name": DIRECT_MODELS[short], "seed": int(seed)}]
        assert [dict(m) for m in cfg.models.surrogates] == model, v  # its own weights
        assert [dict(t) for t in cfg.models.targets] == model, v  # white-box only
        a = cfg.attack
        assert (a.method, a.space, a.eps.rgb, a.steps) == ("pgd", "rgb", 16, 40), v
        assert (a.momentum, a.diversity.p, a.loss.type) == (0.0, 0.0, "topk"), v
        assert (cfg.data.split, cfg.data.n_eval) == ("val", None), v
