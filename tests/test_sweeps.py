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
    # Same attack settings as the m1 RGB run (base.yaml with space=rgb), so it reuses that cache.
    from plate_cloak.attack.stage import fingerprint

    m1 = load_config(BASE_CONFIG, overrides=["attack.space=rgb"])
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
