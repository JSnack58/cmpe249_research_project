import numpy as np
import pytest
import torch
import torch.nn.functional as F
from detectors import Scaled, TinyDetector

from plate_cloak.attack.losses import anchors_on_plates, hinge_loss, topk_loss
from plate_cloak.attack.pgd import (
    PAD_VALUE,
    AttackSettings,
    LossSettings,
    _apply,
    _bound,
    _project,
    audit,
    diverse,
    pgd,
    plate_mask,
    predict_padding,
    random_sign_noise,
    to_uint8,
)
from plate_cloak.config import BASE_CONFIG, load_config


def settings(
    space="rgb", eps=None, steps=10, loss="topk", method="pgd", momentum=0.0, diversity_p=0.0
):
    """AttackSettings from base.yaml with a few test-friendly changes."""
    s = AttackSettings.from_cfg(load_config(BASE_CONFIG).attack)
    if eps is None:
        eps = (8, 8, 8) if space == "rgb" else (2, 8, 8)
    return AttackSettings(
        method=method,
        space=space,
        eps=torch.tensor(eps, dtype=torch.float32) / 255,
        steps=steps,
        step_factor=s.step_factor,
        margin=s.margin,
        mask=s.mask,
        loss=LossSettings(loss, s.loss.k, s.loss.floor),
        quant_margin=s.quant_margin,
        momentum=momentum,
        diversity_p=diversity_p,
        diversity_min_scale=s.diversity_min_scale,
    )


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def fixture_image_with_plate(h=200, w=256, seed=0):
    """A random 8-bit image with one plate box (not a multiple of 32 high, like real images)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (1, 3, h, w), generator=g).float() / 255
    return x, np.array([[60.0, 70.0, 180.0, 120.0]])


def fixture_white_plate():
    """White plate background with black characters: the colors where Spike 01's budget leaked."""
    x, boxes = fixture_image_with_plate()
    x[..., 70:120, 60:180] = 1.0
    x[..., 85:105, 80:160:12] = 0.0  # thin black strokes
    return x, boxes


def raw_of(net, x):
    left, right, top, bottom = predict_padding(*x.shape[-2:])
    padded = torch.nn.functional.pad(x, (left, right, top, bottom), value=114 / 255)
    return net(padded)[0]


# --- settings -------------------------------------------------------------------------------


def test_settings_from_base_config():
    attack = load_config(BASE_CONFIG).attack
    s = AttackSettings.from_cfg(attack)
    assert s.space == "ycbcr" and s.method == "pgd" and s.steps == 40
    # Budgets come from the config in 1/255 units, in Y, Cb, Cr order.
    expected = torch.tensor([attack.eps.y, attack.eps.cb, attack.eps.cr], dtype=torch.float32)
    assert torch.allclose(s.eps * 255, expected)
    assert s.loss == LossSettings("topk", 100, 0.1)
    # The transfer techniques are off by default, so base.yaml is still milestone 1's attack.
    assert s.momentum == 0.0 and s.diversity_p == 0.0 and s.diversity_min_scale == 0.9


def test_settings_rgb_uses_one_budget_for_all_channels():
    cfg = load_config(BASE_CONFIG, overrides=["attack.space=rgb"])
    assert torch.allclose(AttackSettings.from_cfg(cfg.attack).eps * 255, torch.full((3,), 2.0))


def test_sobel_mask_is_not_silently_ignored():
    cfg = load_config(BASE_CONFIG, overrides=["attack.mask=sobel"])
    with pytest.raises(NotImplementedError):
        AttackSettings.from_cfg(cfg.attack)


# --- losses ---------------------------------------------------------------------------------


def test_topk_loss_sums_the_k_highest_scores():
    raw = torch.zeros(1, 5, 6)
    raw[0, 4] = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.0])
    assert torch.isclose(topk_loss(raw[0, 4], 2), torch.tensor(1.7))
    assert torch.isclose(topk_loss(raw[0, 4], 100), torch.tensor(2.3))  # k larger than A is fine


def test_hinge_loss_counts_only_plate_candidates_above_the_floor():
    raw = torch.zeros(1, 5, 4)
    raw[0, 4] = torch.tensor([0.5, 0.05, 0.9, 0.3])
    on_plate = torch.tensor([True, True, False, True])
    # 0.5 -> 0.4, 0.05 is below the floor, 0.9 is off the plate, 0.3 -> 0.2
    assert torch.isclose(hinge_loss(raw[0, 4], 0.1, on_plate), torch.tensor(0.6))
    assert torch.isclose(hinge_loss(raw[0, 4], 0.1, None), torch.tensor(1.4))


def test_anchors_on_plates_needs_real_overlap():
    raw = torch.zeros(1, 5, 3)
    raw[0, :4] = torch.tensor([[15, 15, 10, 10], [50, 50, 10, 10], [5, 5, 10, 10]]).T.float()
    plate = np.array([[10.0, 10.0, 30.0, 30.0]])  # candidate 2 only touches its corner
    assert anchors_on_plates(raw, plate).tolist() == [True, False, False]


# --- mask and padding -----------------------------------------------------------------------


def test_plate_mask_grows_boxes_and_stops_at_the_image_edge():
    m = plate_mask(100, 100, np.array([[20.0, 40.0, 40.0, 50.0], [90.0, 0.0, 100.0, 10.0]]), 0.1)
    assert m.shape == (1, 1, 100, 100)
    rows, cols = torch.nonzero(m[0, 0, :, :60], as_tuple=True)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (39, 50, 18, 41)  # +1 px, +2 px
    assert m[0, 0, 0, 99] == 1 and m[0, 0, 11, 99] == 0  # second box clipped at the edges
    assert plate_mask(10, 10, np.zeros((0, 4)), 0.1).sum() == 0  # no plates: empty mask


def test_padding_matches_ultralytics_predict():
    """Same call predict() makes for a 640 image (imgsz 640, minimum rectangle, stride 32)."""
    from ultralytics.data.augment import LetterBox

    for h, w in [(427, 640), (640, 480), (640, 640), (360, 640), (640, 33)]:
        img = np.full((h, w, 3), 255, np.uint8)
        out = LetterBox(new_shape=640, auto=True, stride=32)(image=img)
        rows = np.nonzero(out[:, :, 0].max(axis=1) == 255)[0]
        cols = np.nonzero(out[:, :, 0].max(axis=0) == 255)[0]
        left, right, top, bottom = predict_padding(h, w)
        assert out.shape[:2] == (h + top + bottom, w + left + right), (h, w)
        assert (rows[0], rows[-1], cols[0], cols[-1]) == (top, top + h - 1, left, left + w - 1)


# --- PGD ------------------------------------------------------------------------------------


@pytest.mark.parametrize("space", ["rgb", "ycbcr"])
def test_pgd_lowers_loss_and_stays_on_plates(space):
    x, boxes = fixture_image_with_plate()
    net = TinyDetector()
    adv = pgd([net], x, boxes, settings(space=space))
    assert topk_loss(raw_of(net, adv)[0, 4], 100) < topk_loss(raw_of(net, x)[0, 4], 100)
    outside = 1 - plate_mask(*x.shape[-2:], boxes, 0.10)
    assert ((adv - x).abs() * outside).max() == 0
    assert adv.min() >= 0 and adv.max() <= 1


def test_pgd_with_hinge_loss_lowers_it():
    x, boxes = fixture_image_with_plate()
    net = TinyDetector()
    adv = pgd([net], x, boxes, settings(loss="hinge"))

    def hinge(img):
        left, _, top, _ = predict_padding(*img.shape[-2:])
        raw = raw_of(net, img)
        return hinge_loss(raw[0, 4], 0.1, anchors_on_plates(raw, boxes + [left, top, left, top]))

    assert hinge(adv) < hinge(x)


def test_ycbcr_budget_holds_after_rounding_on_white_plate():
    x, boxes = fixture_white_plate()
    adv = pgd([TinyDetector()], x, boxes, settings(space="ycbcr", eps=(2, 8, 8), steps=40))
    a = audit(to_uint8(x), to_uint8(adv))
    assert a["Y"] <= 2.0 and a["Cb"] <= 8.0 and a["Cr"] <= 8.0
    assert a["Cb"] > 4.0 or a["Cr"] > 4.0  # the attack did use the chroma budget


def test_rgb_budget_holds_after_rounding():
    x, boxes = fixture_white_plate()
    a = audit(to_uint8(x), to_uint8(pgd([TinyDetector()], x, boxes, settings(space="rgb"))))
    assert max(a["R"], a["G"], a["B"]) <= 8.0


def test_non_multiple_of_32_input_round_trips_shape():
    x = torch.rand(1, 3, 427, 640)
    net = TinyDetector()
    adv = pgd([net], x, np.array([[100.0, 200.0, 300.0, 260.0]]), settings(steps=2))
    assert adv.shape == x.shape
    assert set(net.seen_shapes) == {(1, 3, 448, 640)}


def test_pgd_is_deterministic_and_works_under_no_grad():
    x, boxes = fixture_image_with_plate()
    first = pgd([TinyDetector()], x, boxes, settings(space="ycbcr"))
    with torch.no_grad():
        second = pgd([TinyDetector()], x, boxes, settings(space="ycbcr"))
    assert torch.equal(first, second)


def test_pgd_refuses_a_model_in_training_mode():
    x, boxes = fixture_image_with_plate()
    with pytest.raises(ValueError, match="eval mode"):
        pgd([TinyDetector().train()], x, boxes, settings())


# --- transfer techniques: ensemble, momentum, input diversity -------------------------------


def test_defaults_reproduce_plain_pgd():
    """One surrogate, no momentum, no diversity: exactly milestone 1's sign-gradient step."""
    x, boxes = fixture_image_with_plate()
    net, s = TinyDetector(), settings(space="ycbcr")
    left, right, top, bottom = predict_padding(*x.shape[-2:])
    eps = s.eps.view(1, 3, 1, 1)
    bound, step = _bound(x, boxes, s), s.step_factor * eps / s.steps
    delta = torch.zeros_like(x)
    for _ in range(s.steps):
        delta.requires_grad_(True)
        adv = _apply(x, delta, s.space)
        raw = net(F.pad(adv, (left, right, top, bottom), value=PAD_VALUE))[0]
        (grad,) = torch.autograd.grad(topk_loss(raw[0, 4], s.loss.k), delta)
        with torch.no_grad():
            delta = _project(x, delta - step * grad.sign(), bound, s.space)
    expected = _apply(x, delta.detach(), s.space)
    assert torch.equal(pgd([net], x, boxes, s), expected)


def test_ensemble_lowers_every_members_loss():
    x, boxes = fixture_image_with_plate()
    nets = [TinyDetector(0), TinyDetector(1)]
    adv = pgd(nets, x, boxes, settings())
    for net in nets:
        assert topk_loss(raw_of(net, adv)[0, 4], 100) < topk_loss(raw_of(net, x)[0, 4], 100)


def test_ensemble_normalizes_each_member():
    # A member whose loss is 1000 times larger must not dominate: each loss is divided by its
    # value on the clean image.
    x, boxes = fixture_image_with_plate()
    plain = pgd([TinyDetector(0), TinyDetector(1)], x, boxes, settings())
    scaled = pgd([TinyDetector(0), Scaled(TinyDetector(1), 1000.0)], x, boxes, settings())
    assert torch.equal(plain, scaled)


def test_momentum_changes_the_path_and_keeps_the_budget():
    x, boxes = fixture_image_with_plate()
    net = TinyDetector()
    plain = pgd([net], x, boxes, settings(momentum=0.0))
    mom = pgd([net], x, boxes, settings(momentum=1.0))
    assert not torch.equal(plain, mom)
    assert topk_loss(raw_of(net, mom)[0, 4], 100) < topk_loss(raw_of(net, x)[0, 4], 100)
    outside = 1 - plate_mask(*x.shape[-2:], boxes, 0.10)
    assert ((mom - x).abs() * outside).max() == 0
    assert (mom - x).abs().max() <= 8 / 255 + 1e-6


def test_diverse_keeps_the_shape():
    x = torch.rand(1, 3, 448, 640)
    g = gen()
    for _ in range(20):
        assert diverse(x, 1.0, 0.9, g).shape == x.shape
    assert torch.equal(diverse(x, 0.0, 0.9, g), x)  # p = 0: never transformed


def test_diversity_is_reproducible_and_stays_on_the_stride():
    x, boxes = fixture_image_with_plate()
    s = settings(diversity_p=0.5)
    net = TinyDetector()
    a = pgd([net], x, boxes, s, generator=gen(7))
    b = pgd([net], x, boxes, s, generator=gen(7))
    c = pgd([net], x, boxes, s, generator=gen(8))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert all(h % 32 == 0 and w % 32 == 0 for *_, h, w in net.seen_shapes)


def test_diversity_needs_a_generator():
    x, boxes = fixture_image_with_plate()
    with pytest.raises(ValueError, match="generator"):
        pgd([TinyDetector()], x, boxes, settings(diversity_p=0.5))


def test_diversity_with_hinge_is_refused():
    # DI moves the plates inside the input, so the hinge loss's plate candidates would be wrong.
    x, boxes = fixture_image_with_plate()
    with pytest.raises(ValueError, match="hinge"):
        pgd([TinyDetector()], x, boxes, settings(loss="hinge", diversity_p=0.5), generator=gen())


def test_pgd_needs_at_least_one_surrogate():
    x, boxes = fixture_image_with_plate()
    with pytest.raises(ValueError, match="surrogate"):
        pgd([], x, boxes, settings())


# --- random control -------------------------------------------------------------------------


def test_random_noise_uses_the_full_budget_on_plates_only():
    x = torch.full((1, 3, 64, 64), 0.5)  # mid gray: nothing hits the color-range edge
    boxes = np.array([[10.0, 10.0, 40.0, 30.0]])
    adv = random_sign_noise(x, boxes, settings(), torch.Generator().manual_seed(0))
    inside = plate_mask(64, 64, boxes, 0.10).bool().expand_as(x)
    change = (adv - x).abs() * 255
    assert torch.allclose(change[inside], torch.tensor(8.0), atol=1e-4)
    assert change[~inside].max() == 0
    assert ((adv - x)[inside] > 0).float().mean().item() == pytest.approx(0.5, abs=0.05)


def test_random_noise_is_reproducible_and_respects_ycbcr_budget():
    x, boxes = fixture_white_plate()
    s = settings(space="ycbcr", method="random")
    a = random_sign_noise(x, boxes, s, torch.Generator().manual_seed(3))
    b = random_sign_noise(x, boxes, s, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
    budget = audit(to_uint8(x), to_uint8(a))
    assert budget["Y"] <= 2.0 and budget["Cb"] <= 8.0 and budget["Cr"] <= 8.0


# --- audit ----------------------------------------------------------------------------------


def test_audit_reports_changes_in_both_color_spaces():
    clean = np.full((4, 4, 3), 100, np.uint8)
    adv = clean.copy()
    adv[0, 0] = [110, 100, 100]  # R up 10: Y up 2.99, Cb down 1.69, Cr up 5
    a = audit(clean, adv)
    assert a["R"] == 10 and a["G"] == 0 and a["B"] == 0
    assert a["Y"] == pytest.approx(2.99) and a["Cb"] == pytest.approx(1.68736)
    assert a["Cr"] == pytest.approx(5.0)


def test_audit_rejects_mismatched_images():
    with pytest.raises(ValueError):
        audit(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))


def test_to_uint8_rounds_to_the_nearest_level():
    x = torch.tensor([0.0, 0.4 / 255, 0.6 / 255, 1.0, 1.2]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    u8 = to_uint8(x)
    assert u8.shape == (1, 5, 3) and u8.dtype == np.uint8
    assert u8[0, :, 0].tolist() == [0, 0, 1, 255, 255]


# SHA-256 of pgd's output on CPU, recorded from milestone 2a's code before the per-family views
# (milestone 2b) were added. YOLOv8 attacks must not change, or every cached attack would be a
# different attack under the same fingerprint. Tied to this environment's torch build.
DETECT_PATH_HASHES = {
    "ycbcr": "2550ef0ccee42749a37ac82a88c48f2a05fa0ca3669ee4e8041561ee47297498",
    "ensemble_di_mi": "9d5749e1eb4d169843878d05cf28651806ff5c001c671150ce1e856577eeae9a",
    "hinge_rgb": "cf7ae0ce934eb4f7742260163eb996f47d38dadea4befb4f816c1491cd997edb",
}


def test_pgd_detect_path_is_unchanged():
    import hashlib

    x, boxes = fixture_image_with_plate()
    cases = {
        "ycbcr": pgd([TinyDetector()], x, boxes, settings(space="ycbcr", steps=10)),
        "ensemble_di_mi": pgd(
            [TinyDetector(0), TinyDetector(1)],
            x,
            boxes,
            settings(space="ycbcr", steps=10, momentum=1.0, diversity_p=0.5),
            generator=gen(3),
        ),
        "hinge_rgb": pgd([TinyDetector()], x, boxes, settings(space="rgb", steps=10, loss="hinge")),
    }
    got = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in cases.items()}
    assert got == DETECT_PATH_HASHES
