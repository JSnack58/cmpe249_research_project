import numpy as np
import pytest
import torch
from detectors import TinyDetector

from plate_cloak.attack.losses import anchors_on_plates, hinge_loss, topk_loss
from plate_cloak.attack.pgd import (
    AttackSettings,
    LossSettings,
    audit,
    pgd,
    plate_mask,
    predict_padding,
    random_sign_noise,
    to_uint8,
)
from plate_cloak.config import BASE_CONFIG, load_config


def settings(space="rgb", eps=None, steps=10, loss="topk", method="pgd"):
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
    )


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


def test_settings_rgb_uses_one_budget_for_all_channels():
    cfg = load_config(BASE_CONFIG, overrides=["attack.space=rgb"])
    assert torch.allclose(AttackSettings.from_cfg(cfg.attack).eps * 255, torch.full((3,), 8.0))


def test_sobel_mask_is_not_silently_ignored():
    cfg = load_config(BASE_CONFIG, overrides=["attack.mask=sobel"])
    with pytest.raises(NotImplementedError):
        AttackSettings.from_cfg(cfg.attack)


# --- losses ---------------------------------------------------------------------------------


def test_topk_loss_sums_the_k_highest_scores():
    raw = torch.zeros(1, 5, 6)
    raw[0, 4] = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.0])
    assert torch.isclose(topk_loss(raw, 2), torch.tensor(1.7))
    assert torch.isclose(topk_loss(raw, 100), torch.tensor(2.3))  # k larger than A is fine


def test_hinge_loss_counts_only_plate_candidates_above_the_floor():
    raw = torch.zeros(1, 5, 4)
    raw[0, 4] = torch.tensor([0.5, 0.05, 0.9, 0.3])
    on_plate = torch.tensor([True, True, False, True])
    # 0.5 -> 0.4, 0.05 is below the floor, 0.9 is off the plate, 0.3 -> 0.2
    assert torch.isclose(hinge_loss(raw, 0.1, on_plate), torch.tensor(0.6))
    assert torch.isclose(hinge_loss(raw, 0.1, None), torch.tensor(1.4))


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
    adv = pgd(net, x, boxes, settings(space=space))
    assert topk_loss(raw_of(net, adv), 100) < topk_loss(raw_of(net, x), 100)
    outside = 1 - plate_mask(*x.shape[-2:], boxes, 0.10)
    assert ((adv - x).abs() * outside).max() == 0
    assert adv.min() >= 0 and adv.max() <= 1


def test_pgd_with_hinge_loss_lowers_it():
    x, boxes = fixture_image_with_plate()
    net = TinyDetector()
    adv = pgd(net, x, boxes, settings(loss="hinge"))

    def hinge(img):
        left, _, top, _ = predict_padding(*img.shape[-2:])
        raw = raw_of(net, img)
        return hinge_loss(raw, 0.1, anchors_on_plates(raw, boxes + [left, top, left, top]))

    assert hinge(adv) < hinge(x)


def test_ycbcr_budget_holds_after_rounding_on_white_plate():
    x, boxes = fixture_white_plate()
    adv = pgd(TinyDetector(), x, boxes, settings(space="ycbcr", eps=(2, 8, 8), steps=40))
    a = audit(to_uint8(x), to_uint8(adv))
    assert a["Y"] <= 2.0 and a["Cb"] <= 8.0 and a["Cr"] <= 8.0
    assert a["Cb"] > 4.0 or a["Cr"] > 4.0  # the attack did use the chroma budget


def test_rgb_budget_holds_after_rounding():
    x, boxes = fixture_white_plate()
    a = audit(to_uint8(x), to_uint8(pgd(TinyDetector(), x, boxes, settings(space="rgb"))))
    assert max(a["R"], a["G"], a["B"]) <= 8.0


def test_non_multiple_of_32_input_round_trips_shape():
    x = torch.rand(1, 3, 427, 640)
    net = TinyDetector()
    adv = pgd(net, x, np.array([[100.0, 200.0, 300.0, 260.0]]), settings(steps=2))
    assert adv.shape == x.shape
    assert set(net.seen_shapes) == {(1, 3, 448, 640)}


def test_pgd_is_deterministic_and_works_under_no_grad():
    x, boxes = fixture_image_with_plate()
    first = pgd(TinyDetector(), x, boxes, settings(space="ycbcr"))
    with torch.no_grad():
        second = pgd(TinyDetector(), x, boxes, settings(space="ycbcr"))
    assert torch.equal(first, second)


def test_pgd_refuses_a_model_in_training_mode():
    x, boxes = fixture_image_with_plate()
    with pytest.raises(ValueError, match="eval mode"):
        pgd(TinyDetector().train(), x, boxes, settings())


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
