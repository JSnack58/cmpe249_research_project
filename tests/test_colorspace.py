import torch

from plate_cloak.attack.colorspace import fit_to_gamut, rgb_to_ycbcr, ycbcr_to_rgb

BUDGET = torch.tensor([2.0, 8.0, 8.0]).view(1, 3, 1, 1) / 255  # Y, Cb, Cr


def apply(x, t, d):
    return ycbcr_to_rgb(rgb_to_ycbcr(x) + t * d)


def saturated_fixture(seed=0):
    """Random pixels plus pure white and pure black ones: the cases that broke Spike 01."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(2, 3, 16, 16, generator=g)
    x[:, :, :4] = 1.0  # white rows, like a plate background
    x[:, :, 4:8] = 0.0  # black rows, like plate characters
    return x


def random_noise(x, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(x.shape, generator=g) * 2 - 1) * BUDGET


# --- conversions (already implemented) -------------------------------------------------------


def test_round_trip():
    x = torch.rand(2, 3, 16, 16)
    assert (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max() < 1e-5


def test_white_and_gray_have_no_color():
    gray = torch.full((1, 3, 1, 1), 0.3)
    ycc = rgb_to_ycbcr(gray)
    assert torch.allclose(ycc[0, :, 0, 0], torch.tensor([0.3, 0.5, 0.5]), atol=1e-6)


# --- fit_to_gamut (Task 5, owner) -------------------------------------------------------------


def test_fit_keeps_result_in_gamut():
    x = saturated_fixture()
    d = random_noise(x)
    out = apply(x, fit_to_gamut(x, d), d)
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6


def test_fit_shape_and_range():
    x = saturated_fixture()
    t = fit_to_gamut(x, random_noise(x))
    assert t.shape == (2, 1, 16, 16)
    assert (t >= 0).all() and (t <= 1).all()


def test_fit_never_grows_noise():
    x = saturated_fixture()
    d = random_noise(x)
    scaled = fit_to_gamut(x, d) * d
    assert (scaled.abs() <= d.abs() + 1e-9).all()  # so the Y/Cb/Cr budget still holds


def test_fit_is_identity_when_already_in_gamut():
    x = torch.full((1, 3, 4, 4), 0.5)
    d = torch.full_like(x, 1 / 255)
    assert torch.equal(fit_to_gamut(x, d), torch.ones(1, 1, 4, 4))


def test_fit_only_shrinks_pixels_that_need_it():
    x = torch.full((1, 3, 1, 2), 0.5)
    x[..., 1] = 1.0  # second pixel is pure white
    d = torch.zeros_like(x)
    d[:, 2] = 8 / 255  # push Cr up: makes red brighter, impossible on white
    t = fit_to_gamut(x, d)
    assert t[0, 0, 0, 0] == 1.0  # mid-gray pixel has room
    assert t[0, 0, 0, 1] == 0.0  # white pixel cannot get redder at all


def test_fit_is_tight_not_overly_cautious():
    # A pixel that can take exactly half the noise should get t = 0.5, not something smaller.
    x = torch.full((1, 3, 1, 1), 0.5)
    d = torch.zeros_like(x)
    d[:, 0] = 1.0  # Y up by 1.0 moves R, G, B all up by 1.0; only 0.5 of room exists
    assert torch.allclose(fit_to_gamut(x, d), torch.full((1, 1, 1, 1), 0.5), atol=1e-6)


def test_zero_noise_component_does_not_produce_nan():
    x = torch.ones(1, 3, 2, 2)
    d = torch.zeros_like(x)
    d[:, 1] = 4 / 255
    assert torch.isfinite(fit_to_gamut(x, d)).all()


def test_all_zero_noise_gives_t_one():
    x = saturated_fixture()
    assert torch.equal(fit_to_gamut(x, torch.zeros_like(x)), torch.ones(2, 1, 16, 16))
