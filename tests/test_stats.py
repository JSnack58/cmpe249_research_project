import math

import numpy as np
import pytest

from plate_cloak.evaluation.stats import CI, bootstrap_ci


def test_ci_of_mean_covers_truth():
    x = np.random.default_rng(0).normal(5.0, 1.0, 400)
    r = bootstrap_ci(len(x), lambda w: np.average(x, weights=w))
    assert r.low < 5.0 < r.high and r.high - r.low < 0.3
    assert r.point == pytest.approx(x.mean())


def test_deterministic_for_seed():
    def s(w):
        return float(w[:10].sum())

    assert bootstrap_ci(50, s, seed=7) == bootstrap_ci(50, s, seed=7)
    assert bootstrap_ci(50, s, seed=7) != bootstrap_ci(50, s, seed=8)


def test_statistic_gets_integer_counts_of_the_right_shape():
    seen = []

    def s(w):
        seen.append(w)
        return 0.0

    bootstrap_ci(20, s, n_resamples=30)
    point, resamples = seen[0], seen[1:]
    assert np.array_equal(point, np.ones(20))
    assert len(resamples) == 30
    for w in resamples:
        assert w.shape == (20,) and np.issubdtype(w.dtype, np.integer)
        assert w.sum() == 20 and (w >= 0).all()


def test_paired_difference_of_identical_models_is_exactly_zero():
    x = np.random.default_rng(1).random(100)
    r = bootstrap_ci(100, lambda w: np.average(x, weights=w) - np.average(x, weights=w))
    assert (r.point, r.low, r.high) == (0.0, 0.0, 0.0)


def test_interval_is_the_percentiles_of_the_resamples():
    # With the statistic "weight of item 0", resample values are binomial(10, 0.1) counts.
    vals = []

    def s(w):
        vals.append(float(w[0]))
        return vals[-1]

    r = bootstrap_ci(10, s, n_resamples=2000, ci=0.9)
    resampled = np.array(vals[1:])
    assert (r.low, r.high) == (np.quantile(resampled, 0.05), np.quantile(resampled, 0.95))


def test_narrower_level_gives_narrower_interval():
    x = np.random.default_rng(2).random(200)

    def mean(w):
        return np.average(x, weights=w)

    wide, narrow = bootstrap_ci(200, mean, ci=0.95), bootstrap_ci(200, mean, ci=0.5)
    assert wide.low < narrow.low < narrow.high < wide.high


def test_nan_resamples_skipped():
    def s(w):
        return float("nan") if w[0] == 0 else 1.0

    r = bootstrap_ci(5, s)
    assert (r.point, r.low, r.high) == (1.0, 1.0, 1.0)


def test_all_nan_resamples_give_nan_bounds():
    r = bootstrap_ci(3, lambda w: float("nan"))
    assert math.isnan(r.point) and math.isnan(r.low) and math.isnan(r.high)


def test_as_dict():
    assert CI(0.5, 0.4, 0.6).as_dict() == {"point": 0.5, "low": 0.4, "high": 0.6}
