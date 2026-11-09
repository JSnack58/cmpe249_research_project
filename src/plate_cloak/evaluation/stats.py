"""Bootstrap confidence intervals over evaluation images.

A resample is a weight vector: weights[i] = how many times image i was drawn. The caller's
`statistic` turns weights into one number (for example recall, or the paired difference
"RT-DETR's recall drop minus YOLOv10's"). Because every model inside one statistic sees the same
weights, comparisons between models are paired automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    point: float  # the statistic on the full sample (every weight 1)
    low: float  # lower end of the percentile interval
    high: float  # upper end

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def bootstrap_ci(
    n_items: int,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> CI:
    """Percentile bootstrap interval for statistic(weights).

    Each resample draws n_items indices with replacement and passes their counts (an integer
    array of length n_items that sums to n_items) to `statistic`. Resamples where the statistic
    is nan (for example, no plates drawn) are skipped; if every one is nan, low and high are nan.
    """
    # Regular test set.
    full_sample_draw = np.ones(n_items, dtype=int)
    # Random Number Generator to make draws.
    rng = np.random.default_rng(seed)
    baseline = float(statistic(full_sample_draw))
    resample_values = []
    for _ in range(n_resamples):
        # Draw which indices (images) get picked.
        draw = rng.integers(0, n_items, size=n_items)
        # Count how many times each index (image) was picked.
        weights = np.bincount(draw, minlength=n_items)
        # Score this resample; skip it if the statistic is undefined (e.g. no plates drawn).
        value = statistic(weights)
        if np.isnan(value):
            continue
        resample_values.append(value)
    # Every resample was nan: there is no interval to report.
    if not resample_values:
        return CI(baseline, np.nan, np.nan)
    # The interval is the middle `ci` share of the resample values.
    low_percentile = float(np.quantile(resample_values, (1 - ci) / 2))
    high_percentile = float(np.quantile(resample_values, 1 - (1 - ci) / 2))
    return CI(baseline, low_percentile, high_percentile)
