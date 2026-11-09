"""Average precision and recall from per-image matches (Task 8's ImageMatch).

AP is single-class, COCO style: pool every image's detections, sort by score, and average the
interpolated precision at 101 recall levels (0, 0.01, ..., 1). It is checked against
pycocotools in tests/test_detection.py. One difference by design: COCO keeps only
each image's top 100 detections; we keep all of them (plate images rarely have more than a few).

`weights[i]` counts how many times image i is in the sample. A bootstrap resample is just a
weight vector (image i drawn twice -> weight 2), so it can be scored without matching again.
"""

from __future__ import annotations

import numpy as np

from plate_cloak.evaluation.matching import ImageMatch

RECALL_LEVELS = np.linspace(0, 1, 101)


def average_precision(matches: list[ImageMatch], weights: np.ndarray | None = None) -> float:
    """COCO 101-point interpolated AP; nan if the (weighted) images contain no plates."""
    w = _weights(matches, weights)
    n_gt = int(sum(k * m.n_gt for m, k in zip(matches, w, strict=True)))
    if n_gt == 0:
        return float("nan")
    scores = np.concatenate([np.tile(m.scores, k) for m, k in zip(matches, w, strict=True)] + [[]])
    is_tp = np.concatenate(
        [np.tile(m.is_tp, k) for m, k in zip(matches, w, strict=True)] + [np.zeros(0, bool)]
    )
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")  # stable, as pycocotools does
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    rec = tp / n_gt
    prec = tp / (tp + fp)
    # Interpolate: precision at recall r is the best precision at any recall >= r.
    envelope = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, RECALL_LEVELS, side="left")
    at_levels = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(at_levels.mean())


def recall(matches: list[ImageMatch], weights: np.ndarray | None = None) -> float:
    """Share of plates found at the confidence threshold; nan if there are no plates."""
    w = _weights(matches, weights)
    n_gt = sum(k * m.n_gt for m, k in zip(matches, w, strict=True))
    if n_gt == 0:
        return float("nan")
    found = sum(k * int(m.gt_found.sum()) for m, k in zip(matches, w, strict=True))
    return found / n_gt


def _weights(matches: list[ImageMatch], weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(len(matches), dtype=int)
    w = np.asarray(weights)
    if w.shape != (len(matches),):
        raise ValueError(f"weights has shape {w.shape}, expected ({len(matches)},)")
    if not np.issubdtype(w.dtype, np.integer) or (w < 0).any():
        raise ValueError("weights must be non-negative integers (how often each image is drawn)")
    return w
