"""Average precision and recall from per-image matches (Task 8's ImageMatch).

AP is single-class, COCO style: pool every image's detections, sort by score, and average the
interpolated precision at 101 recall levels (0, 0.01, ..., 1). It is checked against
pycocotools in tests/test_detection.py. One difference by design: COCO keeps only
each image's top 100 detections; we keep all of them (plate images rarely have more than a few).

`weights[i]` counts how many times image i is in the sample. A bootstrap resample is just a
weight vector (image i drawn twice -> weight 2), so it can be scored without matching again.
For many resamples of the same matches, build `Pooled(matches)` once and call its methods.
"""

from __future__ import annotations

import numpy as np

from plate_cloak.evaluation.matching import ImageMatch

RECALL_LEVELS = np.linspace(0, 1, 101)


class Pooled:
    """Every image's matches pooled and ranked once, so each resample only reweights.

    A resample changes how often each image counts, never the order of the detections, so the
    sort is done here once and each `ap(weights)` call is a weighted cumulative sum. That equals
    duplicating image i weights[i] times, except that detections with exactly equal scores in
    the same image may be visited in a different order (ties never happen with real scores).
    """

    def __init__(self, matches: list[ImageMatch]):
        self.n_images = len(matches)
        self.n_gt = np.array([m.n_gt for m in matches], dtype=np.int64)
        self.n_found = np.array([int(m.gt_found.sum()) for m in matches], dtype=np.int64)
        scores = np.concatenate([m.scores for m in matches] + [np.zeros(0)])
        is_tp = np.concatenate([m.is_tp for m in matches] + [np.zeros(0, bool)])
        image = np.repeat(np.arange(len(matches)), [len(m.scores) for m in matches])
        order = np.argsort(-scores, kind="mergesort")  # stable, as pycocotools does
        self.is_tp, self.image = is_tp[order].astype(bool), image[order]

    def ap(self, weights: np.ndarray | None = None) -> float:
        """COCO 101-point interpolated AP; nan if the (weighted) images contain no plates."""
        w = _weights(self.n_images, weights)
        n_gt = int(w @ self.n_gt)
        if n_gt == 0:
            return float("nan")
        det_w = w[self.image]
        kept = det_w > 0  # an image drawn 0 times contributes no detections
        det_w, is_tp = det_w[kept], self.is_tp[kept]
        if det_w.size == 0:
            return 0.0
        tp = np.cumsum(det_w * is_tp)
        fp = np.cumsum(det_w * ~is_tp)
        rec = tp / n_gt
        prec = tp / (tp + fp)
        # Interpolate: precision at recall r is the best precision at any recall >= r.
        envelope = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, RECALL_LEVELS, side="left")
        at_levels = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
        return float(at_levels.mean())

    def recall(self, weights: np.ndarray | None = None) -> float:
        """Share of plates found at the confidence threshold; nan if there are no plates."""
        w = _weights(self.n_images, weights)
        n_gt = int(w @ self.n_gt)
        return float("nan") if n_gt == 0 else int(w @ self.n_found) / n_gt


def average_precision(matches: list[ImageMatch], weights: np.ndarray | None = None) -> float:
    """COCO 101-point interpolated AP; nan if the (weighted) images contain no plates."""
    return Pooled(matches).ap(weights)


def recall(matches: list[ImageMatch], weights: np.ndarray | None = None) -> float:
    """Share of plates found at the confidence threshold; nan if there are no plates."""
    return Pooled(matches).recall(weights)


def _weights(n_images: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(n_images, dtype=np.int64)
    w = np.asarray(weights)
    if w.shape != (n_images,):
        raise ValueError(f"weights has shape {w.shape}, expected ({n_images},)")
    if not np.issubdtype(w.dtype, np.integer) or (w < 0).any():
        raise ValueError("weights must be non-negative integers (how often each image is drawn)")
    return w
