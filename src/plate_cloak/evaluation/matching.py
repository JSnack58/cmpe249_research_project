"""IoU and per-image matching: which detections found which plates.

Boxes are (N, 4) float arrays of x1, y1, x2, y2 in pixels. Matching follows the COCO rules for
one class, with ignored plates (too small to read) treated like COCO's "crowd" regions:
a detection on one is neither a hit nor a false alarm, it simply doesn't count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageMatch:
    """One image's detections after matching, everything AP and recall need.

    scores:   (K,) scores of the counted detections, highest first
    is_tp:    (K,) True where that detection matched a plate (true positive)
    n_gt:     number of plates that count (not ignored)
    gt_found: (n_gt,) True where that plate was matched by a detection scoring >= conf,
              in the same order as the non-ignored plates in gt_boxes
    """

    scores: np.ndarray
    is_tp: np.ndarray
    n_gt: int
    gt_found: np.ndarray


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) intersection over union between every box in a (N, 4) and every box in b (M, 4)."""
    # The overlap of two boxes is itself a box. Indexing with None turns a's edges into a column
    # (N, 1) and b's into a row (1, M), so broadcasting compares every pair: each result is (N, M).
    overlap_left = np.maximum(a[:, None, 0], b[None, :, 0])
    overlap_top = np.maximum(a[:, None, 1], b[None, :, 1])
    overlap_right = np.minimum(a[:, None, 2], b[None, :, 2])
    overlap_bottom = np.minimum(a[:, None, 3], b[None, :, 3])

    # Boxes that don't overlap give a negative width or height; clip it to 0.
    # y grows downward in images, so the height is bottom - top.
    overlap_width = np.clip(overlap_right - overlap_left, 0, None)
    overlap_height = np.clip(overlap_bottom - overlap_top, 0, None)
    overlap_area = overlap_width * overlap_height

    # Union = both areas minus the overlap, which they'd otherwise count twice.
    # area_a is a column (N, 1) and area_b a row (1, M), so the sum is (N, M) like overlap_area.
    area_a = (a[:, None, 2] - a[:, None, 0]) * (a[:, None, 3] - a[:, None, 1])
    area_b = (b[None, :, 2] - b[None, :, 0]) * (b[None, :, 3] - b[None, :, 1])
    union_area = area_a + area_b - overlap_area

    return overlap_area / union_area


def match_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_ignore: np.ndarray,
    iou_thr: float = 0.5,
    conf: float = 0.25,
) -> ImageMatch:
    """Match one image's detections to its plates.

    Go through the detections from highest to lowest score (equal scores keep their input order).
    Each detection:
      1. matches the not-yet-matched, non-ignored plate it overlaps most, if that IoU >= iou_thr
         (a true positive; that plate is now taken);
      2. otherwise, if it overlaps any ignored plate at IoU >= iou_thr, it is dropped entirely;
      3. otherwise it is a false positive.
    Keep every detection whatever its score (AP needs the full ranking); `conf` only decides
    which matches count as "found" for recall.
    """
    # Reorder the detections from highest to lowest score, so row i of every array below is the
    # i-th most confident detection. argsort only returns the order; indexing applies it.
    # "stable" keeps equal scores in their input order.
    order = np.argsort(-pred_scores, kind="stable")
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    # Split the plates: counted ones can be hit or missed, ignored ones only absorb detections.
    counted = gt_boxes[~gt_ignore]
    ignored = gt_boxes[gt_ignore]
    n_gt = len(counted)

    # Overlap of every detection with every plate: (K, n_gt) and (K, n_ignored).
    # These are fixed numbers; only `taken` changes during the loop.
    counted_ious = iou(pred_boxes, counted)
    ignored_ious = iou(pred_boxes, ignored)

    # Per-plate bookkeeping, one entry per counted plate.
    #   taken[j]: some detection already claimed plate j, so no later one can.
    #   found[j]: plate j was claimed by a detection scoring >= conf (returned for recall).
    taken = np.zeros(n_gt, dtype=bool)
    found = np.zeros(n_gt, dtype=bool)

    # Per-detection verdicts for AP. Dropped detections are left out of both lists.
    scores = []
    is_tp = []

    for i in range(len(pred_scores)):
        pred_score = pred_scores[i]

        # Find the best free plate. Taken plates become -1, below any real IoU, so argmax skips
        # them. With no counted plates there's nothing to pick, and the -1 default makes the hit
        # check below fail. best_plate_idx is only set (and only used) when a plate exists.
        masked = np.where(taken, -1, counted_ious[i])
        best_plate_iou = -1
        if masked.size > 0:
            best_plate_idx = masked.argmax()
            best_plate_iou = masked[best_plate_idx]

        if best_plate_iou >= iou_thr:
            # Hit: claim the plate. It only counts as found if the detection clears conf.
            taken[best_plate_idx] = True
            if pred_score >= conf:
                found[best_plate_idx] = True
            scores.append(pred_score)
            is_tp.append(True)
        elif (ignored_ious[i] >= iou_thr).any():
            # On an ignored plate: neither a hit nor a false alarm, so it isn't recorded.
            continue
        else:
            # False alarm: recorded for AP; the plate bookkeeping is untouched.
            scores.append(pred_score)
            is_tp.append(False)

    # Explicit dtypes: an empty list would otherwise become a float64 array.
    return ImageMatch(
        scores=np.array(scores, dtype=float),
        is_tp=np.array(is_tp, dtype=bool),
        n_gt=n_gt,
        gt_found=found,
    )
