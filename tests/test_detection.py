import math

import numpy as np
import pytest

from plate_cloak.evaluation.detection import average_precision, recall
from plate_cloak.evaluation.matching import ImageMatch, match_image


def random_scenes(seed: int, n: int):
    """Random images as (pred_boxes, pred_scores, gt_boxes, gt_ignore) tuples.

    Plates sit on a grid so they never overlap each other. Each plate gets 0-2 detections that
    are small shifts of it (IoU from about 0.4 to 1, so some hits and some near misses), plus
    random false alarms that stay clear of ignored plates. Detections on ignored plates are
    shifted only slightly, so plain IoU and COCO's crowd overlap (intersection over detection
    area) agree on them and the comparison with pycocotools is exact.
    """
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(n):
        n_gt = int(rng.integers(0, 4))
        cells = rng.choice(16, size=n_gt, replace=False)
        gt = np.array(
            [
                [
                    c % 4 * 150 + 10,
                    c // 4 * 150 + 10,
                    c % 4 * 150 + 10 + rng.uniform(40, 120),
                    c // 4 * 150 + 10 + rng.uniform(15, 40),
                ]
                for c in cells
            ]
        ).reshape(-1, 4)
        ignore = rng.random(n_gt) < 0.25
        preds = []
        for g, ig in zip(gt, ignore, strict=True):
            w, h = g[2] - g[0], g[3] - g[1]
            shift = 0.05 if ig else 0.35
            for _ in range(int(rng.integers(0, 3))):
                d = rng.uniform(-shift, shift, 2) * [w, h]
                preds.append(g + [d[0], d[1], d[0], d[1]])
        for _ in range(int(rng.integers(0, 3))):
            x, y = rng.uniform(0, 560), rng.uniform(0, 560)
            box = np.array([x, y, x + rng.uniform(20, 80), y + rng.uniform(10, 30)])
            ig_boxes = gt[ignore]
            clear = all(
                box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3]
                for b in ig_boxes
            )
            if clear:
                preds.append(box)
        pred = np.array(preds).reshape(-1, 4)
        scores = rng.uniform(0.01, 1.0, len(pred))
        scenes.append((pred, scores, gt, ignore))
    return scenes


def pycocotools_ap50(scenes) -> float:
    """AP at IoU 0.5 from the official COCO evaluator, ignored plates passed as crowd regions.

    We compare with pycocotools directly: torchmetrics 1.9 wraps it but its compute() gave a
    different value (0.1866 vs 0.1896 on seed 0) even though pycocotools run on torchmetrics' own
    exported files agrees with us exactly.
    """
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    for i, (pred, scores, gt, ignore) in enumerate(scenes):
        images.append({"id": i})
        for box, crowd in zip(gt, ignore, strict=True):
            w, h = box[2] - box[0], box[3] - box[1]
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": 1,
                    "iscrowd": int(crowd),
                    "bbox": [box[0], box[1], w, h],
                    "area": w * h,
                }
            )
        for box, score in zip(pred, scores, strict=True):
            dets.append(
                {
                    "image_id": i,
                    "category_id": 1,
                    "score": float(score),
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1}]}
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def hand_match(scores, is_tp, n_gt, found=None):
    found = np.zeros(n_gt, bool) if found is None else np.array(found)
    return ImageMatch(np.array(scores, float), np.array(is_tp, bool), n_gt, found)


# --- validated against pycocotools ------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ap_matches_pycocotools_on_random_scenes(seed):
    scenes = random_scenes(seed=seed, n=50)
    assert any(s[0].size for s in scenes)  # the scenes do contain detections
    ours = average_precision([match_image(*s) for s in scenes])
    assert ours == pytest.approx(pycocotools_ap50(scenes), abs=1e-9)


# --- by hand ----------------------------------------------------------------------------------


def test_ap_by_hand():
    # Ranked: TP, FP, TP, out of 2 plates. Precision 1, 0.5, 0.67 at recall 0.5, 0.5, 1.
    # Interpolated: 1.0 for recall levels 0..0.5 (51 levels), 0.67 for 0.51..1 (50 levels).
    m = hand_match([0.9, 0.8, 0.7], [True, False, True], 2)
    assert average_precision([m]) == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_perfect_detector_scores_one():
    assert average_precision([hand_match([0.9, 0.8], [True, True], 2)]) == pytest.approx(1.0)


def test_ap_pools_images_before_ranking():
    # Image 1's false alarm (0.95) outranks image 2's hit (0.6): pooling must see that.
    a = hand_match([0.95], [False], 0)
    b = hand_match([0.6], [True], 1)
    assert average_precision([a, b]) == pytest.approx(0.5)


def test_weights_equal_duplicating_images():
    ms = [match_image(*s) for s in random_scenes(seed=1, n=10)]
    w = np.array([2, 0, 1, 1, 3, 0, 1, 1, 1, 0])
    dup = [m for m, k in zip(ms, w, strict=True) for _ in range(k)]
    assert average_precision(ms, w) == pytest.approx(average_precision(dup))
    assert recall(ms, w) == pytest.approx(recall(dup), nan_ok=True)


def test_recall_counts_found_plates():
    ms = [hand_match([0.9], [True], 2, [True, False]), hand_match([], [], 1, [False])]
    assert recall(ms) == pytest.approx(1 / 3)
    assert recall(ms, np.array([3, 0])) == pytest.approx(0.5)


def test_no_gt_gives_nan_not_error():
    empty = ImageMatch(np.zeros(0), np.zeros(0, bool), 0, np.zeros(0, bool))
    assert math.isnan(average_precision([empty]))
    assert math.isnan(recall([empty]))
    assert math.isnan(average_precision([]))
    only_false_alarms = hand_match([0.9], [False], 0)
    assert math.isnan(average_precision([only_false_alarms]))


def test_resample_without_plates_gives_nan():
    ms = [hand_match([0.9], [True], 1, [True]), hand_match([0.4], [False], 0)]
    assert math.isnan(average_precision(ms, np.array([0, 2])))
    assert math.isnan(recall(ms, np.array([0, 2])))


def test_model_with_no_detections_scores_zero():
    m = match_image(np.zeros((0, 4)), np.zeros(0), np.array([[0, 0, 10, 10]]), np.array([False]))
    assert average_precision([m]) == 0.0 and recall([m]) == 0.0


def test_bad_weights_are_rejected():
    ms = [hand_match([0.9], [True], 1, [True])]
    for bad in [np.array([1, 1]), np.array([-1]), np.array([0.5])]:
        with pytest.raises(ValueError):
            average_precision(ms, bad)
