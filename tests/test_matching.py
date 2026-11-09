import numpy as np

from plate_cloak.evaluation.matching import ImageMatch, iou, match_image


def boxes(*rows):
    return np.array(rows, dtype=float).reshape(-1, 4)


NOT_IGNORED = np.array([False])

# --- iou --------------------------------------------------------------------------------------


def test_iou_known_values():
    got = iou(boxes([0, 0, 10, 10]), boxes([5, 0, 15, 10], [20, 20, 30, 30]))
    np.testing.assert_allclose(got, [[1 / 3, 0]])


def test_iou_shape_is_n_by_m():
    a = boxes([0, 0, 10, 10], [0, 0, 5, 5], [2, 2, 4, 4])
    b = boxes([0, 0, 10, 10], [100, 100, 110, 120])
    got = iou(a, b)
    assert got.shape == (3, 2)
    np.testing.assert_allclose(got[:, 0], [1.0, 0.25, 0.04])


def test_iou_touching_edges_is_zero_not_negative():
    np.testing.assert_allclose(iou(boxes([0, 0, 10, 10]), boxes([10, 0, 20, 10])), [[0.0]])
    np.testing.assert_allclose(iou(boxes([0, 0, 10, 10]), boxes([30, 30, 40, 40])), [[0.0]])


def test_iou_with_no_boxes_keeps_the_shape():
    assert iou(np.zeros((0, 4)), boxes([0, 0, 1, 1])).shape == (0, 1)
    assert iou(boxes([0, 0, 1, 1]), np.zeros((0, 4))).shape == (1, 0)


# --- match_image ------------------------------------------------------------------------------


def test_each_gt_matched_once_highest_score_first():
    m = match_image(
        boxes([0, 0, 10, 10], [1, 0, 11, 10]),
        np.array([0.6, 0.9]),
        boxes([0, 0, 10, 10]),
        NOT_IGNORED,
    )
    assert m.scores.tolist() == [0.9, 0.6]  # sorted by score, highest first
    assert m.is_tp.tolist() == [True, False]  # the plate is taken by the 0.9 detection
    assert m.n_gt == 1 and m.gt_found.tolist() == [True]


def test_detection_takes_the_free_plate_when_its_best_is_taken():
    # Two overlapping plates. The second detection overlaps plate 0 most (IoU 0.82), but plate 0
    # is already taken, and it overlaps plate 1 enough (IoU 0.54), so it takes plate 1 instead
    # of becoming a false positive.
    gt = boxes([0, 0, 10, 10], [4, 0, 14, 10])
    pred = boxes([0, 0, 10, 10], [1, 0, 11, 10])
    m = match_image(pred, np.array([0.9, 0.8]), gt, np.array([False, False]))
    assert m.is_tp.tolist() == [True, True]
    assert m.gt_found.tolist() == [True, True]


def test_detection_on_ignored_plate_is_dropped():
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.8]), boxes([0, 0, 10, 10]), np.array([True]))
    assert m.scores.size == 0 and m.is_tp.size == 0
    assert m.n_gt == 0 and m.gt_found.size == 0


def test_real_plate_wins_over_ignored_plate():
    # A detection overlapping both a counted plate and an ignored one is a hit on the counted one.
    gt = boxes([0, 0, 10, 10], [0, 0, 10, 10])
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.8]), gt, np.array([True, False]))
    assert m.is_tp.tolist() == [True] and m.n_gt == 1 and m.gt_found.tolist() == [True]


def test_duplicate_detection_on_a_taken_plate_is_a_false_positive():
    m = match_image(
        boxes([0, 0, 10, 10], [0, 0, 10, 10]),
        np.array([0.9, 0.7]),
        boxes([0, 0, 10, 10]),
        NOT_IGNORED,
    )
    assert m.is_tp.tolist() == [True, False]


def test_detection_below_the_iou_threshold_is_a_false_positive():
    # IoU 1/3 < 0.5: the plate is missed and the detection is a false alarm.
    m = match_image(boxes([5, 0, 15, 10]), np.array([0.9]), boxes([0, 0, 10, 10]), NOT_IGNORED)
    assert m.is_tp.tolist() == [False] and m.gt_found.tolist() == [False]


def test_iou_threshold_is_inclusive():
    # IoU exactly 0.5 counts as a match.
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.9]), boxes([0, 0, 10, 20]), NOT_IGNORED)
    assert m.is_tp.tolist() == [True]


def test_low_score_match_counts_for_ap_not_recall():
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.1]), boxes([0, 0, 10, 10]), NOT_IGNORED)
    assert m.is_tp.tolist() == [True] and m.gt_found.tolist() == [False]


def test_score_exactly_at_conf_counts_as_found():
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.25]), boxes([0, 0, 10, 10]), NOT_IGNORED)
    assert m.gt_found.tolist() == [True]


def test_equal_scores_keep_input_order():
    # Both detections hit the plate; with equal scores the first one listed takes it.
    pred = boxes([2, 0, 12, 10], [0, 0, 10, 10])
    m = match_image(pred, np.array([0.5, 0.5]), boxes([0, 0, 10, 10]), NOT_IGNORED)
    assert m.is_tp.tolist() == [True, False]


def test_gt_found_follows_the_order_of_counted_plates():
    gt = boxes([0, 0, 10, 10], [50, 50, 52, 52], [20, 0, 30, 10])  # middle plate is ignored
    m = match_image(boxes([20, 0, 30, 10]), np.array([0.9]), gt, np.array([False, True, False]))
    assert m.n_gt == 2 and m.gt_found.tolist() == [False, True]


def test_no_predictions_and_no_gts():
    m = match_image(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), np.zeros(0, bool))
    assert isinstance(m, ImageMatch)
    assert m.n_gt == 0 and m.scores.size == 0 and m.is_tp.size == 0 and m.gt_found.size == 0


def test_no_predictions_means_every_plate_missed():
    m = match_image(np.zeros((0, 4)), np.zeros(0), boxes([0, 0, 10, 10]), NOT_IGNORED)
    assert m.n_gt == 1 and m.gt_found.tolist() == [False] and m.scores.size == 0


def test_predictions_but_no_plates_are_all_false_positives():
    m = match_image(
        boxes([0, 0, 10, 10], [5, 5, 9, 9]),
        np.array([0.3, 0.8]),
        np.zeros((0, 4)),
        np.zeros(0, bool),
    )
    assert m.scores.tolist() == [0.8, 0.3] and m.is_tp.tolist() == [False, False]


def test_output_types():
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.9]), boxes([0, 0, 10, 10]), NOT_IGNORED)
    assert m.is_tp.dtype == bool and m.gt_found.dtype == bool and isinstance(m.n_gt, int)


def test_output_types_when_every_detection_is_dropped():
    # np.array([]) defaults to float64; AP's `~is_tp` fails on floats once images are pooled.
    m = match_image(boxes([0, 0, 10, 10]), np.array([0.8]), boxes([0, 0, 10, 10]), np.array([True]))
    assert m.is_tp.dtype == bool and m.scores.dtype == float
