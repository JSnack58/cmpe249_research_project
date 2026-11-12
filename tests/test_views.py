"""Per-family views: how PGD feeds each detector and reads its plate scores (milestone 2b)."""

import numpy as np
import pytest
import torch
from detectors import FakeEndToEnd, FakeRTDETR, TinyDetector
from test_pgd import fixture_image_with_plate, settings

from plate_cloak.attack.losses import topk_loss
from plate_cloak.attack.pgd import pgd, predict_padding
from plate_cloak.attack.views import DetectView, EndToEndView, RTDETRView, view_for


def test_end_to_end_view_reads_one2one_scores():
    net, x = FakeEndToEnd(), torch.rand(1, 3, 64, 96)
    out = net(x)
    scores = EndToEndView().scores(out)
    assert torch.allclose(scores, out[1]["one2one"]["scores"][0, 0].sigmoid())
    assert not torch.allclose(scores[:6], out[0][0, 4])  # the YOLOv8 reading of the table


def test_rtdetr_view_reads_last_decoder_layer():
    net, x = FakeRTDETR(), torch.rand(1, 3, 64, 64)
    out = net(x)
    scores = RTDETRView(64).scores(out)
    assert torch.allclose(scores, out[1][1][-1, 0, :, 0].sigmoid())
    assert not torch.allclose(scores[:6], out[0][0, 4])


def test_rtdetr_view_stretches_to_a_square():
    view = RTDETRView(640)
    assert view.prepare(torch.rand(1, 3, 480, 640)).shape == (1, 3, 640, 640)
    plates = view.plates_to_input(np.array([[0.0, 0.0, 640.0, 480.0]]), 480, 640)
    assert np.allclose(plates, [[0, 0, 640, 640]])


def test_detect_view_pads_like_predict():
    view, x = DetectView(), torch.rand(1, 3, 480, 630)
    left, right, top, bottom = predict_padding(480, 630)
    seen = view.prepare(x)
    assert seen.shape == (1, 3, 480 + top + bottom, 630 + left + right)
    assert seen.shape[-1] % 32 == 0 and seen.shape[-2] % 32 == 0
    plates = view.plates_to_input(np.array([[10.0, 20.0, 30.0, 40.0]]), 480, 630)
    assert np.allclose(plates, [[10 + left, 20 + top, 30 + left, 40 + top]])


def test_view_for_picks_by_head_not_zoo_family():
    # YOLOv10l is family "yolo" in the zoo, but its head is end-to-end.
    assert isinstance(view_for("yolov10l", FakeEndToEnd(), 640), EndToEndView)
    assert isinstance(view_for("yolov8s", TinyDetector(), 640), DetectView)
    assert isinstance(view_for("rtdetr-l", FakeRTDETR(), 640), RTDETRView)


@pytest.mark.parametrize(
    ("net", "view"),
    [
        (TinyDetector(), DetectView()),
        (FakeEndToEnd(), EndToEndView()),
        (FakeRTDETR(), RTDETRView(256)),
    ],
    ids=["detect", "end2end", "rtdetr"],
)
def test_pgd_lowers_the_score_each_view_reads(net, view):
    x, boxes = fixture_image_with_plate()
    s = settings(space="rgb", eps=(16, 16, 16), steps=10)

    def loss(img):
        with torch.no_grad():
            return float(topk_loss(view.scores(net(view.prepare(img))), s.loss.k))

    adv = pgd([net], x, boxes, s, views=[view])
    assert loss(adv) < loss(x)


def test_hinge_needs_a_view_that_supports_it():
    x, boxes = fixture_image_with_plate()
    with pytest.raises(ValueError, match="hinge"):
        pgd([FakeEndToEnd()], x, boxes, settings(loss="hinge"), views=[EndToEndView()])
