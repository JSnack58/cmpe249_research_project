"""Per-family views: how PGD feeds each detector and reads its plate scores (milestone 2b)."""

import numpy as np
import pytest
import torch
from detectors import FakeEndToEnd, FakeRTDETR, TinyDetector
from test_pgd import fixture_image_with_plate, settings

from plate_cloak.attack.losses import topk_loss
from plate_cloak.attack.pgd import pgd, predict_padding
from plate_cloak.attack.views import DetectView, RTDETRView, view_for


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


def test_view_for_picks_by_family_and_refuses_an_end_to_end_head():
    # Our YOLOv10l checkpoints are not end-to-end (predict() runs them with NMS, like YOLOv8),
    # but one that is would return a top-k table first: reading it as YOLOv8's layout is silent.
    assert isinstance(view_for("yolov10l", TinyDetector(), 640), DetectView)
    assert isinstance(view_for("yolov8s", TinyDetector(), 640), DetectView)
    assert isinstance(view_for("rtdetr-l", FakeRTDETR(), 640), RTDETRView)
    with pytest.raises(NotImplementedError, match="end-to-end"):
        view_for("yolov10l", FakeEndToEnd(), 640)


@pytest.mark.parametrize(
    ("net", "view"),
    [
        (TinyDetector(), DetectView()),
        (FakeRTDETR(), RTDETRView(256)),
    ],
    ids=["detect", "rtdetr"],
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
        pgd([FakeRTDETR()], x, boxes, settings(loss="hinge"), views=[RTDETRView(256)])


def test_rtdetr_attack_scores_add_the_encoders_proposals():
    # The decoder keeps refining whatever the encoder proposes, so pushing only the decoder's
    # scores leaves plates the encoder still proposes (milestone 2b strength check, 2026-11-12).
    net, x = FakeRTDETR(), torch.rand(1, 3, 64, 64)
    out, view = net(x), RTDETRView(64)
    expected = torch.cat([out[1][1][-1, 0, :, 0].sigmoid(), out[1][3][0, :, 0].sigmoid()])
    assert torch.allclose(view.attack_scores(out), expected)
    assert torch.allclose(
        view.scores(out), out[1][1][-1, 0, :, 0].sigmoid()
    )  # what evaluation keeps


def test_detect_attack_scores_are_its_scores():
    net, x = TinyDetector(), torch.rand(1, 3, 64, 64)
    out = net(x)
    assert torch.equal(DetectView().attack_scores(out), DetectView().scores(out))
