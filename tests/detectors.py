"""Stand-in detectors so attack tests run on CPU without real weights."""

import torch
from torch import nn


class TinyDetector(nn.Module):
    """Stand-in for YOLOv8's eval-mode output: returns (y, None) with y of shape (1, 5, A).

    One candidate per 8 x 8 cell; its box is a fixed 32 x 32 square on the cell center and its
    score is the sigmoid of a small conv net, so the score depends on the pixels around it.
    Like YOLOv8, it insists on input sides that are multiples of 32.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, stride=2, padding=1),
        )
        self.seen_shapes = []
        self.eval()

    def forward(self, x):
        self.seen_shapes.append(tuple(x.shape))
        assert x.shape[-2] % 32 == 0 and x.shape[-1] % 32 == 0, x.shape
        score = torch.sigmoid(self.net(x) + 2.0)  # (1, 1, H/8, W/8), starts fairly confident
        gh, gw = score.shape[-2:]
        cy, cx = torch.meshgrid(
            torch.arange(gh) * 8.0 + 4, torch.arange(gw) * 8.0 + 4, indexing="ij"
        )
        size = torch.full_like(cx, 32.0)
        boxes = torch.stack([cx, cy, size, size]).reshape(1, 4, -1)
        return torch.cat([boxes, score.reshape(1, 1, -1)], dim=1), None


class Scaled(nn.Module):
    """Wraps a detector and multiplies its plate scores by `factor`.

    Top-k only sums scores, so this multiplies the wrapped model's loss by `factor` and changes
    nothing else: the same candidates, the same gradient direction.
    """

    def __init__(self, net: nn.Module, factor: float):
        super().__init__()
        self.inner, self.factor = net, factor
        self.eval()

    def forward(self, x):
        y, extra = self.inner(x)
        return torch.cat([y[:, :4], y[:, 4:] * self.factor], dim=1), extra


class _Scorer(nn.Module):
    """A small conv net: one plate logit per 8 x 8 cell, so scores depend on the pixels."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, stride=4, padding=1),
        )

    def forward(self, x):
        return self.net(x).flatten(1) + 2.0  # (1, A) logits, fairly confident at the start


class _EndToEndHead(nn.Module):
    end2end = True


class FakeEndToEnd(nn.Module):
    """Stand-in for YOLOv10's eval-mode output: (top-300 table, raw dict).

    The raw dict holds the one-to-one head's logits as (1, 1, A). The table's row 4 (what the
    YOLOv8 reading takes) is a constant 0.5, so reading the wrong tensor gives different numbers.
    `model[-1]` is the head with `end2end = True`, as in an Ultralytics DetectionModel.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.scorer = _Scorer(seed)
        self.model = nn.ModuleList([self.scorer, _EndToEndHead()])
        self.eval()

    def forward(self, x):
        assert x.shape[-2] % 32 == 0 and x.shape[-1] % 32 == 0, x.shape
        logits = self.scorer(x)
        table = torch.full((1, 300, 6), 0.5)
        one2one = {"boxes": torch.zeros(1, 4, logits.shape[1]), "scores": logits[:, None]}
        return table, {"one2one": one2one, "one2many": one2one}


class FakeRTDETR(nn.Module):
    """Stand-in for RT-DETR's eval-mode output: (top-300 table, decoder outputs).

    The decoder outputs are (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta) with
    dec_scores as (layers=1, batch=1, queries, classes=1) logits and enc_scores, the selected
    proposals' encoder logits, as (batch=1, queries, classes=1). It insists on a square input,
    like RT-DETR's predictor, and its table row 4 is a constant 0.5.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.scorer = _Scorer(seed)
        self.model = nn.ModuleList([self.scorer])
        self.eval()

    def forward(self, x):
        assert x.shape[-2] == x.shape[-1], x.shape
        logits = self.scorer(x)
        q = logits.shape[1]
        dec_scores = logits.reshape(1, 1, q, 1)
        table = torch.full((1, 300, 6), 0.5)
        enc_scores = (logits - 1.0).reshape(1, q, 1)  # the encoder's proposal logits
        return table, (torch.zeros(1, 1, q, 4), dec_scores, torch.zeros(1, q, 4), enc_scores, None)
