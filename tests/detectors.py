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
