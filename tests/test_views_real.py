"""The attack sees what predict() sees, on the real fine-tuned models (milestone 2b, Task 2).

Slow: loads seed 0 of each detector on the CPU. Run with `pytest -m slow tests/test_views_real.py`.

For each family, predict() is run the way evaluation runs it, with hooks on the network that
capture the exact input tensor Ultralytics built and the network's raw output. Then:
- the view's own input from the same image must be within one 8-bit level of predict()'s;
- on predict()'s own input, the scores the view reads must match predict()'s network output to
  0.001, both from the network predict() ran (fused) and from the unfused one the attack loads:
  this checks the reading alone;
- on the view's input, to 0.05, compared sorted: RT-DETR's scores move by about 0.01 when its
  input moves by a rounding step (cv2's 8-bit resize vs. a float one), and its query selection
  can reorder the queries; this records how far the two inputs' differences carry;
- every detection predict() keeps above the evaluation threshold must be one of the scores the
  view reads (to 0.01), so the loss pushes on the scores evaluation counts. RT-DETR has no NMS
  (one score per kept detection), so its kept scores must equal the view's top scores exactly that
  far down; YOLOv8 and our YOLOv10 checkpoints run NMS, which drops overlapping candidates.
"""

import numpy as np
import pytest
import torch

from plate_cloak.attack.views import view_for
from plate_cloak.config import BASE_CONFIG, load_config
from plate_cloak.data.eval_images import eval_image_ids, load_by_id
from plate_cloak.models.zoo import load_detector, weights_path

CONF = 0.25  # eval.conf
N_IMAGES = 3


@pytest.fixture(scope="module")
def images():
    cfg = load_config(BASE_CONFIG, ["data.split=val"])
    return cfg, [load_by_id(cfg, i).image for i in eval_image_ids(cfg)[:N_IMAGES]]


def _keep_first(store: dict, key: str):
    """A forward (pre-)hook that records the network's first input or output and returns None,
    so it never replaces what the network sees or returns."""

    def hook(module, args, output=None):
        store.setdefault(key, args[0] if key == "input" else output)

    return hook


@pytest.mark.slow
@pytest.mark.parametrize("name", ["rtdetr-l", "yolov10l", "yolov8s"])
def test_view_matches_predict(name, images):
    if not weights_path(name, 0).exists():
        pytest.skip(f"{weights_path(name, 0)} not found")
    cfg, pils = images
    model = load_detector(name, 0)
    # The first predict() builds the predictor, which may fuse the network into a new object and
    # keeps running that one; hook the network the predictor actually runs, not model.model.
    model.predict(source=pils[0], imgsz=cfg.data.imgsz, conf=CONF, verbose=False, device="cpu")
    net = model.predictor.model.model
    unfused = load_detector(name, 0).model.float().eval()  # as load_surrogates() loads it
    n_kept = 0
    for pil in pils:
        captured = {}
        pre = net.register_forward_pre_hook(_keep_first(captured, "input"))
        post = net.register_forward_hook(_keep_first(captured, "output"))
        try:
            result = model.predict(
                source=pil, imgsz=cfg.data.imgsz, conf=CONF, verbose=False, device="cpu"
            )
        finally:
            pre.remove()
            post.remove()
        assert set(captured) == {"input", "output"}, f"{name}: predict() ran another network"
        view = view_for(name, net, cfg.data.imgsz)

        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1)[None].float() / 255  # a copy
        theirs_in = captured["input"].float()
        ours_in = view.prepare(x)
        assert ours_in.shape == theirs_in.shape, (name, ours_in.shape, theirs_in.shape)
        assert (ours_in - theirs_in).abs().max() <= 1 / 255 + 1e-6, name

        theirs = view.scores(captured["output"])
        with torch.no_grad():
            on_theirs = view.scores(net(theirs_in))
            attack_reads = view_for(name, unfused, cfg.data.imgsz).scores(unfused(theirs_in))
            ours = view.scores(net(ours_in))
        assert on_theirs.shape == ours.shape == theirs.shape, name
        assert (on_theirs.sort().values - theirs.sort().values).abs().max() <= 1e-3, name
        # The attack's unfused network reads the same scores (YOLOv10: its one-to-many head).
        assert (attack_reads.sort().values - theirs.sort().values).abs().max() <= 1e-3, name
        assert (ours.sort().values - theirs.sort().values).abs().max() <= 0.05, name

        kept = np.sort(result[0].boxes.conf.cpu().numpy())[::-1]
        n_kept += len(kept)
        read = np.sort(theirs.numpy())[::-1]
        for score in kept:
            assert np.abs(read - score).min() <= 0.01, (name, score)
        if name == "rtdetr-l":  # NMS-free: one score per detection
            assert np.allclose(kept, read[: len(kept)], atol=0.01), name
    assert n_kept > 0, f"{name} found no plate in {len(pils)} images: the checks above were empty"
