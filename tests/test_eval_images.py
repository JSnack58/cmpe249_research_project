import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from plate_cloak.config import BASE_CONFIG, load_config
from plate_cloak.data.eval_images import (
    eval_image_ids,
    is_eligible,
    load_eval_image,
    write_eligible_list,
)
from plate_cloak.data.openimages import yolo_label_lines

SIZE = {"imgsz": 640, "min_w": 40, "min_h": 11}


def make_image(path, size, color=(90, 90, 90), mode="RGB", exif_orientation=None):
    img = Image.new(mode, size, color if mode == "RGB" else 90)
    if exif_orientation is None:
        img.save(path)
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        img.save(path, exif=exif)
    return path


def make_label(path, rows):
    """rows of (cx, cy, w, h), normalized, as written by prepare-data."""
    path.write_text("".join(f"0 {cx} {cy} {w} {h}\n" for cx, cy, w, h in rows))
    return path


def test_yolo_label_lines_from_open_images_boxes():
    # Open Images boxes are normalized (x1, y1, x2, y2); YOLO wants class cx cy w h.
    assert yolo_label_lines([(0.2, 0.4, 0.6, 0.5)]) == ["0 0.400000 0.450000 0.400000 0.100000"]


def test_long_side_and_box_scaling(tmp_path):
    img = make_image(tmp_path / "a.jpg", (1000, 500))
    ev = load_eval_image(img, make_label(tmp_path / "a.txt", [(0.5, 0.5, 0.1, 0.1)]), **SIZE)
    assert ev.image.size == (640, 320)
    np.testing.assert_allclose(ev.boxes[0], [288, 144, 352, 176], atol=1)
    assert ev.image_id == "a"


def test_exif_rotated_image_boxes_align(tmp_path):
    # The photo as displayed is 200x100 with a red block where the label says the plate is.
    shown = Image.new("RGB", (200, 100), (90, 90, 90))
    shown.paste((255, 0, 0), (80, 40, 120, 60))
    # Store it rotated, with EXIF orientation 6 ("rotate 90 degrees clockwise to display").
    exif = Image.Exif()
    exif[0x0112] = 6
    shown.transpose(Image.Transpose.ROTATE_90).save(tmp_path / "r.jpg", exif=exif, quality=95)
    label = make_label(tmp_path / "r.txt", [(0.5, 0.5, 0.2, 0.2)])

    ev = load_eval_image(tmp_path / "r.jpg", label, **SIZE)
    assert ev.image.size == (640, 320)
    x1, y1, x2, y2 = ev.boxes[0].astype(int)
    inside = np.asarray(ev.image)[y1 + 4 : y2 - 4, x1 + 4 : x2 - 4]
    assert inside[..., 0].mean() > 200 and inside[..., 1].mean() < 60


def test_small_plates_are_ignored_not_dropped(tmp_path):
    img = make_image(tmp_path / "b.jpg", (640, 480))
    # 80x40 px plate counts; 20x8 px plate is too small to read and becomes an ignore region.
    label = make_label(tmp_path / "b.txt", [(0.5, 0.5, 0.125, 0.0834), (0.2, 0.2, 0.03125, 0.0167)])
    ev = load_eval_image(img, label, **SIZE)
    assert ev.ignore.tolist() == [False, True]
    assert is_eligible(ev)


def test_image_with_only_small_plates_is_not_eligible(tmp_path):
    img = make_image(tmp_path / "c.jpg", (640, 480))
    ev = load_eval_image(img, make_label(tmp_path / "c.txt", [(0.5, 0.5, 0.03, 0.02)]), **SIZE)
    assert not is_eligible(ev)


def test_boxes_are_clipped_to_the_image(tmp_path):
    img = make_image(tmp_path / "d.jpg", (640, 480))
    ev = load_eval_image(img, make_label(tmp_path / "d.txt", [(0.98, 0.5, 0.1, 0.1)]), **SIZE)
    assert ev.boxes[0, 2] == 640


def test_grayscale_image_loads_as_rgb(tmp_path):
    img = make_image(tmp_path / "g.jpg", (640, 480), mode="L")
    ev = load_eval_image(img, make_label(tmp_path / "g.txt", [(0.5, 0.5, 0.2, 0.1)]), **SIZE)
    assert ev.image.mode == "RGB"


def test_eligible_list_is_sorted_and_filtered(tmp_path):
    root = tmp_path / "ds"
    (root / "images/test").mkdir(parents=True)
    (root / "labels/test").mkdir(parents=True)
    for image_id, w in [("zz", 0.2), ("aa", 0.2), ("tiny", 0.01)]:
        make_image(root / f"images/test/{image_id}.jpg", (640, 480))
        make_label(root / f"labels/test/{image_id}.txt", [(0.5, 0.5, w, 0.1)])
    out = tmp_path / "eligible.txt"
    assert write_eligible_list(root, out, **SIZE) == 2
    assert out.read_text().split() == ["aa", "zz"]


@pytest.fixture
def cfg_with_list(tmp_path):
    ids = [f"img{i:03d}" for i in range(50)]
    path = tmp_path / "eligible.txt"
    path.write_text("\n".join(ids) + "\n")
    cfg = OmegaConf.to_container(load_config(BASE_CONFIG))
    cfg["data"]["eligible_list"] = str(path)
    return OmegaConf.create(cfg), ids


def test_eval_ids_all_when_not_subsampled(cfg_with_list):
    cfg, ids = cfg_with_list
    assert eval_image_ids(cfg) == ids


def test_eval_ids_subsample_is_deterministic_and_sorted(cfg_with_list):
    cfg, ids = cfg_with_list
    cfg.data.n_eval = 10
    first = eval_image_ids(cfg)
    assert first == eval_image_ids(cfg) and len(first) == 10
    assert first == sorted(first) and set(first) <= set(ids)
