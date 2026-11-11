import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from detectors import TinyDetector
from PIL import Image

import plate_cloak.attack.stage as attack_stage
import plate_cloak.evaluation.stage as stage
from plate_cloak import cli
from plate_cloak.attack.stage import fingerprint
from plate_cloak.config import BASE_CONFIG, load_config
from plate_cloak.evaluation.stage import evaluate, predict_boxes

IDS = ["img_a", "img_b", "img_c"]
# Each fixture image has one plate at label "0 0.5 0.5 0.5 0.3" on a 160 x 120 image.
PLATE = np.array([40.0, 42.0, 120.0, 78.0])
TARGETS = "models.targets=[{name: yolov8s, seed: 0}, {name: rtdetr-l, seed: 1}]"
COCO_YOLOV8S = Path(__file__).resolve().parents[1] / "weights" / "init" / "yolov8s.pt"


# --- stand-ins -------------------------------------------------------------------------------


class _Boxes:
    def __init__(self, xyxy, conf):
        import torch

        self.xyxy = torch.tensor(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.conf = torch.tensor(np.asarray(conf, dtype=np.float32).reshape(-1))


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)


class BrightnessDetector:
    """Stand-in for an Ultralytics model: one box on the plate, scored by how bright it is.

    The fake attack paints the plate black, so attacked images score 0 and the plate is missed.
    `strength` scales the score, so two models differ. Records what predict() was called with.
    """

    def __init__(self, strength: float):
        self.strength = strength
        self.calls = []

    def predict(self, source, imgsz, conf, verbose, device):
        self.calls.append({"source": source, "imgsz": imgsz, "conf": conf, "device": device})
        assert isinstance(source, Image.Image) and source.mode == "RGB"
        x1, y1, x2, y2 = PLATE.astype(int)
        score = self.strength * np.asarray(source)[y1:y2, x1:x2].mean() / 255
        if score < conf:
            return [_Result(np.zeros((0, 4)), np.zeros(0))]
        return [_Result([PLATE], [score])]


def tiny_cfg(tmp_path, *overrides):
    """Three small images with one plate each, two targets, on CPU."""
    root = tmp_path / "data"
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    for i, image_id in enumerate(IDS):
        rng = np.random.default_rng(i)
        pixels = rng.integers(100, 256, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / "test" / f"{image_id}.jpg", quality=95)
        (root / "labels" / "test" / f"{image_id}.txt").write_text("0 0.5 0.5 0.5 0.3\n")
    eligible = tmp_path / "eligible.txt"
    eligible.write_text("\n".join(IDS) + "\n")
    return load_config(
        BASE_CONFIG,
        overrides=[
            f"data.root={root}",
            f"data.eligible_list={eligible}",
            f"output.cache_root={tmp_path / 'cache'}",
            f"output.runs_root={tmp_path / 'runs'}",
            "data.imgsz=160",
            "device=cpu",
            "eval.bootstrap.n=200",
            TARGETS,
            *overrides,
        ],
    )


def fake_attack_cache(cfg, blacken=("img_a", "img_b", "img_c")) -> Path:
    """A finished attack cache whose images have the plate painted black (for the listed ids)."""
    from plate_cloak.data.eval_images import load_by_id

    out = Path(cfg.output.cache_root) / fingerprint(cfg, IDS)
    out.mkdir(parents=True)
    for image_id in IDS:
        img = np.array(load_by_id(cfg, image_id).image)
        if image_id in blacken:
            x1, y1, x2, y2 = PLATE.astype(int)
            img[y1:y2, x1:x2] = 0
        Image.fromarray(img).save(out / f"{image_id}.png")
    max_change = {"R": 8.0, "G": 8.0, "B": 8.0, "Y": 2.0, "Cb": 8.0, "Cr": 8.0}
    (out / "done.json").write_text(json.dumps({"n_images": 3, "max_change": max_change}))
    return out


@pytest.fixture
def detectors(monkeypatch):
    """Patch model loading: yolov8s gets full-strength scores, rtdetr-l half-strength."""
    made = {}

    def load(name, seed):
        made[(name, seed)] = BrightnessDetector(1.0 if name == "yolov8s" else 0.5)
        return made[(name, seed)]

    monkeypatch.setattr(stage, "load_target", load)
    return made


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


# --- predict_boxes on a real model (skipped when the COCO weights are not on disk) ------------


def _bus():
    import ultralytics

    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    return Image.open(path).convert("RGB")


@pytest.fixture(scope="module")
def yolov8s_coco():
    if not COCO_YOLOV8S.exists():
        pytest.skip(f"{COCO_YOLOV8S} not downloaded")
    from ultralytics import YOLO

    return YOLO(str(COCO_YOLOV8S))


def _top_scores(result, k=5):
    return np.sort(result.boxes.conf.cpu().numpy())[::-1][:k]


def test_pil_and_bgr_array_predictions_agree(yolov8s_coco):
    img = _bus()
    _, scores = predict_boxes(yolov8s_coco, img, device="cpu")
    bgr = np.asarray(img)[:, :, ::-1].copy()
    ref = yolov8s_coco.predict(source=bgr, imgsz=640, conf=0.001, verbose=False, device="cpu")
    np.testing.assert_allclose(np.sort(scores)[::-1][:5], _top_scores(ref[0]), atol=1e-4)


def test_rgb_array_differs_which_is_why_we_use_pil(yolov8s_coco):
    img = _bus()
    _, scores = predict_boxes(yolov8s_coco, img, device="cpu")
    rgb = yolov8s_coco.predict(
        source=np.asarray(img), imgsz=640, conf=0.001, verbose=False, device="cpu"
    )
    assert not np.allclose(np.sort(scores)[::-1][:5], _top_scores(rgb[0]), atol=1e-4)


# --- predict_boxes with a stand-in -----------------------------------------------------------


def test_predict_boxes_shapes_and_settings():
    det = BrightnessDetector(1.0)
    img = Image.new("RGB", (160, 120), (255, 255, 255))
    boxes, scores = predict_boxes(det, img, imgsz=160, device="cpu")
    assert boxes.shape == (1, 4) and scores.shape == (1,) and boxes.dtype == np.float64
    assert det.calls[0]["conf"] == 0.001 and det.calls[0]["imgsz"] == 160

    boxes, scores = predict_boxes(det, Image.new("RGB", (160, 120)), imgsz=160, device="cpu")
    assert boxes.shape == (0, 4) and scores.shape == (0,)


# --- evaluate --------------------------------------------------------------------------------


def test_metrics_for_a_fully_successful_attack(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    m = evaluate(cfg, fake_attack_cache(cfg))
    yolo = m["targets"]["yolov8s_s0"]
    assert yolo["clean"]["recall50"]["point"] == 1.0 and yolo["clean"]["map50"]["point"] == 1.0
    assert yolo["attacked"]["recall50"]["point"] == 0.0 and yolo["attacked"]["map50"]["point"] == 0
    assert yolo["drop_map50"]["point"] == 1.0 and yolo["drop_recall50"]["point"] == 1.0
    assert set(m["targets"]) == {"yolov8s_s0", "rtdetr-l_s1"}
    assert m["n_images"] == 3 and m["attack"]["fingerprint"] == fingerprint(cfg, IDS)
    assert m["attack"]["max_change"]["Y"] == 2.0
    assert "git_commit" in m["provenance"]


def test_drop_is_paired_clean_minus_attacked(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    m = evaluate(cfg, fake_attack_cache(cfg, blacken=("img_a",)))  # one plate of three hidden
    t = m["targets"]["yolov8s_s0"]
    assert t["drop_recall50"]["point"] == pytest.approx(1 / 3)
    clean, attacked = t["clean"]["recall50"]["point"], t["attacked"]["recall50"]["point"]
    assert t["drop_recall50"]["point"] == pytest.approx(clean - attacked)
    assert t["drop_recall50"]["low"] >= 0  # paired: no resample can show a negative drop here


def test_low_scores_count_for_ap_but_not_recall(tmp_path, detectors):
    # rtdetr-l scores ~0.35 on clean plates in these images, yolov8s ~0.7; at conf 0.5 only
    # yolov8s "finds" them, but both rank them first, so both have AP 1.
    cfg = tiny_cfg(tmp_path, "eval.conf=0.5")
    m = evaluate(cfg, fake_attack_cache(cfg))
    rt = m["targets"]["rtdetr-l_s1"]["clean"]
    assert rt["map50"]["point"] == 1.0 and rt["recall50"]["point"] == 0.0
    assert m["targets"]["yolov8s_s0"]["clean"]["recall50"]["point"] == 1.0


def test_predictions_use_pil_images_at_the_eval_size(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    evaluate(cfg, fake_attack_cache(cfg))
    calls = detectors[("yolov8s", 0)].calls
    assert len(calls) == 6  # 3 images x (clean, attacked)
    assert all(c["imgsz"] == 160 and c["device"] == "cpu" and c["conf"] == 0.001 for c in calls)


def test_rows_cover_every_plate_in_the_same_order_for_every_model(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    run_dir = tmp_path / "out"
    evaluate(cfg, fake_attack_cache(cfg), run_dir)
    plates = read_csv(run_dir / "per_plate.csv")
    assert set(plates[0]) == {"image_id", "plate_idx", "model", "seed", "condition", "found"}
    assert len(plates) == 3 * 2 * 2  # images x models x conditions, found or not
    for model in ("yolov8s", "rtdetr-l"):
        for condition in ("clean", "attacked"):
            rows = [r for r in plates if r["model"] == model and r["condition"] == condition]
            assert [r["image_id"] for r in rows] == IDS
    dets = read_csv(run_dir / "per_detection.csv.gz")
    assert set(dets[0]) == {"image_id", "model", "seed", "condition", "score", "is_tp"}
    assert {r["condition"] for r in dets} == {"clean"}  # attacked plates gave no detections
    assert json.loads((run_dir / "metrics.json").read_text())["n_images"] == 3


def test_plate_idx_is_the_label_row_and_ignored_plates_are_left_out(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    # img_a gets a tiny plate as its first label row: it is ignored, so the real plate is row 1.
    label = Path(cfg.data.root) / "labels" / "test" / "img_a.txt"
    label.write_text("0 0.1 0.1 0.05 0.05\n0 0.5 0.5 0.5 0.3\n")
    run_dir = tmp_path / "out"
    evaluate(cfg, fake_attack_cache(cfg), run_dir)
    rows = [r for r in read_csv(run_dir / "per_plate.csv") if r["image_id"] == "img_a"]
    assert {r["plate_idx"] for r in rows} == {"1"}


def test_run_folder_is_created_only_at_the_end(tmp_path, detectors, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    adv = fake_attack_cache(cfg)
    run_dir = tmp_path / "out"
    monkeypatch.setattr(stage, "bootstrap_ci", lambda *a, **k: 1 / 0)  # crash after predicting
    with pytest.raises(ZeroDivisionError):
        evaluate(cfg, adv, run_dir)
    assert not run_dir.exists()  # nothing half-written blocks a rerun under the same name


def test_existing_run_folder_is_refused_before_any_work(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        evaluate(cfg, fake_attack_cache(cfg), run_dir)
    assert detectors == {}


def test_unfinished_cache_is_refused(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    adv = fake_attack_cache(cfg)
    (adv / "done.json").unlink()
    with pytest.raises(RuntimeError, match="not finished"):
        evaluate(cfg, adv)


def test_cache_for_other_settings_is_refused(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    adv = fake_attack_cache(cfg)
    other = load_config(BASE_CONFIG, [*_tiny_overrides(cfg), "attack.eps.cb=8"])
    with pytest.raises(RuntimeError, match="different attack settings"):
        evaluate(other, adv)


def _tiny_overrides(cfg):
    return [
        f"data.root={cfg.data.root}",
        f"data.eligible_list={cfg.data.eligible_list}",
        f"output.cache_root={cfg.output.cache_root}",
        f"output.runs_root={cfg.output.runs_root}",
        "data.imgsz=160",
        "device=cpu",
        "eval.bootstrap.n=200",
        TARGETS,
    ]


# --- CLI -------------------------------------------------------------------------------------


def test_cli_evaluate_writes_a_named_run(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    fake_attack_cache(cfg)
    assert cli.main(["evaluate", "--name", "t1", *_tiny_overrides(cfg)]) == 0
    run_dir = tmp_path / "runs" / "t1"
    names = {p.name for p in run_dir.iterdir()}
    assert names == {"config.yaml", "metrics.json", "per_detection.csv.gz", "per_plate.csv"}
    assert load_config(run_dir / "config.yaml").eval.bootstrap.n == 200


def test_cli_evaluate_without_cache_says_to_attack_first(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="plate_cloak attack"):
        cli.main(["evaluate", "--name", "t1", *_tiny_overrides(cfg)])


def test_cli_refuses_an_existing_run_before_doing_any_work(tmp_path, detectors):
    cfg = tiny_cfg(tmp_path)
    fake_attack_cache(cfg)
    (tmp_path / "runs" / "t1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        cli.main(["evaluate", "--name", "t1", *_tiny_overrides(cfg)])
    assert detectors == {}  # no model was loaded


def test_cli_run_attacks_then_evaluates(tmp_path, detectors, monkeypatch):
    monkeypatch.setattr(attack_stage, "load_surrogates", lambda cfg: [TinyDetector()])
    monkeypatch.setattr(attack_stage, "weights_digests", lambda cfg: {"yolov8s_s0": "weights-v1"})
    cfg = tiny_cfg(tmp_path)
    assert cli.main(["run", "--name", "r1", *_tiny_overrides(cfg), "attack.steps=2"]) == 0
    m = json.loads((tmp_path / "runs" / "r1" / "metrics.json").read_text())
    expected = load_config(BASE_CONFIG, [*_tiny_overrides(cfg), "attack.steps=2"])
    assert m["attack"]["fingerprint"] == fingerprint(expected, IDS)


def test_rerun_gives_byte_identical_outputs(tmp_path, detectors, monkeypatch):
    monkeypatch.setattr(stage, "provenance", lambda: {"git_commit": "x"})  # has a timestamp
    cfg = tiny_cfg(tmp_path)
    adv = fake_attack_cache(cfg)
    evaluate(cfg, adv, tmp_path / "a")
    evaluate(cfg, adv, tmp_path / "b")
    for name in ("per_detection.csv.gz", "per_plate.csv", "metrics.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
