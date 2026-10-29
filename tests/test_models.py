import json
from pathlib import Path

import pytest

from plate_cloak.models import finetune as ft
from plate_cloak.models import zoo


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Point weights/ and runs/finetune/ at a temp folder so tests never touch real files."""
    monkeypatch.setattr(zoo, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ft, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


class FakeModel:
    """Stands in for ultralytics YOLO/RTDETR: records calls and writes what training would."""

    calls: list = []

    def __init__(self, weights):
        self.weights = weights
        self.model = type("Net", (), {"parameters": lambda self: [_P(3), _P(4)]})()

    def train(self, **kwargs):
        FakeModel.calls.append({"weights": self.weights, **kwargs})
        if kwargs.get("resume"):
            run = Path(self.weights).parents[1]  # resume continues in the checkpoint's own run
        else:
            run = Path(kwargs["project"]) / kwargs["name"]  # Ultralytics writes to project/name
        (run / "weights").mkdir(parents=True, exist_ok=True)
        (run / "weights" / "best.pt").write_bytes(b"best")
        (run / "results.csv").write_text(
            "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.70,0.40\n2,0.85,0.50\n3,0.80,0.48\n"
        )


class _P:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


@pytest.fixture
def fake(monkeypatch):
    FakeModel.calls = []
    monkeypatch.setattr(ft, "model_class", lambda family: FakeModel)
    return FakeModel


def test_zoo_lists_every_planned_model():
    assert set(zoo.ZOO) == {"yolov8n", "yolov8s", "yolov8l", "yolov10l", "rtdetr-l"}
    assert zoo.ZOO["rtdetr-l"].family == "rtdetr" and zoo.ZOO["yolov10l"].family == "yolo"


def test_weights_path(dirs):
    assert zoo.weights_path("rtdetr-l", 2) == dirs / "weights" / "rtdetr-l_s2.pt"


def test_missing_weights_message_names_command(dirs):
    with pytest.raises(FileNotFoundError, match="finetune --model yolov10l --seed 1"):
        zoo.load_detector("yolov10l", 1)


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="yolov99"):
        zoo.weights_path("yolov99", 0)


def test_fresh_run_trains_from_coco_weights_without_early_stopping(dirs, fake):
    out = ft.finetune("yolov8s", 0, dirs / "data.yaml", epochs=3, batch=2, workers=0, device="cpu")
    call = fake.calls[0]
    assert call["weights"].endswith("weights/init/yolov8s.pt")
    assert call["patience"] == 0 and call["seed"] == 0 and call["deterministic"] is True
    assert call["epochs"] == 3 and call["imgsz"] == 640 and call["name"] == "yolov8s_s0"
    assert out == zoo.weights_path("yolov8s", 0) and out.read_bytes() == b"best"


def test_summary_records_best_validation_map(dirs, fake):
    ft.finetune("yolov8s", 1, dirs / "data.yaml", epochs=3, batch=2, workers=0, device="cpu")
    summary = json.loads(zoo.weights_path("yolov8s", 1).with_suffix(".json").read_text())
    assert summary["best_val_map50"] == pytest.approx(0.85)
    assert summary["best_epoch"] == 2 and summary["epochs_run"] == 3
    assert summary["parameters"] == 7 and summary["seed"] == 1


def test_resume_chosen_when_unfinished_checkpoint_exists(dirs, fake, monkeypatch):
    last = ft.RUNS_DIR / "yolov8s_s0" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"partial")
    monkeypatch.setattr(ft, "checkpoint_epoch", lambda path: 41)  # interrupted at epoch 41
    ft.finetune("yolov8s", 0, dirs / "data.yaml", epochs=100, batch=2, workers=0, device="cpu")
    assert fake.calls[0]["resume"] is True and fake.calls[0]["weights"] == str(last)


def test_finished_run_is_exported_not_retrained(dirs, fake, monkeypatch):
    run = ft.RUNS_DIR / "yolov8s_s0"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"done")
    (run / "weights" / "best.pt").write_bytes(b"best")
    (run / "results.csv").write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.9,0.6\n")
    monkeypatch.setattr(ft, "checkpoint_epoch", lambda path: -1)  # Ultralytics marks done as -1
    out = ft.finetune("yolov8s", 0, dirs / "data.yaml", epochs=1, batch=2, workers=0, device="cpu")
    assert fake.calls == [] and out.read_bytes() == b"best"


def test_timing_run_never_touches_real_runs_or_weights(dirs, fake, monkeypatch):
    # FakeModel writes results without a "time" column, so give it one for this test.
    real_train = FakeModel.train

    def train_with_time(self, **kwargs):
        real_train(self, **kwargs)
        csv_path = Path(kwargs["project"]) / kwargs["name"] / "results.csv"
        csv_path.write_text(
            "epoch,time,metrics/mAP50(B),metrics/mAP50-95(B)\n1,90,0.1,0.1\n2,150,0.2,0.1\n"
        )

    monkeypatch.setattr(FakeModel, "train", train_with_time)
    out = ft.finetune("rtdetr-l", 0, dirs / "data.yaml", 2, 2, 0, "cpu", timing=True)
    assert out == ft.RUNS_DIR.parent / "timing" / "rtdetr-l_s0"
    assert not (ft.RUNS_DIR / "rtdetr-l_s0").exists()
    assert not zoo.weights_path("rtdetr-l", 0).exists()


def test_already_exported_model_is_skipped(dirs, fake):
    out = zoo.weights_path("yolov8s", 0)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"old")
    out.with_suffix(".json").write_text("{}")
    assert ft.finetune("yolov8s", 0, dirs / "data.yaml", 1, 2, 0, "cpu") == out
    assert fake.calls == [] and out.read_bytes() == b"old"
