from plate_cloak import cli


def test_prepare_data_uses_config_overrides(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        "plate_cloak.data.openimages.prepare_openimages",
        lambda root, workers: seen.update(root=root, workers=workers),
    )
    monkeypatch.setattr(
        "plate_cloak.data.eval_images.write_eligible_list",
        lambda root, out, imgsz, min_w, min_h: seen.update(out=out, min_h=min_h) or 7,
    )
    out = tmp_path / "eligible.txt"
    code = cli.main(
        ["prepare-data", "--workers", "4", f"data.eligible_list={out}", "data.min_plate_h=12"]
    )
    assert code == 0
    assert seen["workers"] == 4 and seen["out"] == out and seen["min_h"] == 12


def test_finetune_passes_config_and_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "plate_cloak.models.finetune.finetune", lambda **kwargs: seen.update(kwargs)
    )
    code = cli.main(
        ["finetune", "--model", "rtdetr-l", "--seed", "2", "--epochs", "2", "models.batch=8"]
    )
    assert code == 0
    assert seen["name"] == "rtdetr-l" and seen["seed"] == 2 and seen["epochs"] == 2
    assert seen["batch"] == 8 and seen["data_yaml"].name == "data.yaml"
    assert seen["timing"] is False


def test_finetune_rejects_unknown_model():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["finetune", "--model", "yolov99", "--seed", "0"])


def test_unbuilt_stage_reports_not_implemented(capsys):
    assert cli.main(["report"]) == 1
    assert "not implemented" in capsys.readouterr().out
