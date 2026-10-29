import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from plate_cloak.config import BASE_CONFIG, load_config, variants


def write(path, text):
    path.write_text(text)
    return path


def test_base_loads_with_expected_values():
    cfg = load_config(BASE_CONFIG)
    assert cfg.attack.steps == 40
    assert cfg.attack.eps.y == 2
    assert cfg.eval.conf == 0.25
    assert cfg.data.min_plate_h == 11


def test_cli_override_is_typed():
    cfg = load_config(BASE_CONFIG, overrides=["attack.eps.cb=6", "attack.space=rgb"])
    assert cfg.attack.eps.cb == 6 and isinstance(cfg.attack.eps.cb, int)
    assert cfg.attack.space == "rgb"


def test_list_override():
    cfg = load_config(BASE_CONFIG, overrides=["models.targets=[{name: yolov8s, seed: 0}]"])
    assert OmegaConf.to_container(cfg.models.targets) == [{"name": "yolov8s", "seed": 0}]


def test_override_and_variant(tmp_path):
    sweep = write(tmp_path / "s.yaml", "id: s\nvariants:\n  cb4: {attack.eps.cb: 4}\n")
    cfg = load_config(sweep, overrides=["attack.steps=10"], variant="cb4")
    assert cfg.attack.eps.cb == 4 and cfg.attack.steps == 10 and cfg.attack.eps.y == 2


def test_sweep_shared_overrides_apply_to_every_variant(tmp_path):
    sweep = write(
        tmp_path / "s.yaml",
        "id: s\noverrides: {attack.steps: 5}\nvariants:\n  a: {attack.eps.cb: 4}\n  b: {}\n",
    )
    assert load_config(sweep, variant="a").attack.steps == 5
    assert load_config(sweep, variant="b").attack.steps == 5


def test_cli_override_beats_variant(tmp_path):
    sweep = write(tmp_path / "s.yaml", "id: s\nvariants:\n  cb4: {attack.eps.cb: 4}\n")
    assert load_config(sweep, overrides=["attack.eps.cb=12"], variant="cb4").attack.eps.cb == 12


def test_unknown_override_key_raises():
    with pytest.raises(KeyError, match="attack.epz.cb"):
        load_config(BASE_CONFIG, overrides=["attack.epz.cb=4"])


def test_unknown_key_in_variant_raises(tmp_path):
    sweep = write(tmp_path / "s.yaml", "id: s\nvariants:\n  bad: {attack.epz.cb: 4}\n")
    with pytest.raises(KeyError, match="attack.epz.cb"):
        load_config(sweep, variant="bad")


def test_override_without_equals_raises():
    with pytest.raises(ValueError, match="key=value"):
        load_config(BASE_CONFIG, overrides=["attack.eps.cb"])


def test_sweep_requires_known_variant(tmp_path):
    sweep = write(tmp_path / "s.yaml", "id: s\nvariants:\n  cb4: {attack.eps.cb: 4}\n")
    with pytest.raises(ValueError, match="variant"):
        load_config(sweep)
    with pytest.raises(KeyError, match="cb5"):
        load_config(sweep, variant="cb5")


def test_variant_on_plain_config_raises():
    with pytest.raises(ValueError, match="variant"):
        load_config(BASE_CONFIG, variant="cb4")


def test_variants_lists_names_in_file_order(tmp_path):
    sweep = write(tmp_path / "s.yaml", "id: s\nvariants:\n  rgb8: {}\n  cb4: {}\n  cb8: {}\n")
    assert variants(sweep) == ["rgb8", "cb4", "cb8"]


def test_full_config_with_missing_key_raises(tmp_path):
    cfg = OmegaConf.to_container(load_config(BASE_CONFIG))
    del cfg["attack"]["steps"]
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    with pytest.raises(KeyError, match="attack.steps"):
        load_config(path)


def test_config_is_read_only():
    cfg = load_config(BASE_CONFIG)
    with pytest.raises(ReadonlyConfigError):
        cfg.attack.steps = 1
