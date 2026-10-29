# Milestone 1 (Core H1 Slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Exception for this project:** tasks marked **OWNER DRAFTS** are written by the repository owner, not by an agent.
> For those tasks the agent writes the tests first (Step 1), then stops and hands over; the owner writes the implementation; the agent reviews it against the tests and the spec, pushes back, and restructures if needed.
> The agent never writes the body of an OWNER DRAFTS function unless the owner asks.
> After every agent-written task, walk the owner through the code.

**Goal:** Produce the H1 result (clean vs. attacked plate mAP@0.5 with paired bootstrap CIs for YOLOv8s, YOLOv10l and RT-DETR-L) from a config-driven, reproducible pipeline on Open Images plates.

**Architecture:** A `plate_cloak` library of pure functions (tensors and plain arguments in, results out) plus thin stage commands (`prepare-data`, `finetune`, `attack`, `evaluate`, `run`, `report`) that read OmegaConf YAML configs and communicate through files: frozen splits, weights, a fingerprinted adversarial-image cache, and committed run folders.

**Tech Stack:** Python 3.12, PyTorch 2.11 (CUDA 12.8), Ultralytics 8.4.178, OmegaConf, NumPy, pandas, Pillow, torchmetrics (test oracle only), matplotlib, pytest, ruff.

**Spec:** [docs/superpowers/specs/2026-10-20-plate-cloak-experiment-design.md](../specs/2026-10-20-plate-cloak-experiment-design.md)

## Global Constraints

- Pinned, isolated conda environment from `environment.yml` (no access to base Anaconda packages); exact versions, e.g. `torch==2.11.0`, `ultralytics==8.4.178`.
- Before any GPU run: CPU core 6 offline (`echo 0 | sudo tee /sys/devices/system/cpu/cpu6/online /sys/devices/system/cpu/cpu22/online`, owner runs it), and jobs pinned with `taskset -c 0-5,7-21,23-31`. Confirm with the owner before launching any job longer than about 10 minutes.
- Datasets, weights and the adversarial cache are never committed; run folders under `results/runs/` are.
- Markdown: one sentence per line, no em dashes. Commits: no AI co-author or sign-off lines.
- Image size 640 (long side). Evaluation eligibility: largest plate at least 40 px wide and 11 px tall at the 640 input; smaller plates are ignore regions.
- Detection: confidence threshold 0.25 for every model; match at IoU >= 0.5.
- Bootstrap: 1,000 resamples, 95% percentile CI, seed 42, paired across models (same resample for every model).
- Attack: PGD, 40 steps, per-channel step = 2.5 x budget / 40, perturbation on every annotated plate box grown by 10% per side, evaluated on the 8-bit image.
- Budgets in 1/255 units: RGB 8; YCbCr Y 2 with Cb/Cr swept over 4, 8, 12; YCbCr projects to budget minus 0.5 (quantization margin).
- Main YCbCr budget = smallest Cb/Cr whose attacked YOLOv8s recall is within 0.05 of the RGB attack's (extend to 16, 20 if none).
- Fine-tuning: COCO-pretrained init, same split, imgsz 640, same epoch count for all models (starting point 100), `patience=0` (early stopping off), best-validation checkpoint, seeds 0, 1, 2, each family's Ultralytics optimizer defaults.
- Every model receives its own native Ultralytics preprocessing at evaluation: images go to `predict()` as PIL RGB images (never raw NumPy, which Ultralytics reads as BGR).

## Review Focus

1. **EXIF-rotated Open Images photos:** boxes must line up with the image as displayed; load every image through `PIL.ImageOps.exif_transpose`. Test in Task 3.
2. **RGB/BGR mix-up at prediction time:** passing an RGB NumPy array to Ultralytics silently swaps channels and lowers accuracy. Predictions must come from PIL images. Test in Task 11.
3. **Images or resamples with no countable plates, and models that detect nothing:** matching, AP, recall and the bootstrap must return defined values (no division by zero, no crash). Tests in Tasks 8, 9 and 10.
4. **Stale attack cache:** changing any attack-relevant setting or the image list must change the fingerprint; changing only evaluation settings must not. Test in Task 7.
5. **Machine crash mid-run:** fine-tuning resumes from `last.pt`; a run whose folder already exists refuses to start instead of overwriting. Tests in Tasks 2 and 4.

---

## File Structure

```
environment.yml, pyproject.toml, .github/workflows/ci.yml       Task 1
configs/base.yaml                                                Task 2
configs/sweeps/cbcr.yaml                                         Task 12 (owner)
splits/openimages_test_eligible.txt                              Task 3 (committed)
src/plate_cloak/
  __init__.py, __main__.py, cli.py                               Tasks 1, 2, 7, 11, 13
  config.py            load configs, overrides, variants         Task 2
  runs.py              run naming, run folders, provenance       Task 2
  data/openimages.py   download, YOLO labels, data.yaml          Task 3
  data/eval_images.py  eligibility, eval image loading           Task 3
  models/zoo.py        model registry, weights paths, loading    Task 4
  models/finetune.py   Ultralytics training wrapper              Task 4
  attack/colorspace.py RGB<->YCbCr, fit_to_gamut (owner)         Task 5
  attack/losses.py     suppression losses on raw outputs         Task 6
  attack/pgd.py        PGD, random control, budget audit         Task 6
  attack/stage.py      attack stage, fingerprint cache           Task 7
  evaluation/matching.py   IoU, per-image matching (owner)       Task 8
  evaluation/detection.py  AP and recall from matches            Task 9
  evaluation/stats.py      bootstrap CI (owner)                  Task 10
  evaluation/stage.py      evaluate stage, run outputs           Task 11
  analysis/report.py       H1 table, paired CI, chart            Task 13
run_ablations.sh                                                 Task 13
tests/test_<module>.py  one test file per module above
```

---

### Task 1: Isolated environment, packaging, CI (agent)

**Files:**
- Delete: the untracked early skeleton (`configs/`, `src/`, `pyproject.toml`, `environment.yml`, `.github/`); it predates the design and was never reviewed.
- Create: `environment.yml`, `pyproject.toml`, `.github/workflows/ci.yml`, `src/plate_cloak/__init__.py`, `src/plate_cloak/__main__.py`, `src/plate_cloak/cli.py`, `tests/test_package.py`
- Modify: `.gitignore` (add `cache/`)

**Interfaces:**
- Produces: package `plate_cloak` installed editable; `python -m plate_cloak --help` lists subcommands; `cli.main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_lists_stages(capsys):
    from plate_cloak import cli
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for stage in ["prepare-data", "finetune", "attack", "evaluate", "run", "report"]:
        assert stage in out
```

- [ ] **Step 2: Create the environment and confirm the test fails**
`environment.yml`: conda-forge channel, `python=3.12.7`, pip section with `--extra-index-url https://download.pytorch.org/whl/cu128`, `torch==2.11.0`, `torchvision==0.26.0`, `ultralytics==8.4.178`, `omegaconf==2.4.0`, `numpy==2.2.6`, `pandas==3.0.6`, `pillow==12.3.0`, `matplotlib==3.11.2`, `torchmetrics==1.9.0`, `pycocotools==2.0.11`, `scikit-image==0.26.0`, `pytest==9.1.1`, `ruff==0.17.1`, `-e .`.
Run: `conda env create -f environment.yml && conda run -n plate-cloak pytest tests/test_package.py -v`
Expected: FAIL (no `cli` module).

- [ ] **Step 3: Implement `cli.main` with argparse subcommands** (each subcommand stubbed to print "not implemented" and return 1 until its task lands). `pyproject.toml`: setuptools, `src` layout, ruff line length 100 with rules `E,F,I,B,UP,SIM`, pytest `testpaths=["tests"]`.

- [ ] **Step 4: Verify**
Run: `conda run -n plate-cloak pytest -v && conda run -n plate-cloak ruff check . && conda run -n plate-cloak python -c "import torch; print(torch.cuda.is_available())"`
Expected: PASS, no lint errors, `True`.
CI workflow: ubuntu, Python 3.12, `pip install -e ".[dev]"` with the CPU torch index, then `ruff check .`, `ruff format --check .`, `pytest`.

- [ ] **Step 5: Commit** `git commit -m "Set up isolated environment, package, and CI"`

---

### Task 2: Config loading and run folders (agent)

**Files:**
- Create: `configs/base.yaml`, `src/plate_cloak/config.py`, `src/plate_cloak/runs.py`, `tests/test_config.py`, `tests/test_runs.py`

**Interfaces:**
- Produces:
  - `load_config(config_path: Path, overrides: list[str] = (), variant: str | None = None) -> DictConfig`: loads `configs/base.yaml`, then `config_path` if it is not the base, then the named variant's overrides, then dotted CLI overrides (`"attack.eps.cb=6"`). Unknown keys raise `KeyError`.
  - `variants(config_path: Path) -> list[str]`: variant names in a sweep/ablation file.
  - `run_name(cfg: DictConfig, base: DictConfig, now: datetime) -> str`: `YYYY-MM-DD_HHMM_` plus `key-value` for each leaf that differs from base, joined by `_`, max 80 chars.
  - `create_run_dir(root: Path, name: str) -> Path`: creates `root/name`; raises `FileExistsError` if it exists.
  - `write_resolved_config(cfg: DictConfig, run_dir: Path) -> None` (writes `config.yaml`), `provenance() -> dict` (git commit, dirty flag, torch/ultralytics versions, GPU name, timestamp).
- Ablation/sweep file format (consumed by Task 12):
  ```yaml
  id: cbcr_sweep
  description: ...
  overrides: {models.targets: [{name: yolov8s, seed: 0}]}   # optional, shared by every variant
  variants:
    cb4: {attack.eps.cb: 4, attack.eps.cr: 4}
  ```
  Precedence (last wins): base.yaml < sweep `overrides` < variant < command line. *(Added during Task 2.)*
  `configs/base.yaml` also has a top-level `device: cuda:0` used by attack and evaluate. *(Added during Task 2.)*
- `configs/base.yaml` keys (all later tasks read these): `seed: 42`; `data: {root: data/openimages_plates, eligible_list: splits/openimages_test_eligible.txt, imgsz: 640, min_plate_w: 40, min_plate_h: 11, n_eval: null}`; `models: {surrogate: {name: yolov8s, seed: 0}, targets: [...], epochs: 100, batch: 16, workers: 8}`; `attack: {method: pgd, space: ycbcr, eps: {rgb: 8, y: 2, cb: 8, cr: 8}, steps: 40, step_factor: 2.5, margin: 0.10, mask: uniform, loss: {type: topk, k: 100, floor: 0.1}, quant_margin: 0.5}`; `eval: {conf: 0.25, iou: 0.5, bootstrap: {n: 1000, ci: 0.95}}`; `output: {runs_root: results/runs, cache_root: cache/adv}`.
  `models.targets` lists `{name, seed}` for yolov8s seeds 0-2, yolov10l seeds 0-2, rtdetr-l seeds 0-2.
  YOLOv8s seeds 1 and 2 are same-architecture transfer targets: they show how much an attack weakens from retraining alone. *(Plan decision.)*

- [ ] **Step 1: Write the failing tests**

```python
def test_override_and_variant(tmp_path):
    sweep = tmp_path / "s.yaml"
    sweep.write_text("id: s\nvariants:\n  cb4: {attack.eps.cb: 4}\n")
    cfg = load_config(sweep, overrides=["attack.steps=10"], variant="cb4")
    assert cfg.attack.eps.cb == 4 and cfg.attack.steps == 10 and cfg.attack.eps.y == 2

def test_unknown_override_key_raises():
    with pytest.raises(KeyError):
        load_config(BASE, overrides=["attack.epz.cb=4"])

def test_run_name_lists_differences():
    cfg = load_config(BASE, overrides=["attack.eps.cb=6"])
    assert run_name(cfg, load_config(BASE), datetime(2026, 11, 2, 14, 32)) == "2026-11-02_1432_attack.eps.cb-6"

def test_existing_run_dir_refuses(tmp_path):
    create_run_dir(tmp_path, "r1")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "r1")

def test_resolved_config_round_trips(tmp_path):
    cfg = load_config(BASE, overrides=["attack.eps.cb=6"])
    write_resolved_config(cfg, tmp_path)
    assert load_config(tmp_path / "config.yaml") == cfg
```

- [ ] **Step 2: Run** `pytest tests/test_config.py tests/test_runs.py -v`. Expected: FAIL (imports).
- [ ] **Step 3: Implement** with `OmegaConf.load`, `OmegaConf.update(..., merge=False)`, `OmegaConf.select` for key checks; a resolved config file loads as a full config, not as an overlay on base.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add config loading and run folders"`

---

### Task 3: Open Images data preparation and evaluation images (agent)

**Files:**
- Create: `src/plate_cloak/data/openimages.py` (absorbs `spikes/02_openimages/download_openimages_plates.py`, which stays as-is in the spike), `src/plate_cloak/data/eval_images.py`, `splits/openimages_test_eligible.txt`, `tests/test_eval_images.py`
- Modify: `src/plate_cloak/cli.py` (`prepare-data`)

**Interfaces:**
- Produces:
  - `prepare_openimages(out: Path, workers: int = 32) -> Path`: downloads and writes the YOLO dataset (as the spike script does), returns `data.yaml`. Images are written EXIF-transposed so labels match pixels for Ultralytics training too.
  - `@dataclass EvalImage: image_id: str; image: PIL.Image.Image (RGB, long side 640); boxes: np.ndarray (M,4) xyxy pixels; ignore: np.ndarray (M,) bool`
  - `load_eval_image(image_path: Path, label_path: Path, imgsz: int, min_w: float, min_h: float) -> EvalImage`: EXIF-transpose, convert RGB, resize long side to `imgsz` (bilinear), scale boxes, clip to image, mark `ignore` for boxes under `min_w` x `min_h`.
  - `is_eligible(ev: EvalImage) -> bool`: at least one non-ignored box.
  - `eval_image_ids(cfg) -> list[str]`: reads `splits/openimages_test_eligible.txt` (sorted IDs), subsampled to `cfg.data.n_eval` with `random.Random(cfg.seed)` if set.
- `prepare-data` writes `splits/openimages_test_eligible.txt` from the test split.

- [ ] **Step 1: Write the failing tests**

```python
def test_exif_rotated_image_boxes_align(tmp_path):
    # 200x100 image stored rotated with EXIF orientation 6; a red block is at the labelled box once displayed
    ev = load_eval_image(make_exif_rotated_fixture(tmp_path), label, imgsz=640, min_w=40, min_h=11)
    x1, y1, x2, y2 = ev.boxes[0].astype(int)
    assert np.asarray(ev.image)[y1:y2, x1:x2, 0].mean() > 200   # red channel high inside the box

def test_long_side_and_box_scaling(tmp_path):
    ev = load_eval_image(fixture_1000x500(tmp_path), label_cx05_w01, imgsz=640, min_w=40, min_h=11)
    assert ev.image.size == (640, 320)
    np.testing.assert_allclose(ev.boxes[0], [288, 144, 352, 176], atol=1)

def test_small_plates_are_ignored_not_dropped(tmp_path):
    ev = load_eval_image(fixture_two_plates(tmp_path), ..., imgsz=640, min_w=40, min_h=11)
    assert ev.ignore.tolist() == [False, True] and is_eligible(ev)

def test_grayscale_image_loads_as_rgb(tmp_path):
    assert load_eval_image(fixture_grayscale(tmp_path), ...).image.mode == "RGB"
```

- [ ] **Step 2: Run** `pytest tests/test_eval_images.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the functions above.
- [ ] **Step 4: Run** tests (PASS), then `python -m plate_cloak prepare-data`; expect 8,154 images, and an eligible list of roughly 1,700 test IDs (record the exact count in the commit message).
- [ ] **Step 5: Commit** `git add src/plate_cloak/data tests/test_eval_images.py splits/ && git commit -m "Add Open Images preparation and frozen eligible test list"`

---

### Task 4: Model zoo and fine-tuning, with a timing run (agent)

**Files:**
- Create: `src/plate_cloak/models/zoo.py`, `src/plate_cloak/models/finetune.py`, `tests/test_zoo.py`
- Modify: `src/plate_cloak/cli.py` (`finetune --model NAME --seed S [--epochs N]`)

**Interfaces:**
- Produces:
  - `ZOO: dict[str, ModelSpec]`, `ModelSpec(family: Literal["yolo", "rtdetr"], init: str)` for `yolov8n, yolov8s, yolov8l, yolov10l, rtdetr-l` with inits `yolov8n.pt, yolov8s.pt, yolov8l.pt, yolov10l.pt, rtdetr-l.pt`.
  - `weights_path(name: str, seed: int) -> Path`: `weights/{name}_s{seed}.pt`.
  - `load_detector(name: str, seed: int)`: `RTDETR` for family `rtdetr`, else `YOLO`; raises `FileNotFoundError` naming the `finetune` command if weights are missing.
  - `finetune(name: str, seed: int, data_yaml: Path, epochs: int, batch: int, workers: int, device: str) -> Path`: calls Ultralytics `train` with `seed=seed, deterministic=True, patience=0, imgsz=640, project="runs/finetune", name=f"{name}_s{seed}", exist_ok=True`; if `runs/finetune/{name}_s{seed}/weights/last.pt` exists and training is unfinished, resumes with `resume=True`; copies `best.pt` to `weights_path`; writes `weights/{name}_s{seed}.json` with final val mAP@0.5, epochs, parameter count, train args.

- [ ] **Step 1: Write the failing tests** (no GPU, no download)

```python
def test_weights_path():
    assert weights_path("rtdetr-l", 2) == Path("weights/rtdetr-l_s2.pt")

def test_missing_weights_message_names_command():
    with pytest.raises(FileNotFoundError, match="finetune --model yolov10l --seed 1"):
        load_detector("yolov10l", 1)

def test_resume_chosen_when_last_checkpoint_exists(tmp_path, monkeypatch):
    calls = record_train_calls(monkeypatch)          # patches ultralytics YOLO.train
    make_unfinished_run(tmp_path / "runs/finetune/yolov8s_s0")
    finetune("yolov8s", 0, DATA_YAML, epochs=100, batch=16, workers=8, device="cpu")
    assert calls[0]["resume"] is True
```

- [ ] **Step 2: Run** `pytest tests/test_zoo.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests (PASS). Then the **timing run** (core 6 offline first): `taskset -c 0-5,7-21,23-31 python -m plate_cloak finetune --model M --seed 0 --epochs 2` for `yolov8s`, `yolov10l`, `rtdetr-l`. Record seconds per epoch and projected totals for 100 epochs x 3 seeds in `docs/experiment-decisions.md`; if the projection exceeds 40 GPU-hours, stop and agree a new epoch count with the owner.
- [ ] **Step 5: Commit** `git commit -m "Add model zoo and fine-tuning wrapper; record timing"`
- [ ] **Step 6: Full fine-tunes (long job; confirm with the owner first).** 9 runs (3 models x seeds 0-2), either GPU via `device`. Afterwards check each RT-DETR validation curve has flattened; if not, raise the epoch count for every model equally and rerun.
`weights/` is git-ignored, so copy each `weights/*.json` summary (val mAP@0.5, parameter count, epochs) into a table in `docs/experiment-decisions.md` and commit that: `git commit -m "Record fine-tuning results"`.

---

### Task 5: YCbCr conversion and gamut fit (OWNER DRAFTS `fit_to_gamut`)

**Files:**
- Create: `src/plate_cloak/attack/colorspace.py`, `tests/test_colorspace.py`

**Interfaces:**
- Produces (agent writes the conversions; owner writes `fit_to_gamut`):
  - `rgb_to_ycbcr(x: Tensor) -> Tensor`, `ycbcr_to_rgb(x: Tensor) -> Tensor`: BT.601 full range, chroma offset 0.5, `(B,3,H,W)` in [0,1] (as in Spike 01).
  - `fit_to_gamut(x_rgb: Tensor, delta_ycc: Tensor) -> Tensor`: per-pixel scale `t` of shape `(B,1,H,W)` in [0,1] such that `ycbcr_to_rgb(rgb_to_ycbcr(x_rgb) + t * delta_ycc)` lies in [0,1] (within 1e-6), with `t = 1` wherever no scaling is needed. Spec section 6.3 describes the idea; the spike notebook's markdown explains the derivation.

- [ ] **Step 1: Agent writes the failing tests, then hands over**

```python
def test_round_trip():
    x = torch.rand(2, 3, 16, 16)
    assert (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max() < 1e-5

def test_fit_keeps_result_in_gamut_and_never_grows_noise():
    x = saturated_fixture()                       # mix of random, pure white, pure black pixels
    d = (torch.rand_like(x) * 2 - 1) * torch.tensor([2, 8, 8]).view(1, 3, 1, 1) / 255
    t = fit_to_gamut(x, d)
    out = ycbcr_to_rgb(rgb_to_ycbcr(x) + t * d)
    assert t.shape == (2, 1, 16, 16) and (t >= 0).all() and (t <= 1).all()
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6

def test_fit_is_identity_when_already_in_gamut():
    x = torch.full((1, 3, 4, 4), 0.5)
    d = torch.full_like(x, 1 / 255)
    assert torch.equal(fit_to_gamut(x, d), torch.ones(1, 1, 4, 4))

def test_zero_noise_component_does_not_produce_nan():
    x = torch.ones(1, 3, 2, 2)
    d = torch.zeros_like(x); d[:, 1] = 4 / 255
    assert torch.isfinite(fit_to_gamut(x, d)).all()
```

- [ ] **Step 2: Run** `pytest tests/test_colorspace.py -v`. Expected: conversion tests PASS, `fit_to_gamut` tests FAIL.
- [ ] **Step 3: OWNER implements `fit_to_gamut`.**
- [ ] **Step 4: Run** the tests (PASS), then agent review: correctness on edge cases (exactly 0 or 1 channels, zero noise components), vectorization (no Python loops over pixels), clarity. Owner revises.
- [ ] **Step 5: Commit** (owner's authorship) `git commit -m "Add YCbCr conversion and gamut fit"`

---

### Task 6: Losses, PGD, random control, budget audit (agent)

**Files:**
- Create: `src/plate_cloak/attack/losses.py`, `src/plate_cloak/attack/pgd.py`, `tests/test_pgd.py`

**Interfaces:**
- Consumes: Task 5 `rgb_to_ycbcr`, `ycbcr_to_rgb`, `fit_to_gamut`.
- Produces:
  - `topk_loss(raw: Tensor, k: int) -> Tensor` and `hinge_loss(raw: Tensor, floor: float, plate_mask_anchors: Tensor | None) -> Tensor`, where `raw` is the YOLOv8 Detect output `(1, 5, A)` (rows 0-3 box xywh pixels, row 4 sigmoid plate score).
  - `@dataclass AttackSettings` built by `AttackSettings.from_cfg(cfg.attack)`: `method, space, eps (Tensor (3,) in [0,1] units), steps, step_factor, margin, mask, loss, quant_margin`.
  - `plate_mask(h: int, w: int, boxes: np.ndarray, margin: float) -> Tensor (1,1,H,W)`: covers every box (including ignored ones), grown by `margin` per side.
  - `pgd(net: nn.Module, x: Tensor (1,3,H,W), boxes: np.ndarray, s: AttackSettings) -> Tensor`: pads H and W to multiples of 32 internally for the forward pass and crops back; for `space="ycbcr"` projects to `eps - quant_margin/255` and applies `fit_to_gamut` after every step; returns the adversarial image in [0,1].
  - `random_sign_noise(x, boxes, s, generator: torch.Generator) -> Tensor`.
  - `audit(clean_u8: np.ndarray, adv_u8: np.ndarray) -> dict[str, float]`: max absolute change in `R, G, B, Y, Cb, Cr` in 1/255 units over the whole image, computed on 8-bit inputs.

- [ ] **Step 1: Write the failing tests** (CPU, tiny stand-in detector)

```python
class TinyDetector(nn.Module):   # conv stack returning (y, None) with y of shape (1, 5, A), score = sigmoid of a conv map
    ...

def test_pgd_lowers_loss_and_stays_on_plates():
    x, boxes = fixture_image_with_plate()
    adv = pgd(TinyDetector(), x, boxes, settings(space="rgb"))
    assert topk_loss(raw_of(adv), 100) < topk_loss(raw_of(x), 100)
    outside = 1 - plate_mask(*x.shape[-2:], boxes, 0.10)
    assert ((adv - x).abs() * outside).max() == 0

def test_ycbcr_budget_holds_after_rounding_on_white_plate():
    x, boxes = fixture_white_plate()
    adv = pgd(TinyDetector(), x, boxes, settings(space="ycbcr", eps=(2, 8, 8)))
    a = audit(to_u8(x), to_u8(adv))
    assert a["Y"] <= 2.0 and a["Cb"] <= 8.0 and a["Cr"] <= 8.0

def test_rgb_budget_holds_after_rounding():
    a = audit(to_u8(x), to_u8(pgd(TinyDetector(), x, boxes, settings(space="rgb", eps=8))))
    assert max(a["R"], a["G"], a["B"]) <= 8.0

def test_non_multiple_of_32_input_round_trips_shape():
    x = torch.rand(1, 3, 427, 640)
    assert pgd(TinyDetector(), x, boxes_for(x), settings(steps=2)).shape == x.shape
```

- [ ] **Step 2: Run** `pytest tests/test_pgd.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Step update: `delta -= step * grad.sign()`, then `torch.clamp(delta, -bound, bound)`, then gamut fit; `bound = eps * plate_mask`.
- [ ] **Step 4: Run** tests. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add PGD with exact YCbCr budget, random control, and audit"`

---

### Task 7: Attack stage with fingerprinted cache (agent)

**Files:**
- Create: `src/plate_cloak/attack/stage.py`, `tests/test_attack_stage.py`
- Modify: `src/plate_cloak/cli.py` (`attack --config PATH [--variant V] [overrides...]`)

**Interfaces:**
- Consumes: Tasks 2, 3, 4, 6.
- Produces:
  - `fingerprint(cfg: DictConfig, image_ids: list[str]) -> str`: first 12 hex chars of SHA-256 over sorted-key JSON of `cfg.attack`, `cfg.models.surrogate`, `cfg.data.imgsz`, `cfg.data.min_plate_w`, `cfg.data.min_plate_h`, `cfg.seed`, and `image_ids`.
  - `run_attack(cfg: DictConfig) -> Path`: returns `cache/adv/<fingerprint>/`; if `done.json` exists there, returns immediately; otherwise writes `<image_id>.png` (8-bit RGB, same size as the `EvalImage`), `audit.csv` (one row per image), `settings.yaml`, then `done.json` last.
  - Uses the surrogate's raw network: `load_detector(name, seed).model.eval()`, parameters frozen, input as float RGB tensor.

- [ ] **Step 1: Write the failing tests**

```python
def test_fingerprint_changes_with_attack_settings_and_images():
    base = load_config(BASE)
    f = fingerprint(base, ["a", "b"])
    assert fingerprint(load_config(BASE, ["attack.eps.cb=4"]), ["a", "b"]) != f
    assert fingerprint(base, ["a"]) != f
    assert fingerprint(load_config(BASE, ["models.surrogate.seed=1"]), ["a", "b"]) != f

def test_fingerprint_ignores_eval_settings():
    assert fingerprint(load_config(BASE, ["eval.conf=0.5"]), ["a"]) == fingerprint(load_config(BASE), ["a"])

def test_completed_cache_is_reused(tmp_path, monkeypatch):
    calls = count_pgd_calls(monkeypatch)
    cfg = tiny_cfg(tmp_path)        # 2 fixture images, TinyDetector patched in as surrogate
    run_attack(cfg); run_attack(cfg)
    assert calls.n == 2              # second call did no work

def test_interrupted_cache_is_redone(tmp_path):
    out = run_attack(tiny_cfg(tmp_path)); (out / "done.json").unlink()
    assert (run_attack(tiny_cfg(tmp_path)) / "done.json").exists()
```

- [ ] **Step 2: Run** `pytest tests/test_attack_stage.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests (PASS), then a smoke run on 10 images with the real YOLOv8s seed 0: `python -m plate_cloak attack --config configs/base.yaml data.n_eval=10` and confirm `audit.csv` maxima are within budget.
- [ ] **Step 5: Commit** `git commit -m "Add attack stage with fingerprinted cache"`

---

### Task 8: IoU and per-image matching (OWNER DRAFTS)

**Files:**
- Create: `src/plate_cloak/evaluation/matching.py`, `tests/test_matching.py`

**Interfaces:**
- Produces (owner writes all bodies):
  - `iou(a: np.ndarray (N,4), b: np.ndarray (M,4)) -> np.ndarray (N,M)` for xyxy boxes.
  - `@dataclass ImageMatch: scores: np.ndarray (K,) float; is_tp: np.ndarray (K,) bool; n_gt: int; gt_found: np.ndarray (n_gt,) bool`
  - `match_image(pred_boxes, pred_scores, gt_boxes, gt_ignore, iou_thr=0.5, conf=0.25) -> ImageMatch`:
    process predictions by descending score; each one matches the unmatched non-ignored GT with highest IoU >= `iou_thr` (true positive); otherwise, if it overlaps any ignored GT at >= `iou_thr`, it is dropped (not counted at all); otherwise it is a false positive.
    `n_gt` counts non-ignored GTs; `gt_found[j]` is True if GT j was matched by a prediction with score >= `conf`.
    All predictions are kept regardless of `conf` (AP needs the full ranking); `conf` only affects `gt_found`.

- [ ] **Step 1: Agent writes the failing tests, then hands over**

```python
def test_iou_known_values():
    np.testing.assert_allclose(iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10], [20, 20, 30, 30]])), [[1 / 3, 0]])

def test_each_gt_matched_once_highest_score_first():
    m = match_image(np.array([[0, 0, 10, 10], [1, 0, 11, 10]]), np.array([0.6, 0.9]),
                    np.array([[0, 0, 10, 10]]), np.array([False]))
    assert m.is_tp.tolist() == [True, False] and m.scores.tolist() == [0.9, 0.6]

def test_detection_on_ignored_plate_is_dropped():
    m = match_image(np.array([[0, 0, 10, 10]]), np.array([0.8]), np.array([[0, 0, 10, 10]]), np.array([True]))
    assert m.scores.size == 0 and m.n_gt == 0

def test_low_score_match_counts_for_ap_not_recall():
    m = match_image(np.array([[0, 0, 10, 10]]), np.array([0.1]), np.array([[0, 0, 10, 10]]), np.array([False]))
    assert m.is_tp.tolist() == [True] and m.gt_found.tolist() == [False]

def test_no_predictions_and_no_gts():
    m = match_image(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), np.zeros(0, bool))
    assert m.n_gt == 0 and m.scores.size == 0
```

- [ ] **Step 2: Run** `pytest tests/test_matching.py -v`. Expected: FAIL.
- [ ] **Step 3: OWNER implements.**
- [ ] **Step 4: Run** tests (PASS); agent review: tie-breaking, vectorized IoU, empty-array shapes.
- [ ] **Step 5: Commit** `git commit -m "Add IoU and per-image matching"`

---

### Task 9: AP and recall from matches, validated against torchmetrics (agent)

**Files:**
- Create: `src/plate_cloak/evaluation/detection.py`, `tests/test_detection.py`

**Interfaces:**
- Consumes: Task 8 `ImageMatch`.
- Produces:
  - `average_precision(matches: list[ImageMatch], weights: np.ndarray | None = None) -> float`: single-class AP at the matches' IoU, COCO-style 101-point interpolated precision; `weights[i]` (non-negative integer counts, default all 1) repeats image i, which is how bootstrap resamples are evaluated without re-matching. Returns `nan` if the weighted GT count is 0.
  - `recall(matches, weights=None) -> float`: weighted `gt_found` count over weighted `n_gt`; `nan` if no GT.

- [ ] **Step 1: Write the failing tests**

```python
def test_ap_matches_torchmetrics_on_random_scenes():
    scenes = random_scenes(seed=0, n=50)               # boxes, scores, gts, some ignored
    ours = average_precision([match_image(*s) for s in scenes])
    ref = torchmetrics_map50(scenes)                   # MeanAveragePrecision(iou_thresholds=[0.5]), ignored GTs as iscrowd
    assert abs(ours - ref) < 1e-3

def test_weights_equal_duplicating_images():
    ms = [match_image(*s) for s in random_scenes(seed=1, n=10)]
    w = np.array([2, 0, 1, 1, 3, 0, 1, 1, 1, 0])
    dup = [m for m, k in zip(ms, w) for _ in range(k)]
    assert average_precision(ms, w) == pytest.approx(average_precision(dup))

def test_no_gt_gives_nan_not_error():
    assert math.isnan(average_precision([ImageMatch(np.zeros(0), np.zeros(0, bool), 0, np.zeros(0, bool))]))

def test_model_with_no_detections_scores_zero():
    m = match_image(np.zeros((0, 4)), np.zeros(0), np.array([[0, 0, 10, 10]]), np.array([False]))
    assert average_precision([m]) == 0.0 and recall([m]) == 0.0
```

- [ ] **Step 2: Run** `pytest tests/test_detection.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** by concatenating each image's `scores`/`is_tp` repeated by its weight, sorting by score, cumulative TP/FP, precision envelope, 101 recall points.
- [ ] **Step 4: Run** tests. Expected: PASS. If the torchmetrics comparison differs, resolve before continuing (it is the evidence our mAP is right).
- [ ] **Step 5: Commit** `git commit -m "Add AP and recall from matches, validated against torchmetrics"`

---

### Task 10: Bootstrap confidence intervals (OWNER DRAFTS)

**Files:**
- Create: `src/plate_cloak/evaluation/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Produces (owner writes the body):
  - `@dataclass(frozen=True) CI: point: float; low: float; high: float` with `as_dict() -> dict`.
  - `bootstrap_ci(n_items: int, statistic: Callable[[np.ndarray], float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 42) -> CI`:
    each resample draws `n_items` indices with replacement and passes their **counts** as a weight vector of length `n_items` to `statistic`; `point = statistic(np.ones(n_items))`; percentile interval; resamples where the statistic is `nan` are skipped.
  - Pairing comes from the caller: one `statistic` that computes, for example, `drop_rtdetr(w) - drop_yolov10l(w)` sees the same weights for both models.

- [ ] **Step 1: Agent writes the failing tests, then hands over**

```python
def test_ci_of_mean_covers_truth():
    x = np.random.default_rng(0).normal(5.0, 1.0, 400)
    r = bootstrap_ci(len(x), lambda w: np.average(x, weights=w))
    assert r.low < 5.0 < r.high and r.high - r.low < 0.3

def test_deterministic_for_seed():
    s = lambda w: float(w[:10].sum())
    assert bootstrap_ci(50, s, seed=7) == bootstrap_ci(50, s, seed=7)

def test_paired_difference_of_identical_models_is_exactly_zero():
    x = np.random.default_rng(1).random(100)
    r = bootstrap_ci(100, lambda w: np.average(x, weights=w) - np.average(x, weights=w))
    assert (r.point, r.low, r.high) == (0.0, 0.0, 0.0)

def test_nan_resamples_skipped():
    s = lambda w: float("nan") if w[0] == 0 else 1.0
    assert bootstrap_ci(5, s).point == 1.0
```

- [ ] **Step 2: Run** `pytest tests/test_stats.py -v`. Expected: FAIL.
- [ ] **Step 3: OWNER implements.**
- [ ] **Step 4: Run** tests (PASS); agent review: `numpy.random.default_rng`, `np.bincount` for counts, quantile choice.
- [ ] **Step 5: Commit** `git commit -m "Add bootstrap confidence intervals"`

---

### Task 11: Evaluate stage and run outputs (agent)

**Files:**
- Create: `src/plate_cloak/evaluation/stage.py`, `tests/test_eval_stage.py`
- Modify: `src/plate_cloak/cli.py` (`evaluate`, and `run` = attack then evaluate; both accept `--config`, `--variant`, `--name`, overrides)

**Interfaces:**
- Consumes: Tasks 2, 3, 4, 7, 8, 9, 10.
- Produces:
  - `predict_boxes(model, image: PIL.Image.Image, conf_floor: float = 0.001) -> tuple[np.ndarray (N,4), np.ndarray (N,)]`: Ultralytics `predict(source=image, imgsz=640, conf=conf_floor, verbose=False)`; the low floor keeps the full ranking for AP, while recall uses `cfg.eval.conf`.
  - `evaluate(cfg: DictConfig, adv_dir: Path, run_dir: Path) -> dict`: for every target in `cfg.models.targets`, predicts on each clean `EvalImage.image` and on the cached adversarial PNG of the same size, matches, and writes into `run_dir`:
    - `per_detection.csv`: `image_id, model, seed, condition, score, is_tp`
    - `per_plate.csv`: `image_id, plate_idx, model, seed, condition, found`
    - `metrics.json`: per target and condition `map50`, `recall50` as CI dicts, plus `drop_map50` (clean minus attacked, paired), the audit maxima from `adv_dir/audit.csv`, image count, attack fingerprint, and `provenance()`.
    - `config.yaml` (resolved).
  - All rows use the same image order for every model so the report can pair resamples.

- [ ] **Step 1: Write the failing tests**

```python
def test_pil_and_bgr_array_predictions_agree(yolov8s_coco):
    img = Image.open(FIXTURE_CAR).convert("RGB")
    a = predict_boxes(yolov8s_coco, img)
    b = yolov8s_coco.predict(source=np.asarray(img)[:, :, ::-1].copy(), imgsz=640, conf=0.001, verbose=False)[0]
    np.testing.assert_allclose(a[1][:5], b.boxes.conf.cpu().numpy()[:5], atol=1e-4)

def test_rgb_array_differs_which_is_why_we_use_pil(yolov8s_coco):
    img = Image.open(FIXTURE_CAR).convert("RGB")
    c = yolov8s_coco.predict(source=np.asarray(img), imgsz=640, conf=0.001, verbose=False)[0]
    assert not np.allclose(predict_boxes(yolov8s_coco, img)[1][:5], c.boxes.conf.cpu().numpy()[:5], atol=1e-4)

def test_run_outputs_and_pairing(tmp_path):
    out = run_tiny_pipeline(tmp_path)                  # 3 fixture images, stub models
    per = pd.read_csv(out / "per_detection.csv")
    assert set(per.condition) == {"clean", "attacked"}
    m = json.loads((out / "metrics.json").read_text())
    assert {"map50", "recall50"} <= m["targets"]["yolov8s_s0"]["clean"].keys()
    assert "drop_map50" in m["targets"]["yolov8s_s0"]
```

(`yolov8s_coco` is a session fixture loading the stock `yolov8s.pt`; mark those two tests `@pytest.mark.gpu_optional` and skip in CI if the weights are absent.)

- [ ] **Step 2: Run** `pytest tests/test_eval_stage.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests (PASS), then smoke: `python -m plate_cloak run --config configs/base.yaml data.n_eval=20 --name smoke` and check clean recall of `yolov8s_s0` is close to its fine-tuning val recall. Delete `results/runs/smoke` afterwards (smoke runs are not committed).
- [ ] **Step 5: Commit** `git commit -m "Add evaluate stage, run command, and run outputs"`

---

### Task 12: Cb/Cr sweep config, loss time box, main budget (OWNER DRAFTS the config)

**Files:**
- Create: `configs/sweeps/cbcr.yaml` (owner), `configs/sweeps/loss.yaml` (agent)
- Modify: `docs/experiment-decisions.md`

**Interfaces:**
- Consumes: Task 2 sweep format, Task 11 `run`.
- `cbcr.yaml` (owner): variants `rgb8` (`attack.space: rgb`), `cb4`, `cb8`, `cb12` (`attack.space: ycbcr`, Y 2, Cb = Cr = value), `random8` (`attack.method: random`, `attack.space: rgb`, the control showing PGD rather than noise size causes evasion), and `targets` overridden to `[{name: yolov8s, seed: 0}]` only (the rule only needs the white-box surrogate, which keeps the sweep cheap).
- `loss.yaml` (agent): variants `topk` and `hinge` on `rgb8` and `cb8`.

- [ ] **Step 1: Owner drafts `configs/sweeps/cbcr.yaml`; agent reviews** (one factor per variant, nothing else changes; `variants(...)` lists the five names).
- [ ] **Step 2: Loss time box (2 working days).** Run `loss.yaml` on `data.n_eval=300`; pick the loss with lower attacked YOLOv8s recall; record the numbers and the choice in `docs/experiment-decisions.md`; set it in `configs/base.yaml`.
- [ ] **Step 3: Run the sweep** on all eligible images: `for v in rgb8 cb4 cb8 cb12 random8; do python -m plate_cloak run --config configs/sweeps/cbcr.yaml --variant $v; done`
- [ ] **Step 4: Apply the rule mechanically:** main Cb/Cr = smallest value with recall within 0.05 of `rgb8`'s; if none, add `cb16`, then `cb20`. Record the table and the choice in `docs/experiment-decisions.md`; set `attack.eps.cb/cr` in `configs/base.yaml`.
- [ ] **Step 5: Commit** `git add configs results/runs docs/experiment-decisions.md && git commit -m "Run Cb/Cr sweep and fix main YCbCr budget by rule"`

---

### Task 13: H1 report and the milestone run (agent)

**Files:**
- Create: `src/plate_cloak/analysis/report.py`, `run_ablations.sh`, `tests/test_report.py`
- Modify: `src/plate_cloak/cli.py` (`report --runs DIR... --out DIR`)

**Interfaces:**
- Consumes: Task 9 `average_precision`, Task 10 `bootstrap_ci`, Task 11 run-folder files.
- Produces:
  - `h1_table(run_dir: Path) -> pd.DataFrame`: one row per architecture (`yolov8s` seed 0 white-box; `yolov8s` seeds 1-2; `yolov10l`; `rtdetr-l`): clean mAP, attacked mAP, drop, each as seed-mean with 95% paired bootstrap CI over images, plus min/max across seeds.
  - `h1_verdict(run_dir: Path, threshold: float = 0.35) -> dict`: CI of `mean drop(yolov10l) - mean drop(rtdetr-l)` (seed means inside one statistic so pairing holds); `supported = point >= threshold and low > 0`; also the secondary statistic: attack success rate on plates every target found when clean.
  - `report(runs: list[Path], out: Path) -> None`: writes `h1_table.md`, `h1.json`, `h1_bars.png` (grouped bars clean vs. attacked per architecture, CI error bars).
  - `run_ablations.sh`: runs every variant of every file passed as an argument, then `report`.

- [ ] **Step 1: Write the failing tests** (synthetic `per_detection.csv` / `per_plate.csv`)

```python
def test_verdict_supported_when_rtdetr_resists(tmp_path):
    write_synthetic_run(tmp_path, drops={"yolov10l": 0.6, "rtdetr-l": 0.1})
    v = h1_verdict(tmp_path)
    assert v["supported"] and v["difference"]["point"] == pytest.approx(0.5, abs=0.05)

def test_verdict_not_supported_when_drops_equal(tmp_path):
    write_synthetic_run(tmp_path, drops={"yolov10l": 0.4, "rtdetr-l": 0.4})
    assert not h1_verdict(tmp_path)["supported"]

def test_table_reports_seed_spread(tmp_path):
    write_synthetic_run(tmp_path, drops={"yolov10l": 0.6, "rtdetr-l": 0.1}, seed_jitter=0.05)
    t = h1_table(tmp_path)
    assert {"drop_seed_min", "drop_seed_max"} <= set(t.columns)
```

- [ ] **Step 2: Run** `pytest tests/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests (PASS). Then the **milestone run (long job; confirm with the owner; core 6 offline):** `python -m plate_cloak run --config configs/base.yaml --name m1_h1_main` with the RGB attack as a second run (`attack.space=rgb --name m1_h1_rgb`), then `python -m plate_cloak report --runs results/runs/m1_h1_main results/runs/m1_h1_rgb --out results/reports/m1`.
- [ ] **Step 5: Walk through the H1 table and chart with the owner; update the README status section with the numbers; commit** `git add results docs README.md && git commit -m "Milestone 1: H1 result"`

---

## Later milestones (outline; each gets its own detailed plan when reached)

- **Milestone 2, H2:** fidelity module with LPIPS (primary) and Wang et al. SSIM on Y and RGB, fixed 11x11 Gaussian window, plates under 11 px tall skipped (**OWNER DRAFTS the SSIM function**); Sobel mask variant; H2 decision rule (recall within 0.05 of RGB with CI inside +/-0.05, lower LPIPS); visibility chart and crop grid. **Owner drafts the A2 ablation config.**
- **Milestone 3, JPEG robustness (A5):** real JPEG Q80 post-transform at evaluation; optional differentiable JPEG EOT inside PGD; both variants evaluated after compression.
- **Milestone 4, OCR:** EasyOCR on clean and adversarial plate crops; CER of adversarial read against clean read; readable subset (EasyOCR reads the clean crop) reported for H1 and H2.
- **Milestone 5, surrogate scale (A3):** fine-tune YOLOv8n and YOLOv8l (seed 0); attacks from each surrogate; transfer decay compared.
- **Milestone 6, RT-DETR saliency:** select plates RT-DETR finds and YOLOv8s misses; Grad-CAM on encoder features and plots of deformable sampling points.
