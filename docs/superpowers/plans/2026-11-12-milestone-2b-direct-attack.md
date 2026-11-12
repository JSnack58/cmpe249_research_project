# Milestone 2b: Direct Attack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attack RT-DETR-L and YOLOv10l directly with their own gradients at the eye-tested budget, prove the attack works on both, and give the pre-registered H1-direct verdict; score H2 by its written rule.

**Architecture:** PGD stops assuming YOLOv8's head: each surrogate gets a *view* that prepares the input the way that family's `predict()` does and reads one plate score per candidate from that family's eval output. New analysis code combines several runs (one per attacked model and seed) with a shared image bootstrap for the strength check and the verdict, and scores H2 on milestone 1's caches.

**Tech Stack:** Python 3.12, PyTorch, Ultralytics 8.4.178, OmegaConf, pandas, pytest, ruff (existing `plate-cloak` conda env).

**Spec:** `docs/superpowers/specs/2026-11-12-direct-attack-design.md`

## Global Constraints

- Deadline 2026-11-17; the strength check must pass for both models by 2026-11-15 or 2b reports no verdict.
- Models: RT-DETR-L and YOLOv10l, seeds 0, 1, 2, fine-tuned weights in `weights/`. Attack: plain PGD, top-k loss k 100, 40 steps, step factor 2.5, margin 0.10, momentum 0, diversity p 0.
- Budget: YCbCr Y 3, Cb/Cr 4 (written into every sweep file, not inherited); step-down for both models together: Y 2, Cb/Cr 4.
- Strength check: RGB 16, plain PGD, 40 steps, validation (604 images); every seed of both models needs a white-box mAP@0.5 drop of at least 0.80 (point estimate).
- H1-direct: supported if YOLOv10l's seed-mean drop minus RT-DETR-L's is at least 0.35 and the 95% paired bootstrap CI of the difference excludes zero (low > 0); each seed's images attacked with that seed's weights.
- H2: YCbCr (`m1_h1_main`, cache `360ea0dd4bfc`) vs RGB (`m1_h1_rgb`, cache `39d655a3e07b`): attacked YOLOv8s s0 recall difference CI inside plus or minus 0.05, and lower mean LPIPS.
- `predict()` match tolerance (RT-DETR): preprocessed input within 1/255 per pixel; every detection above confidence 0.25 matched with box IoU at least 0.99 and score within 0.01.
- YOLOv8 attacks must stay bit-identical to milestone 2a's on CPU (existing caches stay valid).
- Test set used once (Task 6). Owner runs GPU jobs over about 10 minutes unless they ask Claude; commands are split over short lines with `\`.
- Markdown: one sentence per line, no em dashes. Commits: no AI co-author or sign-off lines. `ruff check`, `ruff format --check` and `pytest` green at every commit.

## Review Focus

1. **Reading the wrong tensor from a tuple output:** RT-DETR's and YOLOv10's eval output is `(top-300 table, raw)`; the existing `_raw` takes element 0 and reads row 4, which is silently wrong. Expected: each view reads its family's raw scores; a test feeds each view a fake output whose wrong reading gives different numbers (Task 1).
2. **Choosing the view by zoo family:** YOLOv10l is family `yolo` in the zoo but has an end-to-end head. Expected: `view_for` picks by head (end-to-end or not) and by family; test with a fake end-to-end head (Task 1).
3. **Combining runs over different image lists:** the verdict bootstraps across six run folders. Expected: refuse unless every run has the same image ids in the same order (Task 3).
4. **A strength or verdict run whose white-box model is not its first target, or has two surrogates:** expected: the white-box model is the run's single surrogate, looked up among its targets; anything else is refused (Task 3).
5. **Pointing the visibility stage at a cache that does not match the config:** expected: refuse when the cache's data settings or image count differ (Task 4).

---

### Task 1: Per-family views in PGD

**Files:**
- Create: `src/plate_cloak/attack/views.py`
- Modify: `src/plate_cloak/attack/pgd.py` (replace `predict_padding`/`_raw`/`_loss` use inside `pgd` with views), `src/plate_cloak/attack/losses.py` (`topk_loss`, `hinge_loss` take scores), `src/plate_cloak/attack/stage.py` (`load_surrogates` returns views too)
- Test: `tests/test_views.py`, `tests/test_pgd.py`, `tests/test_losses.py` (if present, else the tests that call the losses), `tests/detectors.py`

**Interfaces:**
- Produces:
  - `class View(Protocol)`: `prepare(x: Tensor) -> Tensor` (eval-size image (1,3,H,W) in [0,1] to the network input), `plates_to_input(boxes: np.ndarray, h: int, w: int) -> np.ndarray` (plate boxes into the input frame), `scores(output) -> Tensor` ((A,) plate confidence per candidate, differentiable), `supports_hinge: bool`.
  - `DetectView()` (YOLOv8: pad with `predict_padding`, value `PAD_VALUE`; scores `output[0][0, 4]` or `output[0, 4]`; `supports_hinge = True`).
  - `EndToEndView()` (YOLOv10: same padding; scores `output[1]["one2one"]["scores"][0, 0].sigmoid()`, before the head's own top-k; `supports_hinge = False`).
  - `RTDETRView(imgsz: int)` (stretch to `imgsz` x `imgsz` with `F.interpolate(..., mode="bilinear", align_corners=False)`; scores: the decoder's last layer, batch 0, class 0, sigmoid, from `output[1][1]`; `supports_hinge = False`).
  - `view_for(name: str, net: nn.Module, imgsz: int) -> View`: `RTDETRView` for zoo family `rtdetr`; for `yolo`, `EndToEndView` if the net's last module has `end2end` true, else `DetectView`.
  - `topk_loss(scores: Tensor, k: int) -> Tensor`; `hinge_loss(scores: Tensor, floor: float, plate_mask_anchors: Tensor | None) -> Tensor`; `anchors_on_plates(raw, boxes)` unchanged (Detect only).
  - `pgd(nets, x, boxes, s, generator=None, views: Sequence[View] | None = None)`; `views=None` means `DetectView()` for every net (all existing callers unchanged).
  - `load_surrogates(cfg) -> list[nn.Module]` unchanged; new `surrogate_views(cfg, nets) -> list[View]`; `run_attack` passes them to `pgd`.

- [ ] **Step 1: Pin today's YOLOv8 output.** Before touching `pgd`, add `test_pgd_detect_path_is_unchanged` to `tests/test_pgd.py`: run `pgd` on CPU with the existing tiny Detect-style test detector, a fixed image and boxes, YCbCr settings and an ensemble-with-DI settings case; assert `hashlib.sha256(out.numpy().tobytes()).hexdigest()` equals constants you record now from the current code. Run it: PASS (it pins current behavior).
- [ ] **Step 2: Write the view tests** in `tests/test_views.py`, with fakes in `tests/detectors.py`:
  - `FakeEndToEnd`: an `nn.Module` whose last module has `end2end = True`; eval output `(table, {"one2one": {"scores": logits (1,1,A)}, "one2many": {...}})`, where `table` is a (1, 300, 6) tensor whose `[0, 4]` row differs from the true scores.
  - `FakeRTDETR`: eval output `(table, (dec_bboxes, dec_scores, enc_bboxes, enc_scores, None))` with `dec_scores` shaped (1, 1, Q, 1) logits, and `table[0, 4]` different from them.
  - `test_end_to_end_view_reads_one2one_scores`: `EndToEndView().scores(out)` equals `logits[0, 0].sigmoid()` and not `table[0, 4]`.
  - `test_rtdetr_view_reads_last_decoder_layer`: equals `dec_scores[-1, 0, :, 0].sigmoid()`.
  - `test_rtdetr_view_stretches_to_a_square`: `RTDETRView(640).prepare(torch.rand(1,3,480,640)).shape == (1,3,640,640)`; `plates_to_input([[0,0,640,480]], 480, 640)` gives `[[0,0,640,640]]`.
  - `test_detect_view_pads_like_predict`: `DetectView().prepare` on a 480 x 630 image gives sides that are multiples of 32, matching `predict_padding`.
  - `test_view_for_picks_by_head_not_zoo_family`: `view_for("yolov10l", FakeEndToEnd(), 640)` is an `EndToEndView`; `view_for("yolov8s", TinyDetector(), 640)` a `DetectView`; `view_for("rtdetr-l", FakeRTDETR(), 640)` an `RTDETRView`.
  - `test_pgd_lowers_the_score_each_view_reads`: for each fake (built so its scores depend on the input pixels), `pgd(..., views=[view])` lowers `view.scores` summed top-k below the clean value.
  - `test_hinge_needs_a_view_that_supports_it`: `pgd` with hinge settings and an `EndToEndView` raises `ValueError` matching "hinge".
- [ ] **Step 3: Run** `pytest tests/test_views.py -q`. Expected: FAIL (no module `plate_cloak.attack.views`).
- [ ] **Step 4: Implement** `views.py`, change the losses to take `scores`, and make `pgd` prepare `clean`/`seen` and read scores through each net's view (the clean-loss scale and the ensemble averaging stay as they are; input diversity is applied to the prepared input). Keep `predict_padding` and `PAD_VALUE` in `pgd.py` and import them in `views.py`.
- [ ] **Step 5: Run** `pytest -q` and `ruff check . && ruff format --check .`. Expected: all pass, including `test_pgd_detect_path_is_unchanged` with the hashes from Step 1.
- [ ] **Step 6: Commit** `Read each detector family's own scores and input layout in PGD`.

### Task 2: The attack matches `predict()` on the real models

**Files:**
- Test: `tests/test_views_real.py` (marked `slow`; skipped when a weights file is missing)

**Interfaces:**
- Consumes: `view_for`, `load_detector(name, seed)`, `load_by_id(cfg, image_id)`.

- [ ] **Step 1: Write** `test_view_matches_predict[rtdetr-l]` and `[yolov10l]` (and `[yolov8s]` as a control), each on 3 validation images with seed 0 weights, on CPU:
  - Build the network input with `view.prepare` from the eval-size image; compare with the tensor Ultralytics' predictor builds for the same image (call `model.predictor.preprocess` after `model.predictor.setup_model`, or capture it with a forward pre-hook on `model.model` during `model.predict(...)`): max absolute difference at most 1/255.
  - Run `model.model(view.prepare(x))` and take the detections above confidence 0.25 from its table; compare with `model.predict(image, conf=0.25)`: same count, each matched with IoU at least 0.99 and score within 0.01.
  - `view.scores(output)` top values equal (within 1e-4) the table's scores above 0.25, which shows the scores the loss reads are the ones evaluation keeps.
- [ ] **Step 2: Run** `pytest -m slow tests/test_views_real.py -q`. If a check fails, fix the view (Task 1's file) until it passes, adding a fast test in `tests/test_views.py` for each bug found. Expected at the end: PASS.
- [ ] **Step 3: Time one image:** `run` with `data.n_eval=2`, `data.split=val`, surrogate and target `rtdetr-l` s0, `output.runs_root` and `output.cache_root` in the scratchpad; record seconds per image for RT-DETR-L and YOLOv10l on each GPU in the ledger.
- [ ] **Step 4: Commit** `Check that the attack sees what predict() sees for every detector family`.

### Task 3: Multi-run analysis: strength check and H1-direct

**Files:**
- Create: `src/plate_cloak/analysis/direct.py`
- Modify: `src/plate_cloak/cli.py` (new `direct` command)
- Test: `tests/test_direct.py`

**Interfaces:**
- Consumes: `load_run(run_dir) -> Run`, `Run.drop(targets, metric, w)`, `bootstrap_ci(n, statistic, n_resamples, ci, seed)`, `surrogates_of(cfg)`.
- Produces:
  - `white_box(run: Run) -> tuple[str, int]`: the run's single surrogate, which must also be one of its targets; `ValueError` otherwise.
  - `strength_check(runs: list[Path], bar: float = 0.80) -> list[dict]`: per run `{"run", "model", "seed", "drop": CI dict, "passed": drop point >= bar}`.
  - `h1_direct(runs: list[Path], threshold: float = 0.35, cnn: str = "yolov10l", transformer: str = "rtdetr-l") -> dict`: groups runs by `white_box`; needs the same seeds for both models; `ValueError` if image ids or their order differ between runs, or a (model, seed) appears twice. One shared bootstrap over images: statistic(w) = mean over seeds of each CNN run's own white-box drop minus the same for the Transformer. Returns `{"seeds", "cnn_drop", "transformer_drop", "difference" (CI dicts), "threshold", "supported", "attack_success"}` with `supported = point >= threshold and low > 0`, where `attack_success` is `report._attack_success(run)` for each run (secondary reading: plates every model finds when clean).
  - `transfer_matrix(runs: list[Path]) -> pd.DataFrame`: rows = attacked model and seed, columns = every target's mAP50 drop (by-product, not the verdict).
  - CLI: `python -m plate_cloak direct strength --runs DIR... --out DIR` writes `strength.json` and `strength.md`; `python -m plate_cloak direct verdict --runs DIR... --out DIR` writes `h1_direct.json`, `h1_direct.md` (the verdict line, both drops, the difference, the transfer matrix).

- [ ] **Step 1: Write tests** in `tests/test_direct.py` with small run folders built the way `tests/test_report.py` builds them (reuse its helpers):
  - `test_white_box_is_the_single_surrogate`, and `..._refuses_two_surrogates` and `..._refuses_a_surrogate_that_is_not_a_target`.
  - `test_strength_check_passes_at_the_bar_point_estimate`: a run with drop exactly 0.80 passes, 0.79 fails.
  - `test_h1_direct_supported_when_cnn_drops_much_more`: CNN runs drop about 0.9, Transformer runs about 0.1, then `supported` is true and `difference["point"]` is about 0.8; the mirror case is not supported.
  - `test_h1_direct_refuses_runs_over_different_images` and `..._refuses_a_repeated_model_seed` and `..._refuses_unequal_seeds`.
  - `test_h1_direct_bootstrap_is_shared_across_runs`: with identical CNN and Transformer runs the difference CI is exactly `[0, 0]`.
  - `test_cli_direct_writes_both_reports`.
- [ ] **Step 2: Run** `pytest tests/test_direct.py -q`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** `direct.py` and the CLI command (bootstrap settings from the first run's `cfg.eval.bootstrap` and `cfg.seed`).
- [ ] **Step 4: Run** `pytest -q`, ruff. Expected: all pass.
- [ ] **Step 5: Commit** `Add the strength check and the H1-direct verdict over several runs`.

### Task 4: H2 by its written rule

**Files:**
- Create: `src/plate_cloak/analysis/h2.py`
- Modify: `src/plate_cloak/visibility/lpips_stage.py` (score an explicit cache folder), `src/plate_cloak/cli.py` (`visibility --cache DIR`; new `h2` command)
- Test: `tests/test_h2.py`, `tests/test_visibility.py`

**Interfaces:**
- Consumes: `load_run`, `Run.score`, `bootstrap_ci`, `run_visibility`, `results/eyetest/*/scores.json`.
- Produces:
  - `config_for_cache(adv_dir: Path) -> DictConfig`: base.yaml with `data.split=test`, the cache's `data` values and its `attack` block from `settings.yaml`; `ValueError` if the cache's `n_images` differs from the config's image count.
  - `run_visibility(cfg, name, out_root=None, adv_dir=None)`: with `adv_dir`, uses that folder instead of the fingerprint lookup.
  - `h2_verdict(ycbcr_run: Path, rgb_run: Path, ycbcr_lpips: Path, rgb_lpips: Path, tolerance: float = 0.05) -> dict`: attacked YOLOv8s s0 recall50 for each run and their difference (YCbCr minus RGB) with one shared image bootstrap (refuse different image lists); mean LPIPS of each and the paired difference over plates present in both CSVs with a bootstrap CI; `recall_equivalent = -tolerance < low and high < tolerance`; `less_visible = lpips difference point < 0`; `supported = both`.
  - `h2_chart(visibility_dirs: list[Path], eyetest_scores: Path, out: Path)`: attacked recall (from each rung's run) against LPIPS mean, one point per rung, ladders by color, eye-test pass and fail marked by marker shape; saved as PNG.
  - CLI: `python -m plate_cloak h2 --ycbcr-run DIR --rgb-run DIR --ycbcr-lpips CSV --rgb-lpips CSV --out DIR` writes `h2.json`, `h2.md` and the chart.
- [ ] **Step 1: Write tests:**
  - `test_visibility_scores_an_explicit_cache` and `test_config_for_cache_refuses_a_different_image_count` (in `tests/test_visibility.py`, reusing its fake cache).
  - `test_h2_supported_when_recall_matches_and_lpips_is_lower`, `test_h2_not_supported_when_lpips_is_higher`, `test_h2_not_supported_when_the_recall_ci_leaves_the_band`, `test_h2_refuses_runs_over_different_images`.
  - `test_h2_chart_marks_eye_test_results` (the PNG exists; the plotted points' count and markers match the inputs, read back from the function's returned data).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`, ruff. Expected: all pass.
- [ ] **Step 5: Score H2 for real** (no attack runs; LPIPS on GPU takes minutes):
  - `visibility --cache cache/adv/360ea0dd4bfc --name h2_ycbcr` and `--cache cache/adv/39d655a3e07b --name h2_rgb` (device `cuda:1`).
  - `h2 --ycbcr-run results/runs/m1_h1_main --rgb-run results/runs/m1_h1_rgb --ycbcr-lpips results/visibility/h2_ycbcr/lpips.csv --rgb-lpips results/visibility/h2_rgb/lpips.csv --out results/reports/h2`.
  - Record the verdict in `experiment-decisions.md` exactly as the rule reads, with the eye-test evidence beside it labeled as evidence the rule did not name.
- [ ] **Step 6: Commit** `Score H2 by its written rule`.

### Task 5: Gate 1 and Gate 2 on validation (OWNER RUNS)

**Files:**
- Create: `configs/sweeps/direct_strength.yaml`, `configs/sweeps/direct_eye.yaml`
- Test: `tests/test_sweeps.py`
- Modify: `docs/experiment-decisions.md`

- [ ] **Step 1: Sweep files and tests.**
  - `direct_strength.yaml`: overrides `data.split: val`, `attack.method: pgd`, `attack.space: rgb`, `attack.eps.rgb: 16`, `attack.momentum: 0.0`, `attack.diversity.p: 0.0`; variants `rtdetr_s0`, `rtdetr_s1`, `rtdetr_s2`, `v10l_s0`, `v10l_s1`, `v10l_s2`, each setting `models.surrogates` and `models.targets` to that one model and seed.
  - `direct_eye.yaml`: same shape, `attack.space: ycbcr`, `attack.eps.y: 3`, `attack.eps.cb: 4`, `attack.eps.cr: 4`, variants `rtdetr_s0`, `v10l_s0`.
  - `test_direct_strength_attacks_each_model_with_its_own_weights_at_rgb16` and `test_direct_eye_is_the_frozen_plain_pgd_ceiling_on_validation` (budget written in the file; no RT-DETR model ever attacked with another seed's weights).
- [ ] **Step 2: Owner runs the strength check,** one model per GPU, each loop on one terminal (commands split with `\`; Claude gives the time estimate from Task 2 Step 3).
- [ ] **Step 3: Read it:** `direct strength --runs results/runs/direct_strength_* --out results/reports/direct_strength`. If any seed fails, its attack is broken: debug with superpowers:systematic-debugging, fix in the views with a fast test, delete that variant's cache and run folder (they were made by broken code; note it in the ledger), rerun. Deadline for both models passing: 2026-11-15; otherwise record "the direct attack could not be made to work on <model>" and go to Task 7 without a verdict.
- [ ] **Step 4: Eye test.** Owner runs both `direct_eye` variants; `eyetest make --session direct1 --sweeps configs/sweeps/direct_eye.yaml` (80 trials); owner answers; `eyetest score --session direct1`. Also run `visibility` on both variants (LPIPS of each model's noise, a secondary reading). If either fails, add variants `rtdetr_s0_y2` and `v10l_s0_y2` (`attack.eps.y: 2`), run both, session `direct2`; if either fails there too, stop and ask the owner.
- [ ] **Step 5: Record** both gates (per-seed strength drops, eye-test counts, the budget the final run uses) in `experiment-decisions.md`; commit `Pass the strength check and eye-test the direct attacks`.

### Task 6: Final run on the test set and the verdict (OWNER RUNS)

**Files:**
- Create: `configs/sweeps/direct_final.yaml`
- Test: `tests/test_sweeps.py`
- Modify: `docs/experiment-decisions.md`, `README.md` (results line)

- [ ] **Step 1: Sweep file:** test split, every image; the budget Task 5 fixed, written in the file; variants `rtdetr_s0..s2`, `v10l_s0..s2`, each with `models.surrogates` = that model and seed and `models.targets` = base.yaml's nine targets. Test `test_direct_final_matches_the_eye_tested_budget_and_every_target`.
- [ ] **Step 2: Owner runs** the six variants, three per GPU, once.
- [ ] **Step 3: Verdict:** `direct verdict --runs results/runs/direct_final_* --out results/reports/direct_final`; apply the spec's rule as written. Record the table, the verdict and the transfer matrix in `experiment-decisions.md`; update the README results line; commit `Run the direct attacks on the test set and record the H1-direct verdict`.
- [ ] **Cut if late** (spec section 6): first skip the transfer matrix, then run seed 0 only and report the verdict on seed 0 with a note.

### Task 7: Study document

**Files:**
- Create: `docs/study/milestone-2b.md`

- [ ] **Step 1:** Write the study notes in the style of `docs/study/milestone-2a.md`: what changed in the attack and why (views, the silent-wrong-tensor trap), the two gates, the H1-direct result, the H2 result with the LPIPS-versus-eye disagreement, check-yourself questions with answers.
- [ ] **Step 2:** Commit `Add study notes for milestone 2b`.
