# Milestone 2a (Transfer Attack) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **This milestone has no OWNER DRAFTS tasks:** the agent writes all code (owner's decision); a study document is written at the end (Task 14).
> Tasks marked **OWNER RUNS** are GPU jobs or eye tests: the agent prepares the exact command, the owner runs it in their own terminal, and the agent picks up from the files it writes.

**Goal:** An attack that transfers to the CNN control (YOLOv10l) at a budget the eye cannot see, then the H1 test rerun with it.

**Architecture:** Extend the existing pipeline rather than add a new one.
`pgd()` takes a list of surrogates and gains two switches (momentum, input diversity) whose "off" values reproduce Milestone 1's attack exactly.
A new `plate_cloak.visibility` package holds plate crops, LPIPS and a small blind eye-test server; the budget ceiling it produces is frozen in `experiment-decisions.md` before any transfer run.
Development runs on a frozen validation split; the test set is used once at the end.

**Tech Stack:** Python 3.12, PyTorch 2.11, Ultralytics 8.4.178, OmegaConf, `lpips==0.1.4` (new), Python's `http.server` for the eye test, pytest, ruff.

**Spec:** [docs/superpowers/specs/2026-11-10-transfer-attack-design.md](../specs/2026-11-10-transfer-attack-design.md) (read it first), building on [the experiment design](../specs/2026-10-20-plate-cloak-experiment-design.md).

## Global Constraints

- Deadline 2026-11-17. If time runs out, cut in this order: the RGB final run, then the Y-ladder rungs (Y stays 2), then YOLOv8l (ensemble = YOLOv8s seed 0 + YOLOv8n). The YCbCr final run, the eye tests and the freeze are never cut.
- Before any GPU run the owner takes CPU cores offline and turns boost off (`echo 0 | sudo tee /sys/devices/system/cpu/cpu{6,7,22,23}/online`, `echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`); jobs are pinned with `taskset -c 0-5,8-21,24-31` (`run_ablations.sh` does this). Jobs over about 10 minutes are run by the owner.
- Transfer bar: YOLOv10l 3-seed mean mAP@0.5 drop >= 0.35 with CI low >= 0.35. H1 rule unchanged (difference >= 0.35, CI low > 0).
- Eye test: 2AFC, 40 trials per budget, pass if the attacked crop is picked in at most 25 of 40; ceiling = largest passing rung with every lower rung passing; ladders extend by their step at either end.
- Ladders: Cb/Cr 12, 16, 20, 24 at Y 2; Y 3, 4 at the chosen Cb/Cr; RGB 8, 10, 12. White-box only (YOLOv8s seed 0), `data.n_eval=300`, test split.
- Transfer techniques: ensemble YOLOv8s s0 + YOLOv8n s0 + YOLOv8l s0, each loss divided by its clean-image value; DI `p=0.5`, `min_scale=0.9`; momentum `mu=1.0`. Off values: one surrogate, `p=0`, `mu=0`.
- Fixed search: 4 variants (ens, ens_di, ens_mi, ens_di_mi) + one 100-step rerun of the winner, kept only if YOLOv10l's drop rises by more than the winner's CI half-width. RT-DETR is not a target during development.
- Markdown: one sentence per line, no em dashes. Commits: no AI co-author or sign-off lines (the owner's rule overrides any attribution reminder).
- `ruff check`, `ruff format --check` and the full `pytest` suite pass at every commit.

## Review Focus

1. **A resumed attack with input diversity:** an interrupted cache is finished image by image, so DI's random draws must depend only on (seed, image id), not on run order. Test in Task 4.
2. **Plates at the image edge in the eye test:** the clean and attacked crop of a trial must be the same region and size even when the context box is clipped; otherwise the size difference gives the answer away. Test in Task 8.
3. **Restarting the eye-test server mid-session:** answers already given survive, the page resumes at the first unanswered trial, and a double-clicked answer is stored once. Test in Task 9.
4. **Committed Milestone 1 run folders** (with `models.surrogate`, no `surrogates`) must still load in the report. Test in Task 6.
5. **DI and ensemble shapes:** every surrogate must see a padded input whose sides are multiples of 32, also after DI's resize-and-pad; hinge loss with DI is refused (DI moves the plate boxes). Test in Task 4.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `configs/base.yaml` | defaults | `data.split`, `models.surrogates`, `attack.momentum`, `attack.diversity` |
| `src/plate_cloak/data/eval_images.py` | images of any split | `split` parameter |
| `src/plate_cloak/attack/pgd.py` | the attack | ensemble, momentum, DI |
| `src/plate_cloak/attack/stage.py` | cache | surrogate list, canonical fingerprint |
| `src/plate_cloak/analysis/report.py` | report | roles for a surrogate list, transfer gate |
| `src/plate_cloak/progress.py` | progress bar | fit the line to the terminal |
| `src/plate_cloak/visibility/crops.py` | plate crops | new |
| `src/plate_cloak/visibility/lpips_stage.py` | LPIPS per plate | new |
| `src/plate_cloak/visibility/eyetest.py` | trials, server, scoring, ceiling | new |
| `src/plate_cloak/cli.py` | commands | `visibility`, `eyetest make/serve/score` |
| `configs/sweeps/visibility_*.yaml`, `transfer_dev.yaml`, `transfer_final.yaml` | experiments | new |
| `docs/experiment-decisions.md` | records | M1 result, ceiling, ablation, verdict |
| `docs/study/milestone-2a.md` | owner's study notes | new |

---

### Task 1: Record Milestone 1 (agent)

**Files:**
- Modify: `docs/experiment-decisions.md`
- Commit as-is: `results/runs/m1_h1_main/`, `results/runs/m1_h1_rgb/`, `results/reports/m1/`, `configs/sweeps/eps.yaml`, `tests/test_sweeps.py`, the two spec/plan files

- [ ] **Step 1: Add three Records entries** to `docs/experiment-decisions.md`, numbers copied from `results/reports/m1/h1.json` and `h1_table.md`:
  - "Milestone 1 H1 result (2026-11-10)": the per-row mAP@0.5 drops for both runs, the H1 differences with CIs (main 0.034 [0.028, 0.040], RGB 0.026 [0.020, 0.032]), attack success rates, the verdict "not supported", and the reading from the spec's section 1 (the CNN control barely moved, so H1 was not really tested).
  - "GPU PGD is not bit-for-bit deterministic": 123 of 1,731 RGB images differed between caches `eeaeb03f3a41` and `39d655a3e07b`, by up to 16/255 on about 3% of a plate's pixels; cause cuDNN's non-deterministic backward plus `sign()`; effect on metrics negligible.
  - "Next: transfer attack", one line linking the new spec.
- [ ] **Step 2: Run** `conda run -n plate-cloak pytest -q` and `conda run -n plate-cloak ruff check . && conda run -n plate-cloak ruff format --check .`. Expected: all pass.
- [ ] **Step 3: Commit** `Record the milestone 1 H1 result and plan the transfer attack`.

---

### Task 2: Progress bar fits the terminal (agent)

**Files:** Modify `src/plate_cloak/progress.py`; Test `tests/test_progress.py`

A line wider than the terminal wraps, and `\r` then redraws only the last row, so the bar stacks up.

- [ ] **Step 1: Failing test** `test_terminal_line_is_cut_to_the_terminal_width`: a `Tty` stream, `Progress(1731, "attack", stream=out, clock=clock, columns=lambda: 40)`; after one update every redrawn line (split on `\r`) is at most 39 characters; a log stream (non-TTY) is never cut.
- [ ] **Step 2: Run** `pytest tests/test_progress.py -v`. Expected: FAIL (unexpected keyword `columns`).
- [ ] **Step 3: Implement** a `columns` parameter defaulting to `lambda: shutil.get_terminal_size().columns`; on a TTY, cut the line to `columns() - 1` and clear the rest of the row (`\x1b[K`) after it. Asked on each redraw, so resizing the window works.
- [ ] **Step 4: Run** the progress tests. Expected: PASS.
- [ ] **Step 5: Commit** `Keep the live progress line within the terminal width`.

---

### Task 3: Validation split (agent)

**Files:** Modify `configs/base.yaml`, `src/plate_cloak/data/eval_images.py`, `src/plate_cloak/cli.py` (`prepare_data`); Test `tests/test_eval_images.py`, `tests/test_config.py`; Create `splits/openimages_val_eligible.txt`

**Interfaces:**
- Produces: `data.split: test` (`test` | `val`) in base.yaml, and `data.eligible_list: splits/openimages_${data.split}_eligible.txt`.
  `write_eligible_list(data_root, out_path, imgsz, min_w, min_h, split="test") -> int`.
  `load_by_id(cfg, image_id)` reads `images/<cfg.data.split>/` and `labels/<cfg.data.split>/`.

- [ ] **Step 1: Failing tests:**
  - `test_eligible_list_for_the_val_split`: fixture images under `images/val`, `write_eligible_list(..., split="val")` lists them and ignores `images/test`.
  - `test_load_by_id_reads_the_configured_split`: with `data.split=val`, `load_by_id` loads from `images/val`.
  - `test_eligible_list_follows_the_split`: `load_config(BASE_CONFIG, ["data.split=val"]).data.eligible_list == "splits/openimages_val_eligible.txt"`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** `prepare_data` freezes the list for `cfg.data.split` and prints which split. An unknown split raises `ValueError` listing `test, val`.
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass (existing tests use the test split).
- [ ] **Step 5: Freeze the list:** `conda run -n plate-cloak python -m plate_cloak prepare-data data.split=val`. Expected: "eligible val images: N" (N at most 724; downloads are skipped, nothing is fetched). Record N in the "Dataset as prepared" entry of `experiment-decisions.md`.
- [ ] **Step 6: Commit** `Add a frozen validation split for attack development`.

---

### Task 4: Ensemble, momentum and input diversity in PGD (agent)

**Files:** Modify `src/plate_cloak/attack/pgd.py`, `configs/base.yaml`; Test `tests/test_pgd.py`, `tests/detectors.py`

**Interfaces:**
- Config: `attack.momentum: 0.0`, `attack.diversity: {p: 0.0, min_scale: 0.9}`.
- `AttackSettings` gains `momentum: float`, `diversity_p: float`, `diversity_min_scale: float` (read in `from_cfg`).
- `pgd(nets: Sequence[nn.Module], x: Tensor, boxes: np.ndarray, s: AttackSettings, generator: torch.Generator | None = None) -> Tensor`. `generator` is a CPU generator, required when `s.diversity_p > 0` (ValueError otherwise).
- `diverse(x_padded: Tensor, p: float, min_scale: float, generator: torch.Generator) -> Tensor`: same shape out as in.

Per step:
- Pad once, then (if DI) apply one `diverse` draw shared by every surrogate.
- Loss = mean over surrogates of `loss_i / loss_i(clean)`. Each `loss_i(clean)` is computed once before the loop, detached and clamped to at least 1e-12.
- Momentum: `g = mu * g + grad / grad.abs().mean().clamp_min(1e-12)`, then step `-step * g.sign()`. With `mu = 0` this is exactly the old step.
- `diverse`: with probability `p`, draw `scale ~ U[min_scale, 1]`, resize bilinearly to `(round(H*scale), round(W*scale))`, and pad back to `(H, W)` at a uniformly random top-left offset with `PAD_VALUE`. Otherwise return the input. All draws come from `generator`.

- [ ] **Step 1: Failing tests:**
  - `test_defaults_reproduce_plain_pgd`: with one `TinyDetector`, `momentum=0`, `diversity_p=0`, `pgd([net], ...)` equals a reference loop written in the test (sign of the raw top-k gradient, project, 10 steps) bit for bit.
  - `test_ensemble_lowers_every_members_loss`: `pgd([TinyDetector(0), TinyDetector(1)], ...)` lowers top-k for both nets.
  - `test_ensemble_normalizes_each_member`: add `Scaled(net, factor)` to `tests/detectors.py`, which multiplies the output's score row by `factor`. Since top-k only sums scores, that multiplies the net's loss by `factor` and keeps everything else the same. `pgd([TinyDetector(0), Scaled(TinyDetector(1), 1000)], ...)` must equal `pgd([TinyDetector(0), TinyDetector(1)], ...)`.
  - `test_momentum_changes_the_path`: `momentum=1.0` gives a different image than `0.0`, still lowers the loss, and stays inside the budget and on the plates.
  - `test_diversity_keeps_shapes_and_is_reproducible`: `diverse` returns the input shape for 20 draws; `pgd` with `diversity_p=0.5` and two generators seeded alike gives equal images; a different seed gives a different image; every shape `TinyDetector` saw is a multiple of 32 (Review Focus 1 and 5).
  - `test_diversity_needs_a_generator` and `test_diversity_with_hinge_is_refused`: ValueError for each.
  - `test_settings_from_base_config`: extend it to `s.momentum == 0.0 and s.diversity_p == 0.0`.
  - Update every existing `pgd(net, ...)` call to `pgd([net], ...)`.
- [ ] **Step 2: Run** `pytest tests/test_pgd.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** per the Interfaces block. Fix the docstring: "starts from zero noise; deterministic on CPU, while on GPU cuDNN's non-deterministic backward can flip a few pixels (see experiment-decisions.md)".
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass, `test_pgd_is_deterministic_and_works_under_no_grad` included (CPU).
- [ ] **Step 5: Commit** `Add ensemble surrogates, momentum and input diversity to PGD (off by default)`.

---

### Task 5: Surrogate list and a fingerprint of only what shapes the pixels (agent)

**Files:** Modify `configs/base.yaml`, `src/plate_cloak/attack/stage.py`, `configs/sweeps/*.yaml` (none set the surrogate; verify); Test `tests/test_attack_stage.py`, `tests/test_eval_stage.py`, `tests/test_sweeps.py`

**Interfaces:**
- Config: `models.surrogate` is replaced by `models.surrogates: [{name: yolov8s, seed: 0}]`.
- `load_surrogates(cfg) -> list[nn.Module]` (eval, float32, frozen), replacing `load_surrogate`.
- `weights_digests(cfg) -> dict[str, str]` keyed `"<name>_s<seed>"`, replacing `weights_digest`.
- `settings.yaml` stores `surrogates: [{name, seed, weights_sha256}]`.
- `fingerprint(cfg, image_ids) -> str` hashes this canonical dict, and nothing else:
  - `method`, `space`, `steps`, `step_factor`, `margin`, `mask`, `momentum`.
  - `eps`: `{rgb}` for rgb, `{y, cb, cr}` for ycbcr.
  - `quant_margin`: ycbcr only.
  - `loss`: `{type, k}` for topk, `{type, floor}` for hinge.
  - `diversity`: `{p, min_scale}` if `p > 0`, otherwise absent.
  - The surrogate list, `data.split`, `imgsz`, `min_plate_w`, `min_plate_h`, `seed`, image ids.
- `_attack_one` passes a CPU generator seeded with `image_seed(seed, image_id)` to `pgd` (as the random control already does).
- Milestone 1 caches get new fingerprints and are recomputed if needed; committed M1 runs keep their recorded fingerprint. Their `config.yaml` no longer loads with `load_config`, so reproducing them means checking out commit `50c594b` (the commit recorded in their provenance). Step 3 adds this sentence to the Milestone 1 entry in `experiment-decisions.md`.

- [ ] **Step 1: Failing tests:**
  - `test_fingerprint_ignores_budgets_the_space_does_not_use`: for an rgb config, changing `attack.eps.cb` keeps the fingerprint; for ycbcr, changing `attack.eps.rgb` keeps it; changing `quant_margin` matters only for ycbcr; `loss.floor` matters only for hinge.
  - `test_fingerprint_changes_with_surrogates_split_momentum_and_diversity`: each of a second surrogate, `data.split=val`, `attack.momentum=1.0`, `attack.diversity.p=0.5` changes it; `attack.diversity.min_scale` alone does not while `p=0`.
  - `test_cache_from_other_weights_is_refused`: now with a two-surrogate digest dict where one entry changes.
  - Update the `fake_surrogate` fixtures (`test_attack_stage.py`, `test_eval_stage.py`) to patch `load_surrogates` / `weights_digests`.
  - `test_eps_rgb8_is_the_milestone_1_rgb_attack` keeps passing unchanged.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** Fix the module docstring's determinism claim the same way as Task 4.
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `Attack a list of surrogates and fingerprint only settings that change the pixels`.

---

### Task 6: Report roles for an ensemble, and the transfer gate (agent)

**Files:** Modify `src/plate_cloak/analysis/report.py`; Test `tests/test_report.py`

**Interfaces:**
- `surrogates_of(cfg) -> list[tuple[str, int]]`: `cfg.models.surrogates`, or `[cfg.models.surrogate]` for Milestone 1 run configs.
- `architecture_rows(cfg)`:
  - each surrogate that is also a target is its own row `"<name> s<seed> (white-box)"`, or `"<name> (white-box)"` when there is a single surrogate, which keeps M1's labels;
  - then one `"<name> (retrained)"` row per surrogate architecture with other seeds among the targets;
  - then every other architecture, in target order.
- `transfer_gate(run, model="yolov10l", bar=0.35) -> dict | None`: `{"model", "bar", "drop": CI dict, "passed": drop.low >= bar}`; None if the model is not a target.
- `h1.json` gains `"gate"` per run; `h1_table.md` prints a gate line above the H1 verdict line.

- [ ] **Step 1: Failing tests:**
  - `test_rows_for_an_ensemble`: surrogates yolov8s s0, yolov8n s0, yolov8l s0 and targets those three plus yolov8s s1, s2, yolov10l s0-2 give rows `yolov8s s0 (white-box)`, `yolov8n s0 (white-box)`, `yolov8l s0 (white-box)`, `yolov8s (retrained)`, `yolov10l`.
  - `test_milestone_1_run_config_still_loads`: a synthetic run whose `config.yaml` has `models.surrogate` and no `surrogates` gives M1's row labels (Review Focus 4).
  - `test_gate_passes_only_when_the_ci_clears_the_bar`: synthetic yolov10l drop 0.6 passes; 0.36 with jitter wide enough that low < 0.35 fails; no yolov10l gives None.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass. Regenerate nothing: committed reports stay as written.
- [ ] **Step 5: Commit** `Report white-box rows for each surrogate and the transfer gate on YOLOv10l`.

---

### Task 7: Fine-tune YOLOv8n and YOLOv8l (OWNER RUNS)

**Files:** Modify `docs/experiment-decisions.md`

- [ ] **Step 1: Owner runs** (cores offline first), one after the other:

```bash
taskset -c 0-5,8-21,24-31 conda run --no-capture-output -n plate-cloak python -m plate_cloak finetune --model yolov8n --seed 0
taskset -c 0-5,8-21,24-31 conda run --no-capture-output -n plate-cloak python -m plate_cloak finetune --model yolov8l --seed 0
```

  Expected: `weights/yolov8n_s0.pt` and `weights/yolov8l_s0.pt` with their `.json`. YOLOv8n should take under 40 min and YOLOv8l about 2 h (estimates). Can run in parallel with Tasks 8 and 9, which are CPU work.
- [ ] **Step 2: Agent checks** each `runs/finetune/*/results.csv`: the curve has flattened, and the last val mAP is not below half the best.
- [ ] **Step 3: Add rows** for both to the fine-tuning table in `experiment-decisions.md` (parameters, mAP@0.5, mAP@0.5:0.95, best epoch, training time), then commit `Record YOLOv8n and YOLOv8l fine-tuning`.

---

### Task 8: Plate crops and LPIPS (agent)

**Files:** Create `src/plate_cloak/visibility/__init__.py`, `crops.py`, `lpips_stage.py`; Modify `environment.yml` (add `lpips==0.1.4`), `src/plate_cloak/cli.py`, `.gitignore` (`results/eyetest/*/crops/`); Test `tests/test_visibility.py`

**Interfaces:**
- `CONTEXT_PX = 16`. The crop is the plate box grown by `attack.margin` per side, then by 16 px. *(My call, so the noise's edge sits inside the crop.)*
- `crop_box(box: np.ndarray, margin: float, context: int, w: int, h: int) -> tuple[int, int, int, int]`: integer `(left, top, right, bottom)`, clipped to the image.
- `plate_crops(cfg, adv_dir: Path, image_id: str) -> list[PlateCrop]`, with `PlateCrop(image_id, plate_idx, clean: Image, adv: Image)`. Counted plates only; `plate_idx` is the label row, as in `per_plate.csv`.
- `lpips_scores(pairs: list[PlateCrop], device) -> np.ndarray`: AlexNet LPIPS, inputs scaled to [-1, 1]; NaN for crops with a side under 32 px.
- CLI: `python -m plate_cloak visibility --config SWEEP --variant V --name N` finds the cache by fingerprint (refusing an unfinished one, like evaluate). It writes `results/visibility/<N>/lpips.csv` (`image_id, plate_idx, lpips`) and `summary.json` (`n, n_skipped, mean, median, p90`, budget, fingerprint).

- [ ] **Step 1: Failing tests:**
  - `test_crop_box_grows_and_clips`: a box at the image corner gives a crop clipped at 0 and at w/h; one in the middle grows by margin plus 16.
  - `test_clean_and_attacked_crops_match_in_size` at an edge plate (Review Focus 2).
  - `test_lpips_is_zero_for_identical_crops_and_positive_for_noise`: identical crops < 1e-6; ±8/255 noise > 0; a 20 x 20 crop gives NaN. Marked `@pytest.mark.slow`, skipped if the LPIPS weights can't be loaded offline. Add the `slow` marker to the pytest config and run it locally once.
  - `test_cli_visibility_writes_csv_and_summary` with a monkeypatched `lpips_scores`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Install** `conda run -n plate-cloak pip install lpips==0.1.4`. Expected: installs without changing torch. Then implement.
- [ ] **Step 4: Run** `pytest -q` and the slow test once (`pytest -m slow`). Expected: pass. The first use downloads AlexNet weights.
- [ ] **Step 5: Commit** `Add plate crops and per-plate LPIPS`.

---

### Task 9: Blind eye test (agent)

**Files:** Create `src/plate_cloak/visibility/eyetest.py`; Modify `src/plate_cloak/cli.py`; Test `tests/test_eyetest.py`

**Interfaces:**
- `N_TRIALS = 40`, `MAX_CORRECT = 25`.
- `make_session(budgets: list[Budget], out: Path, seed: int = 42) -> None`. `Budget(ladder: str, rung: str, cfg)`, where `ladder` is the sweep file's stem and `rung` the variant.
  - Per budget, draw 40 counted plates at random from its images. The draw uses `random.Random(f"{seed}:{ladder}:{rung}")`, so each budget gets its own plates and a viewer can't learn a plate across budgets. *(My call.)*
  - Draw the attacked side per trial, then shuffle all trials together.
  - Write `trials.json` (the key: trial id, ladder, rung, image_id, plate_idx, attacked side) and `crops/<trial>_{left,right}.png`.
- `serve(session: Path, port: int = 8765) -> None`: stdlib `http.server` on 127.0.0.1.
  - `GET /` serves the page. `GET /next` returns the first unanswered trial's id and two crop URLs, never the side. `POST /answer {trial, choice}` appends to `answers.csv` (`trial, choice, time`), ignoring a trial that's already answered.
  - The page shows both crops at native pixel size on a mid-gray background, "trial i of n", and buttons plus keys 1/2. It gives no correctness feedback.
- `score(session: Path) -> dict`: per budget `correct, n, passed (n == 40 and correct <= 25)`; per ladder, in rung order, `ceiling` (the largest rung with every lower rung passing, or None) and `extend` (`"up"` if the top rung passed, `"down"` if the bottom failed, else None). Writes `scores.json`.
- CLI: `eyetest make --session NAME --sweeps FILE [FILE ...] [--variants V ...]`, `eyetest serve --session NAME`, `eyetest score --session NAME`. Sessions live in `results/eyetest/<NAME>/`; crops are git-ignored, the rest is committed.

- [ ] **Step 1: Failing tests** (synthetic caches from two tiny configs):
  - `test_session_has_40_trials_per_budget_and_balanced_sides`: the left/right split is within 40% to 60% per budget.
  - `test_next_never_reveals_the_answer`: the `/next` JSON has no `side` key, and the crop URLs carry only the trial id.
  - `test_answers_survive_a_restart_and_duplicates_are_ignored`: answer 3, double-post one, restart the server, and `/next` is trial 4 with 3 rows in `answers.csv` (Review Focus 3).
  - `test_pass_rule`: 25 of 40 passes, 26 fails, and 39 answers counts as not passed and not complete.
  - `test_ceiling_and_extension`: rungs pass, pass, fail, pass give the 2nd rung as ceiling with `extend` None; all pass gives the top rung and `"up"`; a failing first rung gives None and `"down"`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** The server test uses port 0 and a thread.
- [ ] **Step 4: Run** `pytest -q`. Expected: pass. Then open a 3-trial demo session in a browser (VS Code forwards the port) and check the layout: crops not scaled, buttons aligned, nothing that hints at the answer.
- [ ] **Step 5: Commit** `Add a blind two-alternative eye test with the pre-registered pass rule`.

---

### Task 10: Visibility ceiling (OWNER RUNS the sweeps and the eye test)

**Files:** Create `configs/sweeps/visibility_cbcr.yaml`, `visibility_rgb.yaml`, later `visibility_y.yaml`; Modify `docs/experiment-decisions.md`, `configs/base.yaml`; Test `tests/test_sweeps.py`

- [ ] **Step 1: Sweep files.** Shared overrides: `models.targets: [{name: yolov8s, seed: 0}]`, `data.n_eval: 300`, `data.split: test`. Every variant sets its own method, space and budget.
  - `visibility_cbcr`: `cb12, cb16, cb20, cb24` (Y 2, Cb = Cr).
  - `visibility_rgb`: `rgb8, rgb10, rgb12`.
  - Test `test_visibility_ladders_are_white_box_only_and_change_one_budget`, in the style of the existing cbcr tests.
- [ ] **Step 2: Owner runs** `./run_ablations.sh configs/sweeps/visibility_cbcr.yaml configs/sweeps/visibility_rgb.yaml`, then the agent runs `visibility` on each variant.
- [ ] **Step 3: Eye-test session 1:** `eyetest make --session ceiling1 --sweeps configs/sweeps/visibility_cbcr.yaml configs/sweeps/visibility_rgb.yaml` (280 trials). The owner runs `eyetest serve --session ceiling1` and answers every trial, in more than one sitting if they like. Then `eyetest score`.
- [ ] **Step 4: Extend if `score` says so.** Add the next rungs (`cb28`, `cb32`... or `cb8`, `cb4`; `rgb14`... or `rgb6`), run them, and test them in a new session. Repeat until each ladder has a ceiling.
- [ ] **Step 5: Y ladder** at the Cb/Cr ceiling C: `visibility_y` with `y3_cbC`, `y4_cbC`. Run, eye-test in session `ceiling2`, extend if needed. *(Cut 2: skip this step and keep Y 2.)*
- [ ] **Step 6: Freeze.**
  - Add "Visibility ceiling (date)" to `experiment-decisions.md`: per budget, correct/40, pass, LPIPS mean and median, and white-box recall; the ceilings; and a note that no transfer number existed when this was decided.
  - Write the ceilings into `configs/base.yaml` (`attack.eps.y/cb/cr` = YCbCr ceiling, `attack.eps.rgb` = RGB ceiling), with a comment pointing to the entry, and adjust `test_settings_from_base_config` if it hard-codes values.
  - Commit the session folders (minus crops), visibility results, runs and report.
  - Commit `Freeze the visibility ceiling by the eye test`.

---

### Task 11: Transfer ablation on validation (OWNER RUNS)

**Files:** Create `configs/sweeps/transfer_dev.yaml`; Test `tests/test_sweeps.py`; Modify `docs/experiment-decisions.md`

- [ ] **Step 1: Sweep file.**
  - Overrides: `data.split: val`, the YCbCr ceiling from Task 10, `models.surrogates` = yolov8s s0, yolov8n s0, yolov8l s0, and targets = those three plus yolov8s s1, s2 and yolov10l s0-2 (no RT-DETR).
  - Variants: `ens: {}`, `ens_di: {attack.diversity.p: 0.5}`, `ens_mi: {attack.momentum: 1.0}`, `ens_di_mi: {both}`.
  - Test `test_transfer_dev_never_shows_rtdetr_and_uses_validation`: also check that every variant's budget (set in the file, not inherited) equals base.yaml's frozen YCbCr ceiling.
- [ ] **Step 2: Owner runs** `./run_ablations.sh configs/sweeps/transfer_dev.yaml`. Time one image first (`data.n_eval=5`) and give the owner the estimate.
- [ ] **Step 3: Pick** the variant with the largest YOLOv10l mean mAP@0.5 drop, by point estimate.
- [ ] **Step 4: 100-step rerun.** Add variant `<winner>_100` (`attack.steps: 100`) and have the owner run only it (`--variant`). Keep it only if its YOLOv10l drop exceeds the winner's by more than the winner's CI half-width.
- [ ] **Step 5: Record** the table (each variant's drop per row, the gate on validation) and the choice in `experiment-decisions.md`. Commit `Choose the transfer attack on validation`.

---

### Task 12: Eye test of the chosen attack (OWNER RUNS)

- [ ] **Step 1:** `eyetest make --session final_check --sweeps configs/sweeps/transfer_dev.yaml --variants <chosen>` makes 40 trials from validation crops. The owner answers them, then runs `score`.
- [ ] **Step 2:** If it fails, step one rung down the Cb/Cr ladder. Make a validation cache of the chosen variant at that rung (a new variant in `transfer_dev.yaml`, no ablation rerun) and test it again. Repeat until it passes.
- [ ] **Step 3:** Record the result and commit `Eye-test the chosen transfer attack`.

---

### Task 13: Final run on the test set and verdict (OWNER RUNS)

**Files:** Create `configs/sweeps/transfer_final.yaml`; Modify `docs/experiment-decisions.md`, `README.md` (results line)

- [ ] **Step 1: Sweep file.** Test split, all of base.yaml's targets plus yolov8n s0 and yolov8l s0, the chosen technique settings. Variants: `ycbcr` at the checked ceiling and `rgb` at the RGB ceiling *(cut 1: drop `rgb`)*. Test that the techniques equal the chosen dev variant's.
- [ ] **Step 2: Owner runs** `./run_ablations.sh configs/sweeps/transfer_final.yaml`, once.
- [ ] **Step 3: Read the gate and the verdict** from `results/reports/transfer_final/h1.json` and apply spec section 5, step 6, as written. Record it in `experiment-decisions.md` with the full table, and commit `Run the transfer attack on the test set and record the H1 verdict`.

---

### Task 14: Study document (agent)

**Files:** Create `docs/study/milestone-2a.md`

- [ ] **Step 1:** Write it for the owner, in the walkthrough style used for Milestone 1. Cover:
  - why single-surrogate PGD overfits, with Milestone 1's numbers;
  - what the ensemble loss, its normalization, momentum and input diversity each do, pointing to the lines in `pgd.py`;
  - what LPIPS measures;
  - why the eye test is 2AFC and where 25/40 comes from;
  - why development stayed on validation;
  - how to read the final report.
  End each section with one or two questions to check understanding.
- [ ] **Step 2:** Commit `Add study notes for milestone 2a`.
