# Design: Cross-Architecture Transfer of Perceptual Plate Cloaking

Status: draft for review, 2026-10-20.
Owner: JSnack58 (CMPE 249 research project), working with Claude Code.
Companion documents: the proposal (section 1 below summarizes it), [experiment-decisions.md](../../experiment-decisions.md), and the two spikes in `spikes/`.

## 1. Intent

Automated license plate recognition relies on real-time object detectors.
Perceptual cloaking adds small, bounded noise to a plate so a person sees no change but a detector stops finding it.
Prior work shows this works against CNN detectors (YOLOv8).
Nobody has checked whether that noise transfers to Transformer-based real-time detectors (RT-DETR), whose global self-attention might ignore local noise.

**Research question:** do YCbCr-constrained perturbations generated on a CNN (YOLOv8s) transfer zero-shot to a Transformer detector (RT-DETR-L), or does self-attention act as an implicit defense?

**Hypotheses (from the proposal, made precise in this design):**
- **H1, architectural resistance:** noise built on YOLOv8s loses much more of its effect on RT-DETR-L than on an unseen, size-matched CNN (YOLOv10l).
  Decision rule: each model's mAP@0.5 drop (clean minus attacked) on the same images; H1 is supported if RT-DETR-L's drop is smaller than YOLOv10l's by at least 0.35, with a 95% paired bootstrap CI on that difference excluding zero.
  Secondary: attack success rate on the plates that every model detects when clean (controls for differing clean accuracy).
- **H2, perceptual equivalence:** putting most of the noise budget in chroma (Cb/Cr) hides plates from YOLOv8s about as well as an RGB L_inf attack, while being less visible to people.
  Decision rule: YCbCr's attacked YOLOv8s recall within 0.05 of RGB's (95% CI of the difference inside +/-0.05), with lower LPIPS (primary visibility metric; luminance SSIM secondary).
  The target is relative to RGB, not the proposal's absolute recall of 0.05.

Both outcomes of H1 are publishable results.
The deliverable is a trustworthy answer, presented to a US audience, plus a reproducible public repository.

**Why it is built the way it is:** the owner wants to learn how to structure, write and run research code.
The owner drafts several library pieces; Claude reviews, pushes back, restructures, writes the plumbing, and walks through everything it writes.
Design choices are discussed before code is written.

## 2. Goals

1. **H1 result slide:** clean vs. attacked plate mAP@0.5 (with 95% bootstrap CIs) for YOLOv8s (white-box surrogate), YOLOv10l (CNN transfer control) and RT-DETR-L (Transformer transfer target).
   The CNN control is required: transfer attacks weaken even between CNNs, so H1 is a claim about RT-DETR relative to the control, not relative to YOLOv8s.
2. **H2 result slide:** hiding power (YOLOv8s recall) vs. visibility (LPIPS) for each attack setting, including a sweep of Cb/Cr budgets (e.g. 4, 8, 12 /255) so the chart shows a curve, plus plate crops the audience can judge by eye.
3. **Extras, in priority order:** JPEG robustness (A5), OCR disruption (CER), surrogate scale (A3), Grad-CAM-style analysis of RT-DETR on plates it finds but YOLOv8s misses.
4. **Reproducibility:** pinned environment, config-driven runs, every run's resolved config and raw metrics committed, one script to rerun the ablations, automated plots.

## 3. Non-goals

- No physical-world attacks (printed stickers, real cameras).
- No new detector architectures or training methods; detectors are standard Ultralytics models.
- No attack on the OCR stage directly; OCR error is measured as a side effect.
- No deployment, UI or real-time cloaking tool.

## 4. Anti-goals (would make the project a failure even if it runs)

- A comparison where architecture is not the only difference between models (different data, splits or training budgets).
- Claiming a budget ("Y <= 2/255") that the saved images do not actually satisfy.
- Reporting attack success without the clean accuracy of the same model next to it.
- Results that cannot be traced to an exact config and code version.
- Code the owner cannot explain.

## 5. Constraints

- **Schedule:** about 4 weeks from 2026-10-20, ending around 2026-11-17.
- **Hardware:** one RTX 3090 (24 GB); a second RTX 3080 Ti is present but unused by default.
  CPU core 6 (logical CPUs 6 and 22) is unstable and has hard-reset the machine several times.
  Before any GPU run, take it offline (`echo 0 | sudo tee /sys/devices/system/cpu/cpu6/online /sys/devices/system/cpu/cpu22/online`, needs the owner's password, lasts until reboot), and pin jobs with `taskset -c 0-5,7-21,23-31`.
  Confirm with the owner before launching long jobs.
- **Public repository:** dataset images in committed figures need attribution; datasets and weights are never committed.
- **Owner preferences:** semantic line breaks in Markdown, no em dashes, no AI co-author lines in commits, quality and long-term maintainability over development speed.

## 6. Decided design

### 6.1 Data

- **Dataset: Open Images V7, class "Vehicle registration plate"** (8,154 images; mostly UK/EU, roughly 15-20% US; varied real-world scenes; annotations CC BY 4.0, images CC BY 2.0).
  Fallback if it fails in practice: keremberke/license-plate-object-detection (8,823 images, mostly Vietnamese parking garages).
  `spikes/02_openimages/download_openimages_plates.py` already builds the YOLO-format dataset; the main code should absorb or reuse it.
- **Splits:** Open Images' own train (5,368) / validation (724) / test (2,062) splits, frozen to a committed split file. *(Claude's call: reuses a public, citable split instead of re-splitting.)*
- **Plate eligibility for evaluation:** an image is eligible if its largest plate is at least 40 px wide and 11 px tall at the 640 input (Spike 02's 40x10 rule kept 1,729 of 2,062 test images).
  Training uses all plates regardless of size.
- **Evaluation set:** all eligible test images, with a config option to subsample for fast development runs.
  The set is frozen with the size rule only, so it does not depend on any trained model. *(Claude's call.)*
- **Multiple plates per image:** the perturbation covers every annotated plate in the image (an untouched second plate would otherwise be detected and blur the result).
  Scoring is per plate, over eligible-size plates only; smaller plates are "ignore" regions, so a detection matching one counts as neither a hit nor a false positive. *(Claude's call.)*
- **Readable subset:** results are also reported on plates EasyOCR can read on the clean image, which drops redacted and blurred plates without depending on our detectors. *(Claude's call.)*

### 6.2 Models

- Surrogate: YOLOv8s. Transfer targets: YOLOv10l (CNN control) and RT-DETR-L (Transformer). A3 adds YOLOv8n and YOLOv8l as alternative surrogates.
  YOLOv10l replaces the proposal's YOLOv10n so the CNN control is close to RT-DETR-L in size, and, like RT-DETR, it is NMS-free.
  Check exact parameter counts when the models are loaded and report them.
- All fine-tuned by us from COCO-pretrained weights on the same split, image size (640), epoch budget and augmentation.
  Optimizer settings follow each family's Ultralytics defaults, since RT-DETR does not train well with YOLO's defaults; the exact settings are recorded. *(Claude's call.)*
- **Same fixed epoch count for every model, no early stopping**, keeping the best-validation checkpoint; the count is set high enough for RT-DETR-L to converge, confirmed from its validation curve (starting point about 100). *(Claude's call.)*
- **Seeds:** 3 training seeds for each of the 3 main models; 1 seed for A3's extra surrogates.
  Results report the spread across seeds in addition to the image bootstrap; per-image results are kept per seed.
- **Compute:** estimated 20-28 GPU-hours of fine-tuning in total (worst case 40+).
  Milestone 1 starts by timing 2 epochs of each model and adjusting the epoch count or seed plan if the estimate is far off.
  Training must resume from its last checkpoint after a crash. Either GPU may be used.
- Clean mAP@0.5 on validation is checked and recorded before any attack.

### 6.3 Attack

- PGD with a fixed budget and a fixed step count (40), not a minimal-perturbation search.
  Step size per channel = 2.5 x budget / steps.
- Perturbation restricted to the ground-truth plate box grown by 10% on each side.
- Loss on the surrogate's raw pre-NMS outputs (NMS is not differentiable).
  Starting point: sum of the top-100 plate confidences (Spike 01); candidate improvement: a hinge on candidates above about 0.1 that overlap a plate.
  **Loss tuning is time-boxed to 2 working days**, then the best version is used for all runs, whatever recall it reaches.
- Settings compared:
  - RGB L_inf 8/255 (control)
  - YCbCr with Y <= 2/255 and a larger Cb/Cr budget (numbers open, see 7)
  - YCbCr + Sobel edge mask (A2)
  - random sign noise at the RGB budget (control that PGD, not noise size, causes evasion)
  - optional differentiable-JPEG EOT (A5)
- **YCbCr budget enforcement:** after each PGD step, scale each pixel's YCbCr noise vector along its own direction until the RGB result is in [0, 1] (gamut fit), and project to the budget minus 0.5/255 so it still holds after 8-bit rounding.
  This fixes the leak found in Spike 01 (measured Y change 5.9/255 against a 2/255 budget).
- All evaluation uses the adversarial image rounded to 8-bit.
- Every attack run audits the actual max change in Y, Cb, Cr (or R, G, B) on the saved image and records it.

### 6.4 Evaluation

- Detection: confidence threshold 0.25 for all models; a ground-truth plate counts as found if a detection overlaps it with IoU >= 0.5.
  Metrics: mAP@0.5 and Recall@0.5, each with a 95% percentile bootstrap CI over images (1,000 resamples, fixed seed).
  The bootstrap is paired: each resample uses the same images for every model, so differences between models get their own CI.
  YOLOv8 uses NMS while YOLOv10 and RT-DETR are NMS-free; only the confidence threshold is shared, and the write-up says so.
  The mAP implementation must be validated against Ultralytics' own validation numbers on clean images.
- Clean metrics for every model are reported next to every attacked metric.
- Fidelity (per plate, on the perturbed region): LPIPS is the primary visibility metric; luminance SSIM and RGB SSIM (Wang et al. settings, see 7) are secondary.
- OCR: EasyOCR on clean and adversarial plate crops; neither dataset has plate text, so CER compares the adversarial read to the clean read.
- Per-image results are logged so CIs and failure analysis can be computed later.

### 6.5 Architecture

**Pipeline stages,** each its own command, communicating through files:

1. `prepare-data`: download, convert, filter, freeze the split.
2. `finetune --model <name>`: writes `weights/<name>.pt` and its training metrics.
3. `attack`: surrogate + attack settings, writes adversarial images to `cache/adv/<fingerprint>/`.
4. `evaluate`: every target model on clean and adversarial images, writes a run folder.
5. `report`: reads many run folders, writes the H1 table, H2 chart and figures.

`run` chains `attack` and `evaluate` for everyday use.
Each stage can be checked on its own (clean mAP after 2, budget audit after 3).

**Config:** YAML files read with OmegaConf; a base config plus one file per ablation, overriding one factor at a time; command-line overrides in dotted form (`attack.eps.cb=6`).
YAML for files people write (comments matter); JSON for files programs write (metrics).

**Run folders** (`results/runs/<name>/`, committed):
- `config.yaml`: the fully resolved config, including overrides; can be fed back in to reproduce the run.
- `metrics.json`: aggregate metrics with CIs, plus git commit, library versions and GPU.
- `per_image.csv`: one row per image x model x condition.

Names are generated automatically (timestamp plus settings that differ from the base), or set with `--name`.
A run refuses to start if its folder already exists.

**Caching:** adversarial images are keyed by a hash of every setting that affects them (surrogate, budgets, mask, steps, loss, EOT, seed, image list), so experiments sharing an attack (e.g. both A4 variants) reuse it and stale caches cannot be used by accident.
Fine-tuned weights are reused across runs.

**Code layout:**

```
configs/            base.yaml, ablations/*.yaml
src/plate_cloak/
  data/             loading, filtering, splits
  models/           loading and fine-tuning detectors
  attack/           colorspace, masks, pgd, losses, eot
  evaluation/       detection metrics, fidelity, OCR, bootstrap
  analysis/         saliency, plots
  cli.py            the stage commands (thin)
tests/              unit tests for library functions
spikes/             the two feasibility spikes (kept as-is)
docs/               decisions log, this design
data/ weights/ cache/    git-ignored
results/runs/       committed run folders
```

Library functions take tensors and plain arguments, not paths or configs, so they can be unit-tested.
Entry points read the config, call library functions and write outputs.
Notebooks only read results and make figures.

**Environment:** an isolated conda environment from a pinned `environment.yml` (the spike venv inherited broken packages from base Anaconda and must not be the model).
GitHub Actions runs ruff and pytest on CPU.

### 6.6 Division of labor

- The owner drafts, Claude reviews: YCbCr gamut fit, IoU matching / Recall@0.5, bootstrap CIs, the fidelity (SSIM) function, at least one ablation config.
- Claude writes and walks through: config loading, run folders and logging, caching, CLI, environment, CI, data preparation, fine-tuning wrappers, reporting.
- Each piece: discuss, write, review or walk through, then verify with a test or audit before building on it.

### 6.7 Build order

Build the H1 path end to end first, then add pieces in this order, so a late failure only costs an extra:

1. Core slice (H1): data, fine-tune the 3 main models, attack (RGB and fixed YCbCr), clean vs. attacked mAP with CIs.
2. H2: fidelity metrics, Sobel variant, visibility chart.
3. JPEG robustness (A5).
4. OCR error.
5. Surrogate scale (A3).
6. Saliency analysis of RT-DETR. RT-DETR's decoder uses deformable attention, so classic attention rollout does not apply directly; expect Grad-CAM on encoder features or a plot of deformable sampling points.

## 7. Open questions (settle before the milestone that needs them)

Tracked in detail in [experiment-decisions.md](../../experiment-decisions.md).

- **Optional real ALPR system** as an extra black-box target (presentation value only).

Settled after review (2026-10-20), recorded here because they were open:

- **Main YCbCr budget: chosen by a rule fixed in advance.**
  Y stays at 2/255; Cb/Cr is swept over 4, 8, 12 /255.
  The main budget is the smallest Cb/Cr value whose attacked YOLOv8s recall is within 0.05 of the RGB attack's; H1 and the extras use it.
  If none qualifies, extend the sweep (16, then 20) and say so in the write-up.
  The rule is applied mechanically to avoid choosing a budget after seeing which one suits the hypothesis.
  Note Cb/Cr 8/255 moves RGB by up to about 14/255, so RGB 8/255 and YCbCr (2, 8, 8) are not equal-strength budgets.
- **SSIM settings: Wang et al. (2004).**
  Gaussian window, sigma 1.5, 11x11, one fixed window that is never shrunk; computed on luminance (Y) and on RGB, reported side by side.
  Plates under 11 px tall are skipped for SSIM; the evaluation height minimum is therefore 11 px (eligibility becomes 40 px wide and 11 px tall), so "readable" and "measurable" share one rule.

## 8. Left to the builder

- Internal module APIs, function names and data structures, within the layout above.
- How the CLI is implemented (argparse or similar), as long as it supports the stage commands and dotted overrides.
- Exact cache layout and fingerprint format.
- Plot styling.
- Whether and how to reuse the untracked early skeleton in the working tree (`configs/`, `src/`, `pyproject.toml`, `environment.yml`, `.github/`); it predates this design and was never reviewed.
