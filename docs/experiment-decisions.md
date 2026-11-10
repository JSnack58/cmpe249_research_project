# Experiment Decisions

Running list of design decisions for the main experiment.
As of 2026-10-20, most "Open" items below are settled in the [design doc](superpowers/specs/2026-10-20-plate-cloak-experiment-design.md), which takes precedence; its section 7 lists what is still open.
"Decided" items are settled; "Open" items must be settled before the real runs.
Each entry notes where it came from.

## Decided

- **Fixed noise budget, not minimal perturbation.**
  PGD runs a fixed number of steps under a fixed L_inf budget, and we measure how often detectors fail at that budget.
  This keeps the comparison across architectures fair.
- **Noise only inside the plate box (plus a margin).**
  Placement uses the ground-truth boxes, so a detector mistake cannot contaminate the attack.
- **Same training data, split, and recipe for every detector, except RT-DETR's learning rate.**
  YOLOv8s, YOLOv10l and RT-DETR-L are all fine-tuned by us with the same data, image size, batch, augmentation and epochs.
  The optimizer is pinned per family instead of Ultralytics' "auto": AdamW, momentum 0.9, no bias warm-up, lr 0.002 for YOLO and 1e-4 for RT-DETR (the RT-DETR paper's value).
  The write-up should say each detector is trained with its family's standard learning rate (see the Records entry on the RT-DETR divergence).
  Off-the-shelf plate detectors cannot carry the H1 evidence. (Spike 01 used one, and we cannot rule out test-image leakage.)
- **Random-noise control at the same budget.**
  Spike 01 showed random +/-8/255 noise leaves recall at 0.99, so evasion comes from the optimization, not the noise size.
- **Evaluate on the 8-bit image.**
  All metrics and budget audits use the adversarial image rounded to 8-bit, as it would be saved and shared.
- **Measure constraints on the final output.**
  Every run audits the actual max change per channel against the stated budget (Spike 01 caught a leak this way).

## Records

- **Dataset as prepared (2026-10-28, Task 3).**
  Open Images plates: 5,368 train, 724 validation, 2,065 test images (the 3 test images missing in Spike 02 downloaded this time).
  Eligible test images (a plate at least 40 px wide and 11 px tall at the 640 input): 1,731, frozen in `splits/openimages_test_eligible.txt`.
  Eligible validation images by the same rule (2026-11-10, for attack development in milestone 2): 604 of 724, frozen in `splits/openimages_val_eligible.txt`; no image is in both lists.
- **No image rotation handling needed for Open Images.**
  None of the 8,154 downloaded images carries an EXIF orientation tag, and Open Images' rotation metadata is 0 (or empty) for every val/test image we use.
  The loader still applies `exif_transpose`, which is a no-op here but protects other image sources.
- **Fine-tuning timing (2026-10-28, Task 4, RTX 3090, 2 epochs each, batch 16, 640 px).**
  YOLOv8s 24 s/epoch (100 epochs about 0.7 h); YOLOv10l 81 s/epoch (about 2.3 h); RT-DETR-L 104 s/epoch (about 2.9 h).
  One seed of all three: about 5.9 h; three seeds: about 17.7 h; with A3's extra surrogates roughly 20 GPU-hours, inside the planned 20-28.
  Validation mAP@0.5 after only 2 epochs: YOLOv8s 0.71, YOLOv10l 0.69, RT-DETR-L 0.82, so 100 epochs is ample.
- **RT-DETR training is not bit-for-bit reproducible.**
  Its deformable attention uses `grid_sampler_2d_backward`, which PyTorch has no deterministic CUDA kernel for (Ultralytics warns and continues).
  Same-seed RT-DETR runs can differ slightly; YOLO runs do not have this problem.
  The 3-seed design reports this variation; mention it as a limitation in the write-up.
- **RT-DETR diverged with the shared learning rate (2026-10-28).**
  The first rtdetr-l seed 0 run used what "auto" picked for every model: AdamW at lr 0.002, 20x the RT-DETR paper's 1e-4.
  Validation loss was already NaN in epochs 2, 4 and 6; at epoch 10 the training loss went NaN and val mAP@0.5 fell from 0.851 (epoch 8) to 0 for good.
  The checkpoint stayed finite, but batch-norm statistics grew 10-40x and the model found no boxes; Ultralytics stopped after 3 failed NaN recoveries.
  "auto" is also fragile: past 10,000 iterations it switches to MuSGD at lr 0.01, so a change in epochs or dataset size would silently change the recipe.
  Fix: pinned optimizer settings (above); YOLO's are identical to what "auto" chose, so the finished yolov8s seed 0 run stays valid.
  Fine-tuning now refuses to resume or export a run whose last val mAP is below half its best.
  The diverged run is kept in `runs/finetune/rtdetr-l_s0_diverged_lr002/`.
- **Fine-tuning results (2026-10-30, Task 4, 9 runs, 100 epochs, best-validation checkpoint kept).**
  Validation mAP@0.5, mean ± standard deviation over seeds 0-2 (best epochs in brackets):

  | Model | Parameters | mAP@0.5 | mAP@0.5:0.95 | Best epochs |
  |---|---|---|---|---|
  | YOLOv8s (surrogate) | 11.1 M | 0.897 ± 0.002 | 0.604 ± 0.008 | 58, 80, 47 |
  | YOLOv10l (CNN control) | 25.8 M | 0.893 ± 0.003 | 0.602 ± 0.012 | 55, 84, 86 |
  | RT-DETR-L (Transformer) | 32.8 M | 0.917 ± 0.001 | 0.646 ± 0.008 | 20, 17, 11 |

  Seeds agree closely for every model, so RT-DETR's nondeterministic kernel (above) has no visible effect.
  Convergence check: every curve has flattened. YOLO models plateau around epoch 50-80.
  RT-DETR peaks within 20 epochs and then drifts down slowly (10-epoch mean about 0.912 to 0.895), mild overfitting that the best-validation checkpoint avoids, so no rerun with more epochs is needed.
  RT-DETR is the most accurate model on clean images, which works against H1 (a more accurate detector is usually harder, not easier, to fool), so the clean-accuracy confound does not favor our hypothesis.
  The surrogate and the CNN control are within 0.004 of each other.
  Weights live only in `weights/` (git-ignored).

- **AP is validated against pycocotools directly, not torchmetrics (2026-11-06, Task 9).**
  Our single-class AP@0.5 matches the official pycocotools evaluator to 1e-9 on 5 random test sets of 50 images each, with ignored plates passed as crowd regions.
  torchmetrics 1.9 wraps pycocotools, but its compute() returned 0.1866 where pycocotools gives 0.1896 on the same inputs.
  Running pycocotools on torchmetrics' own exported files gives 0.1896, so the gap is inside torchmetrics, not in our data or our AP.
  We keep all detections per image; COCO keeps the top 100.
  That cap never binds for YOLO (about 16-21 boxes per image above the 0.001 floor), but RT-DETR returns all 300 of its queries, so its AP includes boxes COCO would drop; they rank below its real detections and barely move AP.
- **Loss: top-k, chosen in the loss time box (2026-11-09, Task 12).**
  YOLOv8s seed 0 attacked on 300 eligible test images (seed-42 subsample), clean recall 0.961 and mAP@0.5 0.971; runs `results/runs/loss_*`, report `results/reports/loss/`.

  | Attack | Loss | Attacked recall | Attacked mAP@0.5 | mAP@0.5 drop |
  |---|---|---|---|---|
  | RGB 8 | top-k | 0.077 [0.048, 0.111] | 0.107 | 0.864 [0.818, 0.901] |
  | RGB 8 | hinge | 0.080 [0.049, 0.114] | 0.534 | 0.437 [0.377, 0.476] |
  | YCbCr (2, 8, 8) | top-k | 0.135 [0.099, 0.176] | 0.178 | 0.793 [0.743, 0.838] |
  | YCbCr (2, 8, 8) | hinge | 0.119 [0.084, 0.158] | 0.525 | 0.446 [0.382, 0.493] |

  The planned rule (lower attacked recall wins) gives a tie: top-k wins on RGB by 0.003, hinge on YCbCr by 0.016, with nearly identical CIs.
  We broke the tie with mAP@0.5, the H1 metric, where top-k's drop is about twice hinge's on both budgets with CIs far apart (paired: same images and resamples).
  Why they differ: hinge (floor 0.1) stops pushing once a candidate is under 0.1, so plates fall below the 0.25 threshold but stay ranked above most false alarms, and AP barely moves.
  Top-k (sum of the 100 highest candidate scores) keeps pushing for all 40 steps, so plate scores sink to background level.
  Hinge also hides plates only from one threshold: lowering the threshold to about 0.05 at prediction time would recover them, which top-k does not allow.
  `attack.loss.type: topk` stays in base.yaml (it was already the default), so the provisional H1 runs (`prov_*`) used the chosen loss.
- **Main YCbCr budget: Cb/Cr 12, by the pre-registered rule (2026-11-09, Task 12).**
  Rule, fixed in the design doc before any sweep: the main Cb/Cr is the smallest value whose attacked YOLOv8s recall is within 0.05 of the RGB 8 attack's.
  Sweep on all 1,731 eligible test images, YOLOv8s seed 0 only, top-k loss, Y fixed at 2; clean recall 0.958; runs `results/runs/cbcr_*`, report `results/reports/cbcr/`.

  | Variant | Attacked recall | Gap to rgb8 | Within 0.05 | mAP@0.5 drop | Largest RGB change |
  |---|---|---|---|---|---|
  | rgb8 | 0.081 [0.069, 0.094] | | | 0.857 [0.836, 0.870] | 8 |
  | cb4 | 0.376 [0.355, 0.399] | 0.295 | no | 0.496 [0.473, 0.517] | 8 (blue) |
  | cb8 | 0.138 [0.123, 0.154] | 0.057 | no | 0.776 [0.754, 0.793] | 15 (blue) |
  | cb12 | 0.054 [0.045, 0.065] | 0.027 | yes | 0.896 [0.879, 0.909] | 22 (blue) |
  | random8 | 0.938 [0.928, 0.950] | | | 0.010 [0.005, 0.014] | 8 |

  cb8 misses by 0.007; the rule uses point estimates and is applied as written, so the main budget is YCbCr (2, 12, 12), now in base.yaml.
  The random control at the RGB 8 budget barely moves detection, so evasion comes from PGD's direction, not the noise size.
  YCbCr (2, 12, 12) moves blue by up to 22/255, almost three times the RGB attack's limit; whether it is still less visible is H2's question.
  The provisional H1 run `prov_h1_ycbcr_cb8` used the old placeholder (2, 8, 8) and is superseded by the milestone run.
- **PGD padding follows predict(), not Spike 01 (2026-11-06, Task 6).**
  Ultralytics' predict() pads a 640-wide image to a multiple of 32 centered (a 427-row image gets 10 rows on top, 11 at the bottom) with gray 114.
  Spike 01 padded at the bottom and right only, so its surrogate saw a slightly different layout during the attack than at evaluation.
  PGD now pads exactly like predict(), and a test checks it against Ultralytics' own LetterBox.
- **Milestone 1 H1 result: not supported, and not really tested (2026-11-10, Task 13).**
  All 1,731 eligible test images, 3 seeds per model, PGD on YOLOv8s seed 0 (top-k, 40 steps); runs `results/runs/m1_h1_main` (YCbCr 2, 12, 12) and `m1_h1_rgb` (RGB 8), report `results/reports/m1/`.

  | Target | Clean mAP@0.5 | Drop, YCbCr (2, 12, 12) | Drop, RGB 8 |
  |---|---|---|---|
  | YOLOv8s seed 0 (white-box) | 0.968 | 0.896 [0.879, 0.909] | 0.857 [0.836, 0.870] |
  | YOLOv8s seeds 1-2 (retrained) | 0.966 | 0.065 [0.056, 0.075] | 0.068 [0.058, 0.078] |
  | YOLOv10l (CNN control) | 0.966 | 0.035 [0.029, 0.041] | 0.034 [0.029, 0.041] |
  | RT-DETR-L (Transformer) | 0.983 | 0.001 [-0.001, 0.003] | 0.009 [0.006, 0.012] |

  H1 difference, drop(YOLOv10l) minus drop(RT-DETR-L): 0.034 [0.028, 0.040] for YCbCr and 0.026 [0.020, 0.032] for RGB, far below the 0.35 threshold.
  Attack success on the 1,669 plates every target found when clean (YCbCr): white-box 94%, retrained 9%, YOLOv10l 4%, RT-DETR 0.2%.
  Reading: the attack overfits its one surrogate and barely transfers even to the same architecture retrained.
  With the CNN control losing only 0.035, a 0.35 gap was impossible by construction, so this run did not really test H1.
  RT-DETR is affected slightly less than YOLOv10l (a real but small gap), and RGB behaves like YCbCr, so the color space is not the reason.
  Next: a transfer attack, designed in [the transfer attack spec](superpowers/specs/2026-11-10-transfer-attack-design.md).
- **GPU PGD is not bit-for-bit deterministic (2026-11-10).**
  Two caches of the same RGB 8 attack (`eeaeb03f3a41` and `39d655a3e07b`; the fingerprint changed only because base.yaml's Cb/Cr, which an RGB attack never uses, went from 8 to 12) differ in 123 of 1,731 images, by up to 16/255 on about 3% of a plate's pixels.
  Cause: cuDNN's backward pass is not deterministic, and PGD steps along `sign(gradient)`, so a tiny difference flips a pixel's step.
  The effect on metrics is negligible; the code's docstrings no longer claim the attack is deterministic on GPU.

## Open

- **Dataset: Open Images V7 plates or keremberke.**
  Open Images: 8,154 images, mostly UK/EU with roughly 15-20% US, varied scenes, small plates.
  keremberke: 8,823 images, mostly Vietnamese parking garages, large plates.
  Spike 02 ran; Open Images is preferred if it holds up.
  Spike 01's detector reached only 0.70 clean recall on Open Images, so every model must be fine-tuned on whichever dataset we pick.
- **Minimum plate size.**
  Spike 02 uses at least 40 px wide and 10 px tall at the 640 input.
  The SSIM window (below) may push the height minimum to 11 px.
- **SSIM definition.**
  Proposed: Wang et al. (2004) settings (Gaussian window, sigma 1.5, 11x11) with one fixed window, excluding plates smaller than the window rather than shrinking it.
  Decide RGB-SSIM, luminance-only SSIM, or both; Spike 01 showed they disagree about chroma noise, which can flip H2.
- **Budget enforcement in YCbCr.**
  Proposed: per-pixel gamut fit (shrink each pixel's noise vector until it fits in RGB) plus a 0.5/255 quantization margin, so the budget holds exactly on the saved image.
- **Attack strength vs. the H2 target.**
  Spike 01 reached recall of about 0.25, not the 0.05 target; small plates survive most.
  Options: hinge loss on candidates above a threshold, more steps, or restating the target.
- **OCR metric without text labels.**
  Neither candidate dataset has plate text.
  Proposed: CER of EasyOCR's read on the adversarial image against its read on the clean image.
- **Test set size.**
  The proposal says 5,000 images; the expected H1 effect (delta mAP >= 0.35) is likely detectable with a few hundred, with bootstrap CIs.
- **Ablation scope for a ~4 week schedule.**
  A3 (surrogate scale) needs two extra fine-tunes; rank ablations by how much they matter to H1 and H2.
- **Clean accuracy confound for H1.**
  Spike 02 showed a low-confidence detector is far easier to fool (spike 01's detector on Open Images: clean recall 0.70, attacked recall 0.05; even random noise removed 4%).
  If RT-DETR is less accurate on clean images than YOLOv10n, it will look easier to fool for reasons unrelated to self-attention.
  At minimum, report clean mAP next to every attack number; ideally compare models at similar clean accuracy.
- **Redacted and unreadable plates.**
  Open Images labels plates that uploaders blacked out for privacy, and some plates pass the size filter but are too blurry to read.
  Decide whether to filter these (e.g. drop plates the clean detector or OCR cannot read) and report how many were dropped.
- **Thresholds across architectures.**
  YOLOv8 uses NMS; YOLOv10 and RT-DETR are NMS-free.
  Only the confidence threshold (0.25) is shared, and the write-up should say so.
- **Optional: a real ALPR system as an extra black-box target.**
  Good for the presentation, not evidence for H1.

## Infrastructure

- The workstation's CPU has uncorrectable machine-check errors under long training loads, first on core 6 and then, with core 6 offline, on core 7 (both on the hotter chiplet).
  The BIOS update from F32 to F40d on 2026-10-29 made crashes much rarer but did not stop them (core 7 again on 2026-10-30).
  Before GPU runs, take cores 6 and 7 offline and turn off boost: `echo 0 | sudo tee /sys/devices/system/cpu/cpu{6,7,22,23}/online` and `echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`.
  A crash while Ultralytics is saving can truncate `last.pt`; fine-tuning then stops with a message to move the run aside and retrain.
- Use an isolated environment for the real experiment; the spike's venv inherits broken packages from base Anaconda.
