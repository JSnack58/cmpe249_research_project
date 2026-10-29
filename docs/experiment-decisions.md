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
- **YCbCr budget numbers.**
  Turn "80% of the budget to chroma" into concrete Cb/Cr limits.
  Note that Cb/Cr 8/255 moves RGB by up to about 14/255 (blue), so RGB 8/255 and YCbCr (2, 8, 8) are not equal-strength budgets.
- **Budget enforcement in YCbCr.**
  Proposed: per-pixel gamut fit (shrink each pixel's noise vector until it fits in RGB) plus a 0.5/255 quantization margin, so the budget holds exactly on the saved image.
- **Attack strength vs. the H2 target.**
  Spike 01 reached recall of about 0.25, not the 0.05 target; small plates survive most.
  Options: hinge loss on candidates above a threshold, more steps, or restating the target.
- **Loss function.**
  Current: sum of the top 100 raw plate confidences.
  Candidate: hinge that only penalizes candidates above about 0.1, and only those overlapping the plate.
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

- The workstation's CPU core 6 (logical CPUs 6 and 22) is unstable.
  Take it offline before GPU runs: `echo 0 | sudo tee /sys/devices/system/cpu/cpu6/online /sys/devices/system/cpu/cpu22/online`.
- Use an isolated environment for the real experiment; the spike's venv inherits broken packages from base Anaconda.
