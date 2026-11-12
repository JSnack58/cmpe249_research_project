# Milestone 2b: direct (white-box) attack on RT-DETR vs. YOLOv10l

Status: draft for the owner's approval.
Builds on `2026-10-20-plate-cloak-experiment-design.md` (H1, H2) and `2026-11-10-transfer-attack-design.md` (milestone 2a).
Deadline: 2026-11-17, unchanged; the presentation is built from this milestone's results.

## 1. Why

Milestone 2a ran H1 as a transfer test: noise built on YOLOv8 models, shown to the CNN control (YOLOv10l) and the Transformer (RT-DETR-L).
At a budget the eye test cannot see, the noise barely transferred even to the CNN control (YOLOv10l mAP50 drop 0.069 against a gate of 0.35), so H1 stayed untested.
Transfer fell off even between seeds of the same architecture (0.21 white-box, 0.09 retrained), so transfer cannot tell us anything about Transformers at this budget.

Milestone 2b takes transfer out of the question.
Each model is attacked directly, with gradients from its own weights, under the same invisible budget, and we compare how much each one breaks.
The question becomes: with full access to its weights, is a Transformer still harder to fool than a size-matched CNN at a budget people cannot see?

Both outcomes are results:
RT-DETR dropping clearly less than YOLOv10l is evidence that attention-based detectors resist this noise;
dropping about the same means 2a's small RT-DETR number (0.006) came from transfer, not from the architecture.

## 2. Goals, non-goals, anti-goals

**Goals**
- A pre-registered verdict on **H1-direct** (section 5) from one run on the test set.
- Proof that the attack on RT-DETR actually works (the strength check, section 4), before any verdict is read.
- H2 scored by its written rule on runs that already exist (section 7).
- Everything in the presentation traceable to committed configs, runs and reports, as in 2a.

**Non-goals**
- Transfer techniques (ensemble, input diversity, momentum): there is nothing to transfer to.
- An RGB comparison of the direct attack (cut to fit the deadline).
- New budgets: the budget comes from 2a's frozen ceiling, never from 2b's results.
- Attacking YOLOv8s directly again (2a already has those numbers).

**Anti-goals** (would make this a failure even if it technically runs)
- **A weak attack mistaken for a robust model.** RT-DETR's query selection is a top-k step that gradients do not flow through, and its input is resized differently from YOLO's. A broken or partial gradient would make RT-DETR look robust. Section 4 exists to rule this out.
- **An unfair comparison:** different budgets, steps, losses or images for the two models.
- **A budget or rule chosen after seeing 2b's numbers.**
- **Touching the test set more than once,** or before the eye tests and the strength check pass.

## 3. The comparison

- **Models:** RT-DETR-L (Transformer) and YOLOv10l (CNN control; size-matched and, like RT-DETR, NMS-free), seeds 0, 1, 2 each, the fine-tuned weights already in `weights/`.
- **Attack:** PGD as in milestone 1 (top-k loss, k 100; 40 steps; step factor 2.5; margin 0.10; noise only on the plate region), one model at a time, each seed attacked with its own weights.
  Transfer techniques off (momentum 0, diversity p 0).
- **Each attacked image set is evaluated on the model it was made for** (the white-box drop), and, as a by-product, on the other five direct targets and YOLOv8s seeds 0-2, so cross-architecture transfer in both directions is reported too (not part of the verdict).
- **Data:** the test split, all 1,731 eligible images, once, for the verdict; the validation split (604 images) for the eye tests and the strength check.
- **Scoring:** exactly as in 2a: mAP@0.5 drop (clean minus attacked) per model, seed means, 95% paired bootstrap CIs over images, same thresholds (confidence 0.25, IoU 0.5).

**The attack must see each model's input the way evaluation does.**
Evaluation runs each model through Ultralytics `predict()` on the saved PNG.
In the installed Ultralytics (8.4.178), YOLO pads the image to a multiple of 32 (what the attack already does), while RT-DETR's predictor stretches it to a 640 by 640 square (`scale_fill = True`).
The attack's differentiable preprocessing must match each family's predictor, checked by a test that compares the attack's forward pass with `predict()` on the same image.
Ultralytics resizes RT-DETR's input with `cv2.resize` on 8-bit pixels, which a differentiable float resize cannot reproduce bit for bit, so the test's tolerance is: preprocessed input within 1 level of 255 per pixel, every detection above the confidence threshold matched with box IoU at least 0.99 and score within 0.01.
*(Claude's call: the attack does not copy cv2's rounding; the strength check in section 4, not this test, is what proves the gradient works.)*

**Each family needs a score adapter.**
The current loss expects YOLOv8's Detect-head layout (1, 5, anchors).
RT-DETR in eval mode returns 300 query boxes with scores, already top-k selected; YOLOv10l's one-to-one head returns its own top-k layout.
The builder adds, per family, a differentiable mapping from the network's eval-mode output to one plate-confidence score per candidate, so the same top-k loss applies to all three families.
*(Builder's call: whether to read scores before or after each head's own top-k, as long as the strength check passes and the choice is the same idea for both models; recorded in `experiment-decisions.md`.)*

## 4. Gate 1: the attack strength check (before any verdict)

A weak RT-DETR attack is the easiest way for this experiment to fool us, so its strength is checked first, on validation.

- **Check:** at a deliberately large, visible budget, RGB 16, plain PGD, 40 steps, each model's white-box mAP@0.5 drop must reach at least 0.80 as a point estimate, for every seed of RT-DETR-L and YOLOv10l (each seed attacked with its own weights, 604 validation images).
  *(Claude's call: 0.80 because YOLOv8s white-box at the much smaller RGB 12 already drops 0.94, so a working gradient attack at RGB 16 should all but erase detection; a point estimate, because this is a check that the attack works, not a hypothesis test, and the bar sits far below where a working attack lands.)*
- **The check applies to both models the same way.** If either model fails, its attack is broken, not the model robust: fix the gradient path (adapter, preprocessing, query selection) and rerun the check for that model.
  YOLOv10l's one-to-one head has its own top-k, so it can fail for the same reason RT-DETR can.
  The check, not intuition, decides when it is fixed.
  If no fix passes the check for both models by **2026-11-15**, 2b reports "the direct attack could not be made to work on <model>" with the evidence, and no H1-direct verdict is given.
- The check's budget is visible on purpose; its images are never shown in the eye test or the presentation as "invisible".

## 5. Gate 2: eye tests, then the verdict rule (fixed now, before any 2b number exists)

**Budget.** Both models are attacked at the frozen plain-PGD ceiling, **YCbCr Y 3, Cb/Cr 4** (eye-tested as plain PGD on YOLOv8s, `ceiling4`, 24 of 40).

**Eye tests.** Each model's noise gets its own 40-trial eye test on validation crops (seed 0 of each model), with 2a's rule: pass at 25 or fewer correct of 40, sides balanced, one viewer (the owner).
2a showed that a different noise pattern can fail at the same budget, so neither model's noise is assumed invisible.
If either fails, **both** step down to **Y 2, Cb/Cr 4** and both are re-tested (the budget must stay equal).
If either fails there too, the owner decides how to proceed; there is no eye-tested rung below.

**H1-direct (the verdict).**
With each model attacked directly at the eye-tested budget:
H1-direct is **supported** if RT-DETR-L's mAP@0.5 drop is smaller than YOLOv10l's by at least **0.35**, with the 95% paired bootstrap CI on that difference excluding zero.
Both drops are means over seeds 0-2; each seed's images are attacked with that seed's weights.
Otherwise H1-direct is **not supported**, reported with the difference and its CI, and the write-up says plainly which way it went.
*(Same threshold and CI rule as the original H1, so the two answers are comparable.)*

Secondary readings, reported but not part of the verdict: attack success on plates every model finds when clean; cross-architecture transfer between the direct attacks; LPIPS of each model's noise.

## 6. Order of work

1. Score adapters and per-family preprocessing in the attack, with tests that match `predict()`.
2. Gate 1: strength check on validation (owner runs; about 6 short runs).
3. Gate 2: attack seed 0 of each model on validation at Y 3, Cb/Cr 4; eye test both (one 80-trial session).
4. Final run on the test set: six attacks (two models, three seeds), all evaluated; report and verdict.
5. H2 scoring (section 7); can run any time, it needs no GPU.
6. Records in `experiment-decisions.md`, a study document for the owner (`docs/study/milestone-2b.md`), the README results line.

**If time runs out, cut in this order:** the cross-architecture transfer by-product, then seeds 1-2 of both models (verdict on seed 0 with a note), never the strength check, the eye tests or the pre-registered rule.

## 7. H2, scored by its written rule

H2 (main design): putting most of the budget in chroma hides plates from YOLOv8s about as well as an RGB attack, while being less visible.
Rule: YCbCr's attacked YOLOv8s recall within 0.05 of RGB's (95% CI of the difference inside plus or minus 0.05), with lower LPIPS.

- Scored on milestone 1's pair, the one the rule was written for: `m1_h1_main` (Y 2, Cb/Cr 12) against `m1_h1_rgb` (RGB 8), on the same test images, with a paired bootstrap CI on the recall difference.
  LPIPS for both comes from per-plate crops of those caches (the visibility stage), on the same plates.
  Both caches still hold all 1,731 test images (`cache/adv/360ea0dd4bfc` and `cache/adv/39d655a3e07b`, the fingerprints in those runs' `metrics.json`), so no attack is rerun and the test set is not attacked again; only LPIPS is computed.
- The verdict is reported by the rule as written.
  The data seen so far suggests the recall part holds (0.054 against 0.081) and the LPIPS part fails (0.038 against 0.026 on the 300-image subsample).
- Reported beside it, labeled as evidence the rule did not name: the eye test, where at equal LPIPS the chroma attack passed and the RGB one failed, and the ceilings, where the eye allowed a larger chroma budget.
  The write-up states that LPIPS and the human viewer disagree about chroma noise, and that the rule named LPIPS.
- The H2 chart from the main design: hiding power (attacked YOLOv8s recall) against LPIPS for every visibility rung, with the eye-test result marked on each point.

## 8. Constraints

- Code: Claude writes all of it; documentation for the owner's review comes after (as in 2a).
- Same repo conventions: configs and sweeps under `configs/`, runs and reports committed under `results/`, decisions in `experiment-decisions.md`, tests and lint green at every commit, no AI co-author or sign-off lines in commits.
- GPU runs: the owner takes CPU cores 6, 7, 22, 23 offline and turns boost off first; jobs are pinned with `taskset -c 0-5,8-21,24-31`; jobs over about 10 minutes are run by the owner unless the owner asks Claude to run them.
- Commands given to the owner are split over short lines (a wrapped paste broke one run in 2a).

## 9. Left to the builder

- The adapter and preprocessing implementation, and where the query-selection gradient question is handled, subject to the `predict()` match test and the strength check.
- Sweep file layout and run names.
- How the report shows white-box rows for six attacked image sets and the H1-direct line.
- Timing estimates, measured on one image before each long run and given to the owner.
