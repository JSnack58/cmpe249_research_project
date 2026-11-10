# Milestone 2a: A Transfer Attack That Gives H1 a Fair Test

Date: 2026-11-10.
Builds on the [experiment design](2026-10-20-plate-cloak-experiment-design.md) and the Milestone 1 result recorded in [experiment-decisions.md](../../experiment-decisions.md).

## 1. Why

Milestone 1 ran the planned H1 test on all 1,731 eligible test images with 3 seeds per model.
The attack (PGD on YOLOv8s seed 0, top-k loss, 40 steps) hid plates from the surrogate but barely moved any other model:

| Target | mAP@0.5 drop, main YCbCr (2, 12, 12) |
|---|---|
| YOLOv8s seed 0 (white-box) | 0.896 |
| YOLOv8s seeds 1-2 (retrained) | 0.065 |
| YOLOv10l (CNN control) | 0.035 |
| RT-DETR-L (Transformer) | 0.001 |

H1 asks whether RT-DETR's drop is at least 0.35 smaller than the CNN control's.
With YOLOv10l losing only 0.035, that is impossible by construction, so Milestone 1 did not really test H1.
The attack has plenty of power (0.896 on the surrogate) but overfits the one model it was computed on; the random-noise control (0.010) shows that noise size alone does nothing.

This milestone builds an attack that transfers to the CNN control at an invisible budget, so that H1 gets a fair test.

## 2. Goals, non-goals, anti-goals

**Goal:** an attack that clears the **transfer bar** (below) on YOLOv10l, which never contributes a gradient, while staying invisible to the naked eye.
Then rerun the H1 test with it.

**Transfer bar (agreed with the owner):** YOLOv10l's 3-seed mean mAP@0.5 drop is at least 0.35, with the lower end of its 95% bootstrap CI at or above 0.35.
A single seed clearing it does not count: that would show the attack suits one training run, not the architecture.
0.35 is the H1 threshold; below it, H1 cannot hold however well RT-DETR resists.

**The owner's sequence:**

1. Break the CNN control (this milestone).
2. Transfer test: show the same images to RT-DETR and apply the unchanged H1 rule. Either outcome is a real answer.
3. Direct white-box attack on RT-DETR, to separate "resists transferred noise" from "resists noise". **Not in this milestone:** RT-DETR's output is 300 decoder queries, not YOLO's candidate grid, so it needs its own loss and its own design.

**Non-goals:**

- Changing H1's threshold or decision rule.
- The Sobel mask, SSIM, and the H2 chart (later H2 work); this milestone only builds the part of H2 it needs (LPIPS and plate crops).
- Making GPU PGD bit-for-bit deterministic.

**Anti-goals (a failure even if the numbers come out):**

- **Visible noise.** Breaking YOLOv10l by effectively painting over the plate tests robustness to corruption, not adversarial transfer, and defeats the project's premise.
- **Tuning on the test set.** Trying tricks while watching YOLOv10l's test-set number, and stopping when it clears the bar, makes the final number optimistic. Development uses validation images (section 5).
- **A ceiling chosen after seeing transfer results.** The budget ceiling is fixed by a perceptual test before any transfer run (section 3).
- **Surrogates that are also reported as transfer targets.**

## 3. Constraint: the visibility ceiling

**Owner's constraint:** the noise must stay invisible to the naked eye; budget tweaks are allowed within that, because H2 is tested on the same attack.

Milestone 1's budgets are not the ceiling.
RGB 8/255 is a literature convention, not measured on our plates.
Cb/Cr 12 is the *smallest* value that matches RGB 8's hiding power (the Task 12 rule), not the largest invisible one.
Eyes resolve color more coarsely than brightness, so chroma likely has room, and Y little.

**Procedure (agreed):**

1. **Visibility sweep, white-box only.** Targets: YOLOv8s seed 0 only, so no transfer numbers exist to bias the choice.
   Uses the Milestone 1 attack (single surrogate, no transfer tricks). *(Claude's call: transfer tricks change the noise pattern, so the eye test is repeated on the final attack, step 5 below.)*
   Ladders, all PGD, top-k, 40 steps:
   - YCbCr Cb/Cr at Y 2: 12, 16, 20, 24 (Cb = Cr).
   - YCbCr Y at the largest passing Cb/Cr: 3, 4.
   - RGB: 8, 10, 12.
   Run on a 300-image subsample of the test set (`data.n_eval=300`); only the white-box model sees them. *(Claude's call.)*
2. **LPIPS** on each plate crop (attacked vs. clean), recorded per plate and summarized per budget.
   LPIPS is reported, not used as the pass rule: no threshold for "invisible" is established for our crops. *(Claude's call.)*
3. **Eye test, the pass rule (fixed now, before looking):**
   - The owner is the only viewer. An expert viewer who knows what PGD noise looks like makes this a strict test.
   - Two-alternative forced choice: each trial shows a clean and an attacked crop of the same plate, side by side, in random left/right order; the viewer picks the attacked one.
   - 40 trials per budget, plates drawn at random (fixed seed) from eligible plates; budgets interleaved in random order so the viewer never knows which budget a trial tests.
   - Crops shown at their native pixel size (no zoom), grown by the attack margin plus some context so the noise edge is not a giveaway. *(Context size: builder's call.)*
   - A budget **passes** if the viewer picks the attacked crop in at most 25 of 40 trials (26 or more has one-sided binomial p ≈ 0.04 under guessing).
   - The ceiling on each ladder is the largest budget that passes with every smaller budget on that ladder also passing.
   - **Ladder ends** *(Claude's call)*:
     if the top rung passes, extend the ladder by the same step (Cb/Cr 28, 32, ...; Y 5, ...; RGB 14, 16, ...) until a rung fails;
     if the bottom rung fails, extend downward (Cb/Cr 8, 4; RGB 6, 4) until one passes, and record that the Milestone 1 budget was visible.
     Extending is mechanical and needs no judgment, so it stays blind.
4. **Freeze** the ceilings (one RGB, one YCbCr) in `experiment-decisions.md`, with the trial counts, before any transfer development.
5. **Re-check the final attack, before the test set is touched.** The chosen transfer attack at the ceiling gets its own 40-trial eye test with the same rule, on crops attacked from validation images. If it fails, step down one rung and recheck; the ablation is not redone at the lower rung. *(Claude's call: doing this before the test run keeps the test set single-use and the ceiling independent of transfer results.)*

The eye-test tool is a small local page or script that shows trials and writes each answer (trial id, plate, budget, side, choice) to a committed CSV, so the result is auditable.

## 4. The attack

Three standard transfer techniques, each a config switch so a sweep can show what each one contributes.
All three default to off, so Milestone 1's configs still describe the Milestone 1 attack.

**Ensemble surrogates.**
Each PGD step uses the gradient of the combined loss over several surrogates, so the noise must fool all of them.
- Surrogates: YOLOv8s seed 0, plus YOLOv8n and YOLOv8l fine-tuned with seed 0 (the A3 models, which the spec already plans; owner agreed to fine-tune rather than use third-party weights).
- Combined loss: mean over surrogates of each one's top-k loss, each divided by that surrogate's loss on the clean image, so a model with larger scores does not dominate. *(Claude's call.)*
- Config: `models.surrogate` becomes a list `models.surrogates`; a one-element list is the Milestone 1 attack.

**Input diversity (DI).**
Before each forward pass, with probability `p`, resize the padded image by a random factor in `[r_min, 1]` and pad it back to its size at a random offset, the same transform for every surrogate in that step.
Starting values from the DI literature: `p = 0.5`, `r_min = 0.9`. *(Claude's call.)*
Randomness comes from a per-image generator seeded like the random control (`image_seed`), so a resumed run uses the same draws.

**Momentum (MI-FGSM).**
Accumulate `g = mu * g + grad / mean(|grad|)` over steps and step along `sign(g)`; `mu = 1.0` as in the MI-FGSM paper. *(Claude's call.)*

Everything else is unchanged: top-k loss, 40 steps, step size 2.5 x budget / steps, noise only on plates plus 10%, gamut fit and quantization margin for YCbCr, the 8-bit audit.
More steps are allowed if development shows the ensemble needs them; steps are part of the fingerprint like every other setting.

**Fingerprint.** The attack cache key must change when any new setting changes the pixels, and only then.
Two existing problems are fixed along the way: the fingerprint covers budgets the chosen space never uses (an RGB attack is re-run when Cb/Cr changes), and it covers only one surrogate's weights.
Every surrogate's weights hash goes into `settings.yaml`.
Old cache folders may be recomputed rather than migrated; committed Milestone 1 run folders keep the fingerprint they recorded. *(Builder's call how.)*

**Docstrings.** `pgd()` and the attack stage claim the attack is deterministic; on GPU it is not (cuDNN's non-deterministic backward plus `sign()` flips a few pixels: 123 of 1,731 images differed between two identical RGB runs, with no measurable effect on metrics). Correct the claim and record the finding.

## 5. Experiments and decision rules

All transfer development uses **validation images** (the 724-image validation split, filtered by the same eligibility rule, frozen as its own list).
The test set is used once, for the final run.
The fine-tuned models picked their best checkpoint on validation, which does not bias an attack developed there. *(Claude's call.)*

1. **Fine-tune** YOLOv8n and YOLOv8l, seed 0, with the pinned recipe; record validation mAP next to the existing table.
2. **Visibility ceiling** (section 3), then freeze it.
3. **Transfer ablation on validation**, at the YCbCr ceiling, targets YOLOv8s seeds 0-2, YOLOv8n/l, and YOLOv10l seeds 0-2 (RT-DETR left out while developing, so its number cannot steer choices). *(Claude's call.)*
   Variants: ensemble only; ensemble + DI; ensemble + MI; all three.
   Pick the variant with the largest YOLOv10l 3-seed mean mAP@0.5 drop.
   **The whole search, fixed in advance** *(Claude's call)*: those four variants, plus one rerun of the winner at 100 steps instead of 40, kept only if it raises YOLOv10l's drop by more than its CI half-width.
   DI's `p` and `r_min`, momentum `mu`, top-k `k` and the step factor stay at the values in section 4.
   Whatever wins goes to the test set even if it misses the bar on validation; the test run is the result either way.
4. **Eye test of the final attack** (section 3, step 5), on validation crops.
5. **Final run on the test set, once**: the chosen variant at both ceilings, all targets including RT-DETR, 3 seeds each.
   Report as in Milestone 1, with the ensemble members as white-box rows.
6. **Gate and verdict** *(Claude's call, following the original spec where YCbCr is the main attack)*:
   - The gate and the H1 verdict are decided at the **YCbCr ceiling**. RGB is reported beside it with its own gate and verdict, as a comparison.
   - If YOLOv10l clears the transfer bar, apply the H1 rule unchanged and report the verdict.
   - If it does not, report that this attack does not transfer to the CNN control at an invisible budget, and that H1 remains untested. That is a result, not a reason to keep adding tricks.
   - If only RGB clears the bar, report the H1 verdict for RGB as secondary evidence, and the split itself as an H2-relevant finding (chroma noise transfers worse).

## 6. Reporting

- The report's architecture rows handle a list of surrogates: each surrogate that is also a target is a white-box row; other seeds of a surrogate architecture form a "retrained" row; every other architecture is a transfer row.
- The Milestone 1 report and runs stay committed and unchanged; the new runs sit beside them.
- `experiment-decisions.md` gains: the Milestone 1 H1 result, the GPU non-determinism record, the frozen ceilings with eye-test counts, the fine-tuning results for YOLOv8n/l, the ablation table, and the final verdict.
- The eps diagnostic sweep (`configs/sweeps/eps.yaml`, RGB 8/16/32, all targets) runs independently of this design; its figure shows how transfer grows with budget alone and is labeled as beyond the visibility ceiling.

## 7. Schedule and division of labor

- **Deadline: 2026-11-17** (the original schedule's end, confirmed by the owner).
- **If it does not fit, cut in this order** *(Claude's proposal, not objected to)*: the RGB final run, then the Y-ladder rungs (Y stays at 2), then YOLOv8l (the ensemble becomes YOLOv8s seed 0 + YOLOv8n).
  The YCbCr final run, the eye tests and the freeze are never cut.
- **Claude writes all of this milestone's code** (owner's decision).
  Study documentation explaining the new pieces (ensemble loss, input diversity, momentum, LPIPS, the eye test) is written afterwards for the owner to learn from.
- The owner runs the long GPU jobs from their own terminal (to watch progress) and takes the eye tests.

## 8. Left to the builder

- Module boundaries for the ensemble, DI and momentum, and how the loss normalization is computed.
- Crop context size and layout of the eye-test page, as long as it is blind and logs every answer.
- LPIPS backbone (AlexNet, the LPIPS paper's default, unless there is a reason otherwise) and how crops smaller than its minimum input are handled; report how many were skipped.
- How the validation eligibility list is frozen (same rule and tooling as the test list).
- Sweep file names and variant names.
