# Milestone 2a study notes: making the attack transfer, and keeping it invisible

These notes walk through what Milestone 2a built and why, in the order the work happened.
Every number comes from `docs/experiment-decisions.md` or a committed report; file and line pointers are to the code as of the end of the milestone.
Each section ends with a couple of questions to check your understanding, and short answers are at the end.

## 0. The question and the answer in one paragraph

Milestone 1 could not test H1 (does the CNN-made noise fool a CNN much more than the Transformer?), because the noise barely moved the CNN control, YOLOv10l.
Milestone 2a tried to fix that with three standard transfer techniques, while holding the noise below what a careful viewer can see.
The eye test put that limit far lower than Milestone 1's budgets, and at that limit the improved attack dropped YOLOv10l's mAP@0.5 by 0.069 against the 0.35 the gate needs.
So the result is: this attack does not transfer to the CNN control at an invisible budget, and H1 remains untested.
That is a real finding, not a failure to finish: it says the invisible budget, not the architecture gap, is the binding limit.

## 1. Why single-surrogate PGD overfits

PGD (projected gradient descent) repeats one move 40 times: compute how the detector's plate scores change with each pixel, step every pixel a little in the direction that lowers them, then clip the noise back inside the budget.
The gradient it follows belongs to one set of weights, YOLOv8s seed 0.
With 40 steps the noise finds directions that this particular network is sensitive to, including quirks that come from its random initialization and training order, not from what a plate looks like.

Milestone 1's main run (`results/reports/m1/`) shows the pattern clearly:

| Target | mAP@0.5 drop | Attack success |
|---|---|---|
| YOLOv8s seed 0 (the surrogate, white-box) | 0.896 | 0.942 |
| YOLOv8s seeds 1, 2 (same architecture, retrained) | 0.065 | 0.089 |
| YOLOv10l (another CNN) | 0.035 | 0.041 |
| RT-DETR-L (Transformer) | 0.001 | 0.002 |

The biggest fall is not from CNN to Transformer.
It is from the surrogate to *the same architecture with different weights*: 0.896 to 0.065.
The noise had learned YOLOv8s seed 0, not "how to hide a plate from a YOLO".
That is what "overfitting to the surrogate" means here, the same idea as a model that memorizes its training set.

**Check yourself**
1. Why is the drop on YOLOv8s seeds 1 and 2 the most telling number in that table?
2. If the noise were transferring well, roughly what would you expect the retrained row to look like?

## 2. The three transfer techniques

All three live in `pgd()` in `src/plate_cloak/attack/pgd.py`, and each is a config switch that defaults to off, so Milestone 1's configs still describe Milestone 1's attack (`pgd.py:12-16`, `configs/base.yaml`, `attack.momentum` and `attack.diversity`).

### The ensemble and its normalization

Instead of one surrogate, the attack follows three: YOLOv8s, YOLOv8n and YOLOv8l, all seed 0 (fine-tuned for this milestone; best validation mAP@0.5 0.898, 0.895 and 0.890).
A direction that lowers the scores of three different networks at once is less likely to be one network's quirk.

The combination is at `pgd.py:165-171`: each surrogate's gradient is computed separately, divided by that surrogate's loss on the clean image (`scale`, computed once at `pgd.py:149-151`), and the results are averaged.
The division is the normalization.
Without it, a surrogate whose scores happen to be larger would dominate the sum, and the "ensemble" would quietly be one model.
Dividing by the clean loss puts every surrogate on the same footing: each contributes its *relative* progress.

A detail worth knowing: the code scales each gradient after the backward pass, not the loss before it (the comment at `pgd.py:161-164`).
Scaling the loss first is mathematically the same, but floating-point rounding then flips the sign of near-zero gradient entries, and PGD steps by sign.
With the scaling done afterwards, one surrogate reproduces Milestone 1's images exactly, which is how we know the new code did not change the old attack.

### Momentum (MI-FGSM)

At `pgd.py:174`, `velocity = mu * velocity + grad / mean|grad|`, and the step follows `sign(velocity)` instead of `sign(grad)`.
With `mu = 1.0` the step direction is a running sum of every gradient seen so far, each scaled to the same size.
A single step's gradient can point at a narrow, model-specific dip; the running sum keeps the directions that recur step after step.
It is the same idea as momentum in training, used here to stop the noise from chasing the surrogate's local quirks.

### Input diversity (DI)

At `pgd.py:159-160` and the helper `diverse()` at `pgd.py:180-195`: on each step, with probability 0.5, the image the surrogates see is shrunk by a random factor between 0.9 and 1 and pasted back at a random offset on the gray padding.
The noise therefore has to work on slightly different versions of the image, not on one exact pixel grid.
Noise that only works at one exact alignment is fragile; another detector, which resizes and pools differently, would never see that alignment.

The random draws come from a generator seeded per image (`attack/stage.py:163` and `:173`), so an interrupted attack redoes an image with the same draws and the cache stays consistent.

### What they did, on validation

Validation split, at the ceiling of that moment (Y 3, Cb/Cr 4), mAP@0.5 drop:

| Variant | YOLOv8s s0 (white-box) | YOLOv8s s1 | YOLOv10l |
|---|---|---|---|
| ensemble | 0.399 | 0.054 | 0.038 |
| + input diversity | 0.294 | 0.094 | 0.066 |
| + momentum | 0.323 | 0.084 | 0.061 |
| + both | 0.267 | 0.114 | **0.085** |
| + both, 100 steps | 0.292 | 0.118 | 0.091 |

Read it as a trade: each technique lowers the white-box drop and raises the transfer drops, and the two add up.
That is the expected signature of less overfitting: the noise gives up some of its grip on the surrogates to become more general.
More steps barely help, so the attack was not short of iterations; the budget is what runs out.

**Check yourself**
1. What would go wrong with the ensemble if the per-surrogate division by the clean loss were removed?
2. Why does the white-box drop go *down* when input diversity is switched on, and why is that not a bad sign?

## 3. What LPIPS measures

LPIPS (Learned Perceptual Image Patch Similarity, Zhang et al. 2018) compares two images by passing both through a pretrained network (here AlexNet), taking the activations at several layers, and measuring how different they are, with per-channel weights fitted so the score agrees with how people judged image pairs.
0 means identical; larger means more different.
It is far closer to human judgment than pixel error, because the same pixel change can be invisible on a busy texture and obvious on a flat surface.

The code is `src/plate_cloak/visibility/lpips_stage.py`: one score per plate crop (attacked against clean, `lpips_scores`), summarized per budget in `results/visibility/<run>/summary.json`.
Crops with a side under 32 pixels are skipped, because AlexNet's features are too coarse for them.

LPIPS is *reported, not used as the pass rule*: there is no established LPIPS threshold for "invisible" on small plate crops.
And our data shows why a fixed threshold would have been wrong.
RGB 4 failed the eye test clearly (33 of 40 correct) at LPIPS 0.008, while YCbCr Y 3, Cb/Cr 4 passed (24 of 40) at LPIPS 0.011.
The eye was more sensitive to the RGB noise than LPIPS predicts, so LPIPS ranks budgets well within one color space but does not rank them across spaces.

**Check yourself**
1. Why is LPIPS a better visibility measure than the largest pixel change?
2. Give the pair of results from this milestone that shows LPIPS cannot be the pass rule.

## 4. The eye test: why two-alternative forced choice, and where 25 of 40 comes from

### Why 2AFC

Each trial shows two crops of the same plate side by side, one clean and one attacked, in random order, and you must pick the attacked one (`src/plate_cloak/visibility/eyetest.py`).
The alternative, showing one crop and asking "is this attacked?", mixes up two things: how well you can see the noise, and how willing you are to say yes.
A cautious viewer says no a lot and looks unable to see noise that a bold viewer would report.
Forcing a choice between two removes that bias, and it gives an exact meaning to "cannot see it": a viewer who sees nothing is right half the time, no more and no less.
Showing the clean copy right next to the attacked one is also the hardest case for the attack, which is what a ceiling should be set against.

### Where 25 of 40 comes from

If you are guessing, the number of correct answers in 40 trials follows a binomial distribution with p = 0.5: on average 20, spread around it.
The pass rule asks: how many correct answers would be surprising for a pure guesser?

- P(26 or more correct | guessing) = 0.040
- P(25 or more correct | guessing) = 0.077

26 is the smallest count a guesser reaches less than 5% of the time, so 26 or more counts as "the viewer can see it", and 25 or fewer passes (`N_TRIALS = 40`, `MAX_CORRECT = 25` at `eyetest.py:45-46`; the rule is `passes()` at `eyetest.py:117`).
The rule was fixed before the first trial and never moved.

This is why some rounds felt like guessing and still failed.
26 of 40 is 65%: you were picking up a faint cue slightly more often than chance, even if no single trial felt certain.
Three rungs failed by exactly one answer (Cb/Cr 8, Cb/Cr 12, Y 4).

### The ladders and the ceiling

A ladder is a list of budgets on one axis (Cb/Cr, Y, or RGB).
The ceiling is the largest budget that passes *with every smaller one also passing* (`ladder_ceiling()` at `eyetest.py:124`), so a lucky pass above a failure does not count; Cb/Cr 16 passed at 23 of 40 but sits above failures.
Rungs were added one at a time when a ladder ran out, because testing two lower rungs together would let a miss on the lower one cancel a pass on the higher.

Over five sessions and 520 trials the ceilings came out as YCbCr Y 3, Cb/Cr 4 and RGB 2.
Milestone 1's budgets (Cb/Cr 12, RGB 8) turned out to be visible.
Your cues were a grainy, gridded texture, an impression of blur, and faint swirls near the characters.
The crops were checked for a pipeline artifact behind those cues and none was found: the attacked crop differs from the clean one by at most the budget, and it is slightly *sharper*, not blurrier (the "blur" is most likely the clean plate's smoothness next to the grain).
PGD pushes most pixels to the edge of the budget, which shows as grain on the plate's flat background; that is why RGB stayed visible from 12 all the way down to 4.

### The final check

The chosen transfer attack makes coarser noise than plain PGD (more of its energy survives a blur), and coarser patterns are easier to see.
So the spec re-tests the final attack on its own, on validation crops: at Y 3, Cb/Cr 4 it failed (28 of 40), so it stepped down one rung to Y 2, Cb/Cr 4, where it passed (24 of 40).
That is the budget the test set saw.

A word on strictness: a side-by-side comparison at full size, by a viewer with 500 trials of practice who knows exactly what PGD noise looks like, is much harder on the attack than a passer-by or a camera operator glancing at one image.
The ceiling is conservative, which makes "invisible" a strong claim.

**Check yourself**
1. Why does 2AFC make "the viewer cannot see it" mean "about 50% correct", and why does a yes/no test not give you that?
2. Cb/Cr 16 passed at 23 of 40. Why is it not the Cb/Cr ceiling?

## 5. Why development stayed on validation

Choosing an attack is fitting a model, with the variant as the parameter.
If the variants had been compared on the test set and the best one reported, the reported number would be the maximum of several noisy measurements, biased upward, the same way a model tuned on its test set looks better than it is.
So every choice was made on a separate validation split of 604 images that shares no image with the test set (`splits/openimages_val_eligible.txt`): the four-variant ablation, the 100-step rerun, and the final eye check.
The test set (1,731 images) was attacked once, with the attack fixed in advance (`configs/sweeps/transfer_final.yaml`).

Two more guards, for the same reason:
- RT-DETR was not a target during development, so its number could not steer which technique was picked.
- The ceiling was frozen before any transfer run, from white-box runs only, so the budget could not be raised after seeing that transfer was weak.

The rule for the winner was also fixed in advance: the largest YOLOv10l drop by point estimate, and the 100-step rerun kept only if it beat that by more than the winner's confidence-interval half-width (0.085 plus 0.020; it reached 0.091, so 40 steps stayed).

**Check yourself**
1. Why would picking the best of four variants on the test set overstate the result?
2. Why was RT-DETR left out of the validation runs, when it is the model H1 is about?

## 6. How to read the final report

The report is `results/reports/transfer_final/h1_table.md` (with `h1.json` and `h1_bars.png`), built by `src/plate_cloak/analysis/report.py`.
There is one section per attack: `transfer_final_ycbcr` (the main attack, Y 2, Cb/Cr 4) and `transfer_final_rgb` (the comparison, RGB 2).

**The rows** (`architecture_rows()` at `report.py:104`):
- `(white-box)`: a surrogate the attack was computed on; this is the attack's strength with full knowledge of the model.
- `(retrained)`: other seeds of a surrogate architecture; same design, different weights.
- the rest: transfer targets, the CNN control YOLOv10l and the Transformer RT-DETR-L.

**The columns**: clean and attacked mAP@0.5 and recall, and the drop (clean minus attacked).
Values are means over seeds, with 95% confidence intervals from a paired bootstrap over images: the same resampled images are scored clean and attacked, so the interval reflects how much the drop depends on which images happened to be in the test set.
"Drop range" is the smallest and largest single-seed drop, which shows how much the result depends on the training seed.

**The gate line** (`transfer_gate()` at `report.py:175`): YOLOv10l's drop must have its whole CI at or above 0.35.
This asks whether the attack fools a CNN it was not computed on.
If it does not, a small RT-DETR drop says nothing about attention, because the attack is not fooling unseen CNNs either.

**The H1 line**: YOLOv10l's drop minus RT-DETR's, which H1 needs to be at least 0.35 with its CI above 0.
The report prints it whatever the gate says; read it only when the gate passed.
(This milestone's report prints "H1 not supported" next to a failed gate; the correct reading for that case, and the wording in `experiment-decisions.md`, is "H1 remains untested". The wording fix is noted for later.)

**The attack success line**: of the plates every target found on the clean image, the share each target misses on the attacked one.

The final numbers, mAP@0.5 drop:

| Target | YCbCr Y 2, Cb/Cr 4 | RGB 2 |
|---|---|---|
| YOLOv8s, n, l (white-box) | 0.19 to 0.21 | 0.10 to 0.11 |
| YOLOv8s (retrained) | 0.087 | 0.044 |
| YOLOv10l | 0.069 [0.060, 0.078] | 0.033 [0.028, 0.039] |
| RT-DETR-L | 0.006 | 0.007 |

How to say it:
- The gate is not passed at either ceiling, so H1 remains untested at an invisible budget.
- The ordering is the one H1 predicts (white-box, then retrained, then YOLOv10l, then RT-DETR), but the CNN control moves too little for the gap to be evidence about attention.
- Even the same architecture with other weights loses only 0.087, so at this budget the noise mostly exploits each model's own weights.
- Against Milestone 1 (YOLOv10l 0.035 at the visible Cb/Cr 12), the techniques double the transfer at a third of the chroma budget.
- At their ceilings YCbCr transfers about twice as well as RGB, because the eye tolerates a larger chroma budget than an RGB one; that matters for H2.

**Check yourself**
1. Why is a small RT-DETR drop not evidence for H1 when YOLOv10l's drop is also small?
2. What does the "retrained" row tell you that the YOLOv10l row cannot?

## Answers

**Section 1.**
1. It holds the architecture fixed and changes only the weights, so the fall from 0.896 to 0.065 can only come from the noise fitting one network's weights.
2. Close to the white-box row: a large drop, well above YOLOv10l's.

**Section 2.**
1. The surrogate with the largest raw loss would dominate the gradient, and the ensemble would behave like that single model.
2. The noise is optimized against shifted and shrunk copies, so it fits the exact surrogate input less tightly; giving up white-box strength for generality is the intended trade, and transfer rose.

**Section 3.**
1. LPIPS weighs a change by where it happens, the way people do: the same pixel change is far more visible on a flat background than on a texture.
2. RGB 4 failed the eye test at LPIPS 0.008, while Y 3, Cb/Cr 4 passed at a higher LPIPS of 0.011.

**Section 4.**
1. A viewer who sees nothing picks either side with equal chance, so chance is exactly 50% whatever the viewer's caution; a yes/no answer also depends on how readily the viewer says yes.
2. A smaller rung, Cb/Cr 8 (and 12), failed; the ceiling needs every smaller rung to pass, so a pass above a failure is treated as luck.

**Section 5.**
1. The best of several noisy measurements is biased upward; the reported number would include the luck that made it the best.
2. So that its result could not influence which technique was chosen; it is seen exactly once, on the test set.

**Section 6.**
1. Because the attack is failing to fool an unseen CNN as well; the RT-DETR number cannot then separate "attention resists" from "the noise does not transfer at all".
2. How much of the attack is tied to the surrogate's own weights rather than to the YOLOv8 design: here most of it.
