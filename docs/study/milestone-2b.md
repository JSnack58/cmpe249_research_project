# Milestone 2b study notes: attacking each model directly, and scoring H2

These notes walk through what Milestone 2b built and why, in the order the work happened.
Every number comes from `docs/experiment-decisions.md` or a committed report; file and line pointers are to the code as of the end of the milestone.
Each section ends with a couple of questions to check your understanding, and short answers are at the end.

## 0. The question and the answer in one paragraph

Milestone 2a could not test H1, because noise built on YOLOv8 barely transferred to any other model at a budget the eye cannot see.
Milestone 2b took transfer out of the question: each model is attacked with gradients from its own weights, at the same invisible budget, and we compare how much each one breaks.
The question became: with full access to its weights, is a Transformer still much harder to fool than a size-matched CNN?
The answer is no.
YOLOv10l's mAP@0.5 drops by 0.466 and RT-DETR-L's by 0.381, a gap of 0.085 [0.069, 0.101] against the pre-registered 0.35, so **H1-direct is not supported**.
The gap is at most 0.085, and part of it may come from the attack being somewhat weaker on RT-DETR rather than from the architecture; either way it is far from decisive, and 2a's tiny RT-DETR number (0.006) came from transfer, not from attention.
H2 was also scored by its written rule on Milestone 1's runs: chroma noise hides plates as well as RGB noise but scores a higher LPIPS, so **H2 is not supported as written**, even though the eye test tells the opposite story about visibility.

## 1. Why attack directly

Transfer asks two things at once: does the noise survive the jump to new weights, and does it survive the jump to a new architecture?
2a showed the first jump already fails at an invisible budget: noise built on YOLOv8s seed 0 lost most of its effect on YOLOv8s seeds 1 and 2 (0.21 to 0.09), before any architecture changed.
So a small RT-DETR drop in a transfer test cannot tell "attention resists" apart from "the noise does not transfer at all".

A direct (white-box) attack removes the first jump.
Each of the six models (RT-DETR-L and YOLOv10l, seeds 0 to 2) gets its own attack, computed on its own weights.
If RT-DETR still breaks much less than YOLOv10l, that is about the architecture, because both attacks had full knowledge of their model.

The cost: a white-box attack is the strongest case, not a realistic one (a real adversary rarely has the weights).
The finding is therefore an upper bound on how much each architecture can be hurt at this budget.

**Check yourself**
1. Why does a transfer test need the CNN control to break first before RT-DETR's number means anything?
2. What does a white-box attack give up, compared with a transfer attack, as a picture of the real world?

## 2. The silent-wrong-tensor trap: making the attack see what evaluation sees

The attack code was written for YOLOv8, and two things about RT-DETR differ from it.

**The input.**
Evaluation runs each model through Ultralytics `predict()`.
For YOLO, `predict()` pads the image to a multiple of 32 pixels.
For RT-DETR, it *stretches* the image to a 640 by 640 square.
If the attack fed RT-DETR a padded image, it would optimize noise for an image the model never sees at evaluation.

**The output.**
YOLOv8 returns one score per anchor (thousands of candidate boxes), and the loss pushes the highest of them down.
RT-DETR returns a 300-row table of its best queries, plus the decoder's and encoder's raw scores tucked into a second element.
Reading RT-DETR's table as if it were YOLOv8's layout does not crash: it quietly optimizes a few numbers of the wrong tensor, and the model looks robust.
That is the most dangerous kind of bug in this project, because it produces exactly the result H1 predicts.

The fix is a **view** per family (`src/plate_cloak/attack/views.py`): it builds the input the way `predict()` does and reads the plate scores from the right part of the output.
`DetectView` pads and reads YOLO's per-anchor scores; `RTDETRView` stretches and reads the decoder's scores (`views.py:87`).
A slow test (`tests/test_views_real.py`) runs `predict()` on real images with hooks on the network, and checks that the view's input is within one 8-bit level of `predict()`'s, and that every detection `predict()` keeps is one of the scores the view reads.

That test also turned up a surprise: **our YOLOv10l is not NMS-free.**
Our fine-tuned checkpoints load with `end2end = False`, so `predict()` drops YOLOv10's one-to-one head and runs its one-to-many head with NMS, exactly like YOLOv8.
Earlier documents called the CNN control NMS-free; no number changes (evaluation was the same throughout), but the write-up must not say that.

**Check yourself**
1. Why is a wrong score reading worse for this project than a crash?
2. What two things does the `predict()` match test compare?

## 3. Gate 1: does the attack actually work?

Before any verdict, the attack had to prove it can hurt both models.
The check uses a deliberately large, visible budget (RGB 16) on validation: if a model barely drops even there, the attack is broken, not the model robust.
The pre-registered bar was a 0.80 mAP@0.5 drop for every seed.

The first run:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| RT-DETR-L, first attack | 0.319 | 0.324 | 0.396 |
| YOLOv10l | 0.892 | 0.860 | 0.747 |

RT-DETR failing this badly was the warning the check exists for.

**What was wrong.**
RT-DETR works in two stages.
Its *encoder* scores every location in the image and passes its top 300 to the *decoder* as queries; the decoder refines them into the final boxes and scores.
The first attack pushed down only the decoder's scores.
Each step, the encoder still proposed the plate as a query, and the decoder found it again.
The evidence: plates survived exactly where the encoder still proposed them.
The fix makes the attack push the encoder's proposal scores too (`RTDETRView.attack_scores`, `views.py:112`), with the loss unchanged.
On the same 60 validation images the drop went from 0.287 to 0.815.
More steps (100 instead of 40) and a different loss were tried on the way and ruled out.
The cache fingerprint now records how RT-DETR's scores are read, so the broken attack's images can never be reused by mistake.

After the fix:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| RT-DETR-L, fixed attack | 0.774 | 0.737 | 0.810 |
| YOLOv10l | 0.892 | 0.860 | 0.747 |

Three seeds sat just under 0.80.
You lowered the bar to 0.70 for every seed of both models, and the decision is recorded as made after seeing the data.
The reasons: 0.80 was a guess from YOLOv8s (0.94 at RGB 12) made before any 2b data; both models land in the same band; the check had already done its job by catching the broken attack; and the bar applies to both models alike and does not touch the H1 threshold.
The lower bar cuts both ways, and it matters which way.
It let a weaker CNN attack through (YOLOv10l seed 2), which can only shrink the gap H1 needs.
It also let a weaker RT-DETR attack through (seeds 0 and 1), which can only *widen* that gap.
So the bar cannot have turned a supported verdict into "not supported", nor faked one: a stronger RT-DETR attack could only shrink 0.085 further, so "not supported" is safe.
What it does mean is that the small gap that remains (0.085) may partly be a weaker attack on RT-DETR, not RT-DETR resisting.
The reviewer also noticed that RT-DETR's loss is dominated by encoder scores (97 of its top 100 on one image), so the attack's balance between RT-DETR's two stages is a plausible reason it is a little weaker.

**Check yourself**
1. Why was the decoder-only attack so weak, even though the decoder produces the final scores?
2. Lowering the strength bar let weaker attacks through on both models. Which conclusion does that leave safe, and which does it weaken?

## 4. Gate 2: is each model's noise invisible?

2a showed that two attacks at the same budget can look different to the eye, so neither model's noise was assumed invisible.
Seed 0 of each model was attacked at the frozen ceiling, Y 3, Cb/Cr 4, on validation, and you took one 80-trial eye test (session `direct1`, 40 trials per model, sides balanced 20 left and 20 right).
The rule is 2a's: pass at 25 or fewer correct of 40.

- RT-DETR-L's noise: 17 of 40, pass.
- YOLOv10l's noise: 25 of 40, pass (exactly at the line).

Both passed, so the final run uses Y 3, Cb/Cr 4 for both models, with no step-down.

**Check yourself**
1. Why does the step-down rule move *both* models to Y 2 if only one fails?

## 5. The verdict, and how to read the report

The final run attacked all six models on all 1,731 test images, once, and showed every attacked set to all nine models.
The report is `results/reports/direct_final/h1_direct.md`, built by `src/plate_cloak/analysis/direct.py`.

**The verdict line** (`h1_direct()`): the CNN control's white-box drop averaged over seeds, minus the Transformer's, with one bootstrap over images shared by all six runs.
Because the runs attack the same images, resampling the same images for every run makes the difference paired: it measures how the gap depends on which images happened to be in the test set.

| | mAP@0.5 drop (seed mean) | Seeds |
|---|---|---|
| YOLOv10l (CNN control) | 0.466 [0.446, 0.485] | 0.483, 0.478, 0.437 |
| RT-DETR-L (Transformer) | 0.381 [0.362, 0.399] | 0.424, 0.329, 0.391 |
| Difference | **0.085 [0.069, 0.101]** | needs at least 0.35, CI above 0 |

**The transfer matrix** (a by-product, not the verdict): each attacked set's drop on every model.
It is near zero off the diagonal in every direction: RT-DETR's noise moves the YOLO models by at most 0.010, YOLOv10l's moves RT-DETR by at most 0.004, and even other seeds of the *same* model move only 0.03 to 0.05.

How to say it:
- H1-direct is not supported: with access to its own weights, RT-DETR breaks slightly less than YOLOv10l (0.38 against 0.47), nowhere near the 0.35 gap the rule asked for.
- The 0.085 gap is an upper bound on any architecture effect, not proof of one: the attack may be somewhat weaker on RT-DETR (see section 3).
- 2a's 0.006 for RT-DETR was a transfer effect: attacked directly at an invisible budget, RT-DETR loses 0.38.
- At this budget each attack exploits its own model's weights almost entirely; that, not the CNN-to-Transformer gap, is what limits cross-model cloaking.
- The same ordering shows at the visible strength-check budget (seed means 0.83 against 0.77); a slightly weaker RT-DETR attack would produce that ordering too, so it does not settle where the gap comes from.

**Check yourself**
1. Why does the verdict use one bootstrap shared by all six runs instead of one per run?
2. What does the near-zero transfer between seeds of the same model tell you?

## 6. H2, scored by its written rule

H2 (from the main design): putting most of the budget in chroma hides plates from YOLOv8s about as well as RGB noise, while being less visible.
The rule: YCbCr's attacked YOLOv8s recall within 0.05 of RGB's (the whole 95% CI of the difference inside plus or minus 0.05), *and* lower LPIPS.
It was scored on the pair it was written for, Milestone 1's `m1_h1_main` (Y 2, Cb/Cr 12) against `m1_h1_rgb` (RGB 8), on the same test images, from their saved attacked images (only LPIPS was computed; nothing was attacked again).
The report is `results/reports/h2/h2.md`, built by `src/plate_cloak/analysis/h2.py`.

- Recall: YCbCr 0.054, RGB 0.081; difference -0.027 [-0.036, -0.019], inside the band. **Holds** (chroma even hides slightly better).
- LPIPS: YCbCr 0.039, RGB 0.026; difference +0.012 [0.012, 0.013]. **Fails.**

So H2 is not supported as written.

The chart beside it (`h2_chart.png`) plots each budget's hiding power against its LPIPS, with the eye-test outcome marked.
It shows why the written rule and the eye disagree:
- At equal LPIPS, the eye saw RGB noise and not chroma noise: Cb/Cr 4 (LPIPS 0.0083) passed, 24 of 40; RGB 4 (LPIPS 0.0080) failed, 33 of 40.
- So at the eye-test ceilings, chroma hides more: Cb/Cr 4 leaves YOLOv8s recall at 0.37, RGB 2 at 0.66.
- LPIPS is a learned model of human judgment, trained mostly on brightness and texture differences; it overrates how visible chroma noise is, which matches what is known about human vision (we resolve color changes less finely than brightness changes).

How to say it: H2 fails the rule it was registered with, because LPIPS rates chroma noise as more visible; the human eye test, which the rule did not name, supports H2's idea.
Report both, and say plainly which one the rule named.

**Check yourself**
1. Which half of the H2 rule failed, and which piece of evidence disagrees with it?
2. Why must the write-up still say "not supported" even though the eye test favors chroma?

## Answers

**Section 1.**
1. If the noise does not even fool an unseen CNN, a small RT-DETR drop could just mean "the noise does not transfer", with nothing to say about attention.
2. Realism: a real adversary rarely has a deployed model's weights, so a white-box drop is an upper bound on the harm, not a likely case.

**Section 2.**
1. A crash is noticed and fixed; a wrong reading silently makes RT-DETR look robust, which is exactly the result H1 predicts, so it could have been reported as a finding.
2. The input the view builds against `predict()`'s input (within one 8-bit level), and the scores the view reads against the detections `predict()` keeps.

**Section 3.**
1. The encoder decides which 300 candidates the decoder ever sees; lowering decoder scores left the encoder proposing the plate every step, and the decoder found it again.
2. "Not supported" stays safe: a weaker CNN attack shrinks the gap, and a stronger RT-DETR attack could only shrink it further, so no fix to the attack could lift 0.085 to 0.35. What it weakens is reading the remaining 0.085 as RT-DETR resisting: a weaker RT-DETR attack (seeds 0 and 1 passed only at the lower bar) widens the gap, so part of it may be the attack, not the architecture.

**Section 4.**
1. The comparison needs equal budgets: if one model got a larger budget, a bigger drop could come from the budget rather than the architecture.

**Section 5.**
1. Every run attacks the same images, so a hard image is hard for both models; sharing the resamples cancels that common part, giving a tighter and correct interval for the difference.
2. The noise is fitted to one set of weights, not to the architecture or to "what a plate looks like"; that is why cross-model cloaking fails at an invisible budget.

**Section 6.**
1. The LPIPS half (chroma scored higher); the eye test, where chroma noise passed at the same LPIPS as RGB noise that failed.
2. The rule was registered before the data, and it names LPIPS; changing the measure after seeing the result would be choosing the rule to fit the answer.
