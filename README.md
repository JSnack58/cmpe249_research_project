# Cross-Architecture Transferability of Perceptual License Plate Cloaking

Research project for CMPE 249.

**Question:** Do YCbCr-constrained adversarial perturbations generated on a CNN detector (YOLOv8) transfer zero-shot to a Transformer detector (RT-DETR) for license plate evasion?
Or does global self-attention act as an implicit defense against local feature suppression?

## Status

Planning the main experiment.
So far there are two feasibility spikes; open design questions are tracked in [docs/experiment-decisions.md](docs/experiment-decisions.md).

### Spike 01: attack feasibility

[`spikes/01_attack_feasibility/`](spikes/01_attack_feasibility/) is a throwaway test of whether plate-restricted PGD noise can beat a YOLOv8 plate detector at all.
On 200 test images (YOLOv8n, 40 PGD steps, noise only inside the plate box):

| Attack | Plate recall |
|---|---|
| Clean | 0.98 |
| PGD, RGB L_inf 8/255 | 0.24 |
| PGD, YCbCr (Y 2/255, Cb/Cr 8/255) | 0.27 |
| Random noise, RGB 8/255 | 0.99 |

Findings that shape the main experiment:

- PGD works, but does not yet reach the target of recall at or below 0.05; small plates survive most often.
- The first YCbCr implementation leaks its Y budget (measured 5.9/255 vs. 2/255) because clamping in RGB alters luminance on bright pixels.
- RGB SSIM and luminance SSIM can disagree about chroma noise, so the fidelity metric must be fixed before running the real experiments.

### Spike 02: same attack on Open Images plates

[`spikes/02_openimages/`](spikes/02_openimages/) reruns the Spike 01 attack code unchanged on Open Images V7 plates (mostly UK/EU, roughly 15-20% US), using Spike 01's detector without retraining.
200 test images with a plate at least 40x10 px at the 640 input:

| Attack | Plate recall |
|---|---|
| Clean | 0.70 |
| PGD, RGB L_inf 8/255 | 0.05 |
| PGD, YCbCr (Y 2/255, Cb/Cr 8/255) | 0.06 |
| Random noise, RGB 8/255 | 0.67 |

Findings:

- The detector transfers poorly to the new domain (0.98 to 0.70 clean recall), so every model in the main experiment must be fine-tuned on the chosen dataset.
- The attack looks far stronger here, but against a low-confidence, out-of-domain detector; this is evidence that clean accuracy confounds attack success, not that these plates are easier to hide.
- Open Images labels some plates that uploaders redacted for privacy, and some plates large enough to pass the size filter are too blurry to read.

To rebuild the dataset: `python spikes/02_openimages/download_openimages_plates.py` (about 2.5 GB).

## Data and model credits

- Dataset: [keremberke/license-plate-object-detection](https://huggingface.co/datasets/keremberke/license-plate-object-detection), originally the Roboflow "Vehicle Registration Plates" dataset by Augmented Startups, licensed CC BY 4.0.
  Images shown in the spike figures come from this dataset.
- Dataset: [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), class "Vehicle registration plate".
  Annotations are licensed CC BY 4.0 by Google LLC; images are listed as CC BY 2.0 by their Flickr authors.
  Images shown in the Spike 02 figures come from this dataset.
- Spike detector: [yasirfaizahmed/license-plate-object-detection](https://huggingface.co/yasirfaizahmed/license-plate-object-detection) (YOLOv8n, Apache-2.0).

Datasets and model weights are not committed; see `.gitignore`.

## License

MIT, see [LICENSE](LICENSE).
