# H1-direct: each model attacked with its own weights

mAP50 drops are means over seeds 0, 1, 2, each seed's images attacked with that seed's weights; 95% bootstrap CIs over images, one resampling shared by every run.

- yolov10l (CNN control): 0.466 [0.446, 0.485]
- rtdetr-l (Transformer): 0.381 [0.362, 0.399]

**H1-direct not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.085 [0.069, 0.101] mAP50 (needs >= 0.35 with the CI above 0).

Attack success (share of plates missed when attacked, among plates every target found when clean, on the attacked model itself):

- yolov10l_s0: 0.500 of 1669 plates
- yolov10l_s1: 0.481 of 1669 plates
- yolov10l_s2: 0.445 of 1669 plates
- rtdetr-l_s0: 0.499 of 1669 plates
- rtdetr-l_s1: 0.325 of 1669 plates
- rtdetr-l_s2: 0.385 of 1669 plates

## Transfer (by-product, not part of the verdict)

Rows: the attacked model and seed. Columns: each target's mAP50 drop on those images.

| Attacked | yolov8s_s0 | yolov8s_s1 | yolov8s_s2 | yolov10l_s0 | yolov10l_s1 | yolov10l_s2 | rtdetr-l_s0 | rtdetr-l_s1 | rtdetr-l_s2 |
|---|---|---|---|---|---|---|---|---|---|
| rtdetr-l_s0 | 0.007 | 0.007 | 0.004 | 0.008 | 0.007 | 0.008 | 0.424 | 0.031 | 0.053 |
| rtdetr-l_s1 | 0.004 | 0.007 | 0.004 | 0.008 | 0.007 | 0.009 | 0.030 | 0.329 | 0.031 |
| rtdetr-l_s2 | 0.006 | 0.006 | 0.005 | 0.008 | 0.007 | 0.010 | 0.043 | 0.035 | 0.391 |
| yolov10l_s0 | 0.018 | 0.018 | 0.013 | 0.483 | 0.029 | 0.036 | 0.004 | 0.004 | 0.003 |
| yolov10l_s1 | 0.016 | 0.021 | 0.014 | 0.034 | 0.478 | 0.028 | 0.003 | 0.001 | 0.001 |
| yolov10l_s2 | 0.018 | 0.019 | 0.014 | 0.038 | 0.029 | 0.437 | 0.002 | 0.002 | 0.002 |
