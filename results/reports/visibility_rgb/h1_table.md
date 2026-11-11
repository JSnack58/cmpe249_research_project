# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## visibility_rgb_rgb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.107 [0.073, 0.154] | 0.864 [0.817, 0.901] | 0.864 to 0.864 | 0.961 [0.938, 0.981] | 0.077 [0.048, 0.111] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## visibility_rgb_rgb10

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.058 [0.033, 0.091] | 0.913 [0.879, 0.942] | 0.913 to 0.913 | 0.961 [0.938, 0.981] | 0.039 [0.019, 0.064] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## visibility_rgb_rgb12

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.029 [0.014, 0.050] | 0.942 [0.917, 0.963] | 0.942 to 0.942 | 0.961 [0.938, 0.981] | 0.019 [0.006, 0.036] |

No H1 verdict: this run lacks the CNN control or the Transformer.
