# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## loss_topk_rgb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.107 [0.073, 0.154] | 0.864 [0.818, 0.901] | 0.864 to 0.864 | 0.961 [0.938, 0.981] | 0.077 [0.048, 0.111] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## loss_hinge_rgb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.534 [0.490, 0.595] | 0.437 [0.377, 0.476] | 0.437 to 0.437 | 0.961 [0.938, 0.981] | 0.080 [0.049, 0.114] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## loss_topk_cb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.178 [0.133, 0.229] | 0.793 [0.743, 0.838] | 0.793 to 0.793 | 0.961 [0.938, 0.981] | 0.135 [0.099, 0.176] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## loss_hinge_cb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.971 [0.953, 0.985] | 0.525 [0.476, 0.594] | 0.446 [0.382, 0.493] | 0.446 to 0.446 | 0.961 [0.938, 0.981] | 0.119 [0.084, 0.158] |

No H1 verdict: this run lacks the CNN control or the Transformer.
