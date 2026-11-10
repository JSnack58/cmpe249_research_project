# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## cbcr_rgb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.111 [0.097, 0.131] | 0.857 [0.836, 0.870] | 0.857 to 0.857 | 0.958 [0.949, 0.967] | 0.081 [0.069, 0.094] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## cbcr_cb4

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.472 [0.449, 0.496] | 0.496 [0.473, 0.517] | 0.496 to 0.496 | 0.958 [0.949, 0.967] | 0.376 [0.355, 0.399] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## cbcr_cb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.192 [0.173, 0.213] | 0.776 [0.754, 0.793] | 0.776 to 0.776 | 0.958 [0.949, 0.967] | 0.138 [0.123, 0.154] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## cbcr_cb12

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.072 [0.059, 0.088] | 0.896 [0.879, 0.909] | 0.896 to 0.896 | 0.958 [0.949, 0.967] | 0.054 [0.045, 0.065] |

No H1 verdict: this run lacks the CNN control or the Transformer.

## cbcr_random8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.958 [0.950, 0.965] | 0.010 [0.005, 0.014] | 0.010 to 0.010 | 0.958 [0.949, 0.967] | 0.938 [0.928, 0.950] |

No H1 verdict: this run lacks the CNN control or the Transformer.
