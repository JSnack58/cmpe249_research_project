# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## transfer_dev_ens

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.956, 0.978] | 0.569 [0.533, 0.606] | 0.399 [0.364, 0.432] | 0.399 to 0.399 | 0.967 [0.952, 0.979] | 0.449 [0.410, 0.485] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.960, 0.981] | 0.644 [0.609, 0.679] | 0.327 [0.295, 0.360] | 0.327 to 0.327 | 0.970 [0.955, 0.982] | 0.562 [0.524, 0.601] |
| yolov8l s0 (white-box) | 0 | 0.961 [0.947, 0.973] | 0.608 [0.572, 0.643] | 0.353 [0.320, 0.385] | 0.353 to 0.353 | 0.962 [0.945, 0.976] | 0.533 [0.496, 0.570] |
| yolov8s (retrained) | 1,2 | 0.970 [0.959, 0.980] | 0.917 [0.900, 0.933] | 0.053 [0.040, 0.065] | 0.053 to 0.054 | 0.970 [0.957, 0.981] | 0.881 [0.859, 0.901] |
| yolov10l | 0,1,2 | 0.965 [0.952, 0.976] | 0.928 [0.911, 0.944] | 0.038 [0.028, 0.047] | 0.032 to 0.044 | 0.968 [0.955, 0.980] | 0.912 [0.892, 0.930] |

**Transfer gate not passed.** Drop(yolov10l) = 0.038 [0.028, 0.047] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

No H1 verdict: this run lacks the CNN control or the Transformer.

## transfer_dev_ens_di

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.956, 0.978] | 0.674 [0.639, 0.706] | 0.294 [0.262, 0.327] | 0.294 to 0.294 | 0.967 [0.952, 0.979] | 0.584 [0.545, 0.621] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.960, 0.981] | 0.715 [0.683, 0.746] | 0.256 [0.225, 0.287] | 0.256 to 0.256 | 0.970 [0.955, 0.982] | 0.651 [0.614, 0.690] |
| yolov8l s0 (white-box) | 0 | 0.961 [0.947, 0.973] | 0.697 [0.663, 0.730] | 0.263 [0.234, 0.294] | 0.263 to 0.263 | 0.962 [0.945, 0.976] | 0.642 [0.606, 0.678] |
| yolov8s (retrained) | 1,2 | 0.970 [0.959, 0.980] | 0.879 [0.859, 0.899] | 0.092 [0.074, 0.109] | 0.089 to 0.094 | 0.970 [0.957, 0.981] | 0.841 [0.814, 0.864] |
| yolov10l | 0,1,2 | 0.965 [0.952, 0.976] | 0.899 [0.879, 0.917] | 0.066 [0.050, 0.082] | 0.061 to 0.076 | 0.968 [0.955, 0.980] | 0.877 [0.854, 0.898] |

**Transfer gate not passed.** Drop(yolov10l) = 0.066 [0.050, 0.082] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

No H1 verdict: this run lacks the CNN control or the Transformer.

## transfer_dev_ens_mi

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.956, 0.978] | 0.645 [0.611, 0.680] | 0.323 [0.289, 0.355] | 0.323 to 0.323 | 0.967 [0.952, 0.979] | 0.524 [0.484, 0.561] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.960, 0.981] | 0.707 [0.674, 0.738] | 0.264 [0.235, 0.295] | 0.264 to 0.264 | 0.970 [0.955, 0.982] | 0.632 [0.595, 0.667] |
| yolov8l s0 (white-box) | 0 | 0.961 [0.947, 0.973] | 0.677 [0.643, 0.709] | 0.283 [0.254, 0.314] | 0.283 to 0.283 | 0.962 [0.945, 0.976] | 0.629 [0.592, 0.665] |
| yolov8s (retrained) | 1,2 | 0.970 [0.959, 0.980] | 0.887 [0.869, 0.908] | 0.083 [0.065, 0.099] | 0.082 to 0.084 | 0.970 [0.957, 0.981] | 0.851 [0.826, 0.875] |
| yolov10l | 0,1,2 | 0.965 [0.952, 0.976] | 0.905 [0.886, 0.922] | 0.061 [0.044, 0.077] | 0.055 to 0.071 | 0.968 [0.955, 0.980] | 0.887 [0.864, 0.907] |

**Transfer gate not passed.** Drop(yolov10l) = 0.061 [0.044, 0.077] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

No H1 verdict: this run lacks the CNN control or the Transformer.

## transfer_dev_ens_di_mi

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.956, 0.978] | 0.701 [0.668, 0.734] | 0.267 [0.234, 0.298] | 0.267 to 0.267 | 0.967 [0.952, 0.979] | 0.610 [0.571, 0.647] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.960, 0.981] | 0.743 [0.711, 0.774] | 0.228 [0.198, 0.259] | 0.228 to 0.228 | 0.970 [0.955, 0.982] | 0.688 [0.651, 0.723] |
| yolov8l s0 (white-box) | 0 | 0.961 [0.947, 0.973] | 0.716 [0.684, 0.747] | 0.245 [0.215, 0.274] | 0.245 to 0.245 | 0.962 [0.945, 0.976] | 0.669 [0.633, 0.704] |
| yolov8s (retrained) | 1,2 | 0.970 [0.959, 0.980] | 0.859 [0.837, 0.882] | 0.112 [0.091, 0.131] | 0.109 to 0.114 | 0.970 [0.957, 0.981] | 0.832 [0.805, 0.857] |
| yolov10l | 0,1,2 | 0.965 [0.952, 0.976] | 0.880 [0.859, 0.901] | 0.085 [0.065, 0.104] | 0.075 to 0.097 | 0.968 [0.955, 0.980] | 0.855 [0.830, 0.879] |

**Transfer gate not passed.** Drop(yolov10l) = 0.085 [0.065, 0.104] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

No H1 verdict: this run lacks the CNN control or the Transformer.

## transfer_dev_ens_di_mi_100

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.956, 0.978] | 0.676 [0.641, 0.710] | 0.292 [0.258, 0.324] | 0.292 to 0.292 | 0.967 [0.952, 0.979] | 0.580 [0.541, 0.619] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.960, 0.981] | 0.730 [0.696, 0.760] | 0.241 [0.211, 0.273] | 0.241 to 0.241 | 0.970 [0.955, 0.982] | 0.674 [0.638, 0.708] |
| yolov8l s0 (white-box) | 0 | 0.961 [0.947, 0.973] | 0.699 [0.665, 0.731] | 0.262 [0.232, 0.292] | 0.262 to 0.262 | 0.962 [0.945, 0.976] | 0.656 [0.620, 0.691] |
| yolov8s (retrained) | 1,2 | 0.970 [0.959, 0.980] | 0.853 [0.829, 0.876] | 0.118 [0.096, 0.139] | 0.118 to 0.118 | 0.970 [0.957, 0.981] | 0.822 [0.794, 0.848] |
| yolov10l | 0,1,2 | 0.965 [0.952, 0.976] | 0.874 [0.851, 0.896] | 0.091 [0.070, 0.109] | 0.079 to 0.102 | 0.968 [0.955, 0.980] | 0.848 [0.822, 0.874] |

**Transfer gate not passed.** Drop(yolov10l) = 0.091 [0.070, 0.109] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

No H1 verdict: this run lacks the CNN control or the Transformer.
