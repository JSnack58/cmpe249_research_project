# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## transfer_final_ycbcr

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.960, 0.974] | 0.759 [0.741, 0.779] | 0.209 [0.190, 0.225] | 0.209 to 0.209 | 0.958 [0.949, 0.967] | 0.674 [0.652, 0.696] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.965, 0.978] | 0.779 [0.761, 0.797] | 0.193 [0.177, 0.210] | 0.193 to 0.193 | 0.965 [0.956, 0.974] | 0.712 [0.690, 0.732] |
| yolov8l s0 (white-box) | 0 | 0.966 [0.960, 0.973] | 0.755 [0.737, 0.775] | 0.211 [0.194, 0.228] | 0.211 to 0.211 | 0.964 [0.955, 0.972] | 0.702 [0.681, 0.724] |
| yolov8s (retrained) | 1,2 | 0.966 [0.960, 0.973] | 0.879 [0.866, 0.891] | 0.087 [0.077, 0.099] | 0.084 to 0.089 | 0.960 [0.952, 0.967] | 0.843 [0.827, 0.859] |
| yolov10l | 0,1,2 | 0.966 [0.960, 0.972] | 0.897 [0.886, 0.910] | 0.069 [0.060, 0.078] | 0.063 to 0.075 | 0.962 [0.954, 0.970] | 0.867 [0.853, 0.881] |
| rtdetr-l | 0,1,2 | 0.983 [0.979, 0.986] | 0.976 [0.972, 0.981] | 0.006 [0.004, 0.009] | 0.006 to 0.007 | 0.982 [0.977, 0.987] | 0.973 [0.967, 0.979] |

**Transfer gate not passed.** Drop(yolov10l) = 0.069 [0.060, 0.078] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

**H1 not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.062 [0.054, 0.071] mAP50 (needs >= 0.35 with the CI above 0).

Attack success on the 1661 plates every target found when clean: yolov8s s0 (white-box) 0.265, yolov8n s0 (white-box) 0.225, yolov8l s0 (white-box) 0.236, yolov8s (retrained) 0.091, yolov10l 0.069, rtdetr-l 0.004.

## transfer_final_rgb

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s s0 (white-box) | 0 | 0.968 [0.960, 0.974] | 0.860 [0.845, 0.876] | 0.108 [0.094, 0.119] | 0.108 to 0.108 | 0.958 [0.949, 0.967] | 0.812 [0.794, 0.830] |
| yolov8n s0 (white-box) | 0 | 0.971 [0.965, 0.978] | 0.865 [0.850, 0.880] | 0.106 [0.095, 0.120] | 0.106 to 0.106 | 0.965 [0.956, 0.974] | 0.818 [0.799, 0.835] |
| yolov8l s0 (white-box) | 0 | 0.966 [0.960, 0.973] | 0.865 [0.850, 0.880] | 0.102 [0.089, 0.114] | 0.102 to 0.102 | 0.964 [0.955, 0.972] | 0.828 [0.811, 0.846] |
| yolov8s (retrained) | 1,2 | 0.966 [0.960, 0.973] | 0.921 [0.911, 0.932] | 0.044 [0.037, 0.052] | 0.042 to 0.046 | 0.960 [0.952, 0.967] | 0.900 [0.887, 0.913] |
| yolov10l | 0,1,2 | 0.966 [0.960, 0.972] | 0.932 [0.923, 0.942] | 0.033 [0.028, 0.039] | 0.031 to 0.035 | 0.962 [0.954, 0.970] | 0.917 [0.905, 0.928] |
| rtdetr-l | 0,1,2 | 0.983 [0.979, 0.986] | 0.976 [0.971, 0.981] | 0.007 [0.005, 0.009] | 0.007 to 0.007 | 0.982 [0.977, 0.987] | 0.972 [0.965, 0.978] |

**Transfer gate not passed.** Drop(yolov10l) = 0.033 [0.028, 0.039] mAP50 (needs the whole CI at or above 0.35 for H1 to be testable).

**H1 not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.026 [0.022, 0.033] mAP50 (needs >= 0.35 with the CI above 0).

Attack success on the 1661 plates every target found when clean: yolov8s s0 (white-box) 0.117, yolov8n s0 (white-box) 0.113, yolov8l s0 (white-box) 0.104, yolov8s (retrained) 0.037, yolov10l 0.026, rtdetr-l 0.003.
