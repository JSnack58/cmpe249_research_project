# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## prov_h1_ycbcr_cb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.192 [0.173, 0.213] | 0.776 [0.754, 0.793] | 0.776 to 0.776 | 0.958 [0.949, 0.967] | 0.138 [0.123, 0.154] |
| yolov8s (retrained) | 1,2 | 0.966 [0.960, 0.973] | 0.918 [0.908, 0.927] | 0.048 [0.041, 0.056] | 0.046 to 0.049 | 0.960 [0.952, 0.967] | 0.881 [0.866, 0.893] |
| yolov10l | 0,1,2 | 0.966 [0.960, 0.972] | 0.940 [0.933, 0.948] | 0.025 [0.021, 0.030] | 0.020 to 0.029 | 0.962 [0.954, 0.970] | 0.921 [0.911, 0.932] |
| rtdetr-l | 0,1,2 | 0.983 [0.979, 0.986] | 0.981 [0.978, 0.985] | 0.001 [-0.001, 0.003] | 0.001 to 0.002 | 0.982 [0.977, 0.987] | 0.981 [0.976, 0.986] |

**H1 not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.024 [0.020, 0.030] mAP50 (needs >= 0.35 with the CI above 0).

Attack success on the 1669 plates every target found when clean: yolov8s (white-box) 0.851, yolov8s (retrained) 0.067, yolov10l 0.029, rtdetr-l 0.002.

## prov_h1_rgb8

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.111 [0.097, 0.131] | 0.857 [0.836, 0.870] | 0.857 to 0.857 | 0.958 [0.949, 0.967] | 0.081 [0.069, 0.094] |
| yolov8s (retrained) | 1,2 | 0.966 [0.960, 0.973] | 0.897 [0.886, 0.910] | 0.069 [0.059, 0.078] | 0.067 to 0.070 | 0.960 [0.952, 0.967] | 0.853 [0.838, 0.867] |
| yolov10l | 0,1,2 | 0.966 [0.960, 0.972] | 0.931 [0.922, 0.940] | 0.035 [0.029, 0.041] | 0.028 to 0.042 | 0.962 [0.954, 0.970] | 0.907 [0.895, 0.918] |
| rtdetr-l | 0,1,2 | 0.983 [0.979, 0.986] | 0.974 [0.969, 0.979] | 0.009 [0.005, 0.012] | 0.007 to 0.010 | 0.982 [0.977, 0.987] | 0.971 [0.964, 0.977] |

**H1 not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.026 [0.020, 0.032] mAP50 (needs >= 0.35 with the CI above 0).

Attack success on the 1669 plates every target found when clean: yolov8s (white-box) 0.913, yolov8s (retrained) 0.091, yolov10l 0.042, rtdetr-l 0.006.
