# H1: clean vs. attacked plate detection

Values are seed means with 95% paired bootstrap CIs over images.
Drop range is the smallest and largest single-seed mAP50 drop.

## m1_h1_main

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.072 [0.059, 0.088] | 0.896 [0.879, 0.909] | 0.896 to 0.896 | 0.958 [0.949, 0.967] | 0.054 [0.045, 0.065] |
| yolov8s (retrained) | 1,2 | 0.966 [0.960, 0.973] | 0.900 [0.889, 0.912] | 0.065 [0.056, 0.075] | 0.065 to 0.065 | 0.960 [0.952, 0.967] | 0.857 [0.842, 0.871] |
| yolov10l | 0,1,2 | 0.966 [0.960, 0.972] | 0.931 [0.922, 0.940] | 0.035 [0.029, 0.041] | 0.026 to 0.043 | 0.962 [0.954, 0.970] | 0.908 [0.896, 0.920] |
| rtdetr-l | 0,1,2 | 0.983 [0.979, 0.986] | 0.981 [0.977, 0.985] | 0.001 [-0.001, 0.003] | 0.001 to 0.002 | 0.982 [0.977, 0.987] | 0.981 [0.976, 0.986] |

**H1 not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.034 [0.028, 0.040] mAP50 (needs >= 0.35 with the CI above 0).

Attack success on the 1669 plates every target found when clean: yolov8s (white-box) 0.942, yolov8s (retrained) 0.089, yolov10l 0.041, rtdetr-l 0.002.

## m1_h1_rgb

| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range | Clean recall | Attacked recall |
|---|---|---|---|---|---|---|---|
| yolov8s (white-box) | 0 | 0.968 [0.960, 0.974] | 0.111 [0.097, 0.131] | 0.857 [0.836, 0.870] | 0.857 to 0.857 | 0.958 [0.949, 0.967] | 0.081 [0.069, 0.094] |
| yolov8s (retrained) | 1,2 | 0.966 [0.960, 0.973] | 0.897 [0.886, 0.910] | 0.068 [0.058, 0.078] | 0.067 to 0.070 | 0.960 [0.952, 0.967] | 0.854 [0.839, 0.868] |
| yolov10l | 0,1,2 | 0.966 [0.960, 0.972] | 0.931 [0.922, 0.940] | 0.034 [0.029, 0.041] | 0.028 to 0.041 | 0.962 [0.954, 0.970] | 0.908 [0.896, 0.919] |
| rtdetr-l | 0,1,2 | 0.983 [0.979, 0.986] | 0.974 [0.969, 0.979] | 0.009 [0.006, 0.012] | 0.007 to 0.010 | 0.982 [0.977, 0.987] | 0.971 [0.964, 0.977] |

**H1 not supported.** Drop(yolov10l) - drop(rtdetr-l) = 0.026 [0.020, 0.032] mAP50 (needs >= 0.35 with the CI above 0).

Attack success on the 1669 plates every target found when clean: yolov8s (white-box) 0.913, yolov8s (retrained) 0.091, yolov10l 0.041, rtdetr-l 0.006.
