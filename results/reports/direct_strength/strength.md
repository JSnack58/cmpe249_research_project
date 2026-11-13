# Strength check: white-box mAP50 drop per run

Passes when the point estimate reaches 0.7 (95% bootstrap CI over images shown).

| Attacked model | Run | mAP50 drop | Passed |
|---|---|---|---|
| rtdetr-l_s0 | direct_strength_rtdetr_s0 | 0.774 [0.740, 0.807] | yes |
| rtdetr-l_s1 | direct_strength_rtdetr_s1 | 0.737 [0.698, 0.768] | yes |
| rtdetr-l_s2 | direct_strength_rtdetr_s2 | 0.810 [0.772, 0.841] | yes |
| yolov10l_s0 | direct_strength_v10l_s0 | 0.892 [0.863, 0.912] | yes |
| yolov10l_s1 | direct_strength_v10l_s1 | 0.860 [0.827, 0.881] | yes |
| yolov10l_s2 | direct_strength_v10l_s2 | 0.747 [0.709, 0.782] | yes |
