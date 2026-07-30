# Results

## Dataset inventory

- 4,954 labeled UAV images; video files excluded.
- 4,693 images with one or more smoke boxes.
- 261 images with empty label files.
- 4,862 total smoke boxes.
- One YOLO class: `0 = smoke`.

| Location | Images | Smoke | Empty label | Boxes |
|---|---:|---:|---:|---:|
| Evo | 931 | 930 | 1 | 957 |
| Heinola | 1,123 | 906 | 217 | 917 |
| Karkkila | 1,133 | 1,095 | 38 | 1,104 |
| Ruokolahti | 1,767 | 1,762 | 5 | 1,884 |

## Location holdout results at 640x360

Each row is a different YOLO11n model. The named location was excluded entirely from training and internal validation.

| Held-out location | Precision | Recall | mAP@0.50 | mAP@0.50-0.95 |
|---|---:|---:|---:|---:|
| Evo | 0.9417 | 0.9119 | 0.9104 | 0.4429 |
| Heinola | 0.7886 | 0.8138 | 0.7596 | 0.3286 |
| Karkkila | 0.7970 | 0.7165 | 0.8474 | 0.4961 |
| Ruokolahti | 0.9966 | 0.9276 | 0.9304 | 0.7378 |

Heinola was the hardest overall: it had the lowest mAP values at the training-matched 640x360 setting and the lowest mean mAP@0.50-0.95 across all four tested resolutions. Karkkila had the lowest 640x360 recall.

## Resolution results

Macro averages give every held-out location equal weight.

| Resolution | Precision | Recall | mAP@0.50 | mAP@0.50-0.95 | Worst-location recall |
|---|---:|---:|---:|---:|---:|
| 1920x1080 | 0.1706 | 0.1754 | 0.0833 | 0.0189 | 0.0064 |
| 1280x720 | 0.5538 | 0.5671 | 0.5065 | 0.1770 | 0.2818 |
| 640x360 | 0.8810 | 0.8424 | 0.8619 | 0.5013 | 0.7165 |
| 320x180 | 0.8832 | 0.8619 | 0.8710 | 0.5154 | 0.7710 |

The operational reliability rule was macro recall >= 0.80, macro mAP@0.50 >= 0.80, and recall >= 0.70 at every held-out location. Both 640x360 and 320x180 passed, making 320x180 the lowest tested reliable resolution.

This conclusion is specific to alerting/detection. mAP@0.50-0.95 remained near 0.50, indicating only moderate precision when tightly outlining smoke plumes.

## Scale caveat

These tests changed both the stored image resolution and the model's inference canvas. The network was trained at a 640-pixel canvas. Direct 1920-pixel inference therefore created a large object-scale shift and performed poorly. This does not imply that high-resolution source cameras are inferior; a deployment can retain high-resolution capture while resizing frames to the model's trained input scale.

## Error audit at 320x180

Operating threshold: confidence >= 0.25. A prediction required IoU >= 0.50 with an unmatched ground-truth box to count as correct.

| Location | Images with FP | FP boxes | Missed boxes | Empty-label images | Empty-label images flagged |
|---|---:|---:|---:|---:|---:|
| Evo | 59 | 64 | 77 | 1 | 1 |
| Heinola | 252 | 254 | 282 | 217 | 1 |
| Karkkila | 274 | 314 | 130 | 38 | 2 |
| Ruokolahti | 7 | 7 | 143 | 5 | 3 |

Many unmatched predictions in smoke-positive frames were duplicate or oversized plume boxes. These are localization errors, not necessarily smoke/no-smoke classification mistakes.

Seven empty-label images triggered detections. Six visibly contain real smoke and appear to be annotation omissions. The clearest genuine environmental false alarm is `heinola_DJI_0048_frame277.jpg`, where a broad detection covers bright, slightly hazy sky above treetops. No clear fog-only or water-reflection false alarm was observed at the selected threshold.

The full 16-run table, per-image error records, and reviewed examples are included in the adjacent `metrics/` and `figures/` directories.

