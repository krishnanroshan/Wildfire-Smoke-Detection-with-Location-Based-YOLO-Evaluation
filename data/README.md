# Dataset setup

## Official source

- Dataset page: https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898
- DOI: https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9
- Dataset article: https://doi.org/10.1038/s41597-025-05634-0
- License: CC BY 4.0

This benchmark uses only **Boreal-Forest-Fire-Subset-A**, the bounding-box annotated UAV images. The video and segmentation subsets are not used.

## Verified contents

| Location | Images | Smoke images | Empty-label images | Smoke boxes |
|---|---:|---:|---:|---:|
| Evo | 931 | 930 | 1 | 957 |
| Heinola | 1,123 | 906 | 217 | 917 |
| Karkkila | 1,133 | 1,095 | 38 | 1,104 |
| Ruokolahti | 1,767 | 1,762 | 5 | 1,884 |
| Total | 4,954 | 4,693 | 261 | 4,862 |

Image dimensions in the official subset:

- 3,728 images at 4096x2160
- 1,133 images at 3840x2160
- 93 images at 1920x1080

## YOLO annotations

The repository includes all 4,954 text annotations in `data/labels.zip`. Extract it to create `labels/<Location>/`. Each row has:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. The single class is `0 = smoke`. Empty files represent nominally smoke-free images.

## Expected local image layout

Raw image binaries are not committed. After downloading Subset A, arrange or symlink the images as:

```text
<dataset-root>/
  images/
    Evo/
    Heinola/
    Karkkila/
    Ruokolahti/
  labels/
    Evo/
    Heinola/
    Karkkila/
    Ruokolahti/
```

The exact train, validation, and test membership is stored in `data/manifests/splits/`. Every path is relative to `<dataset-root>`.

## Why images are not in GitHub

The working benchmark directory is approximately 43 GB because it contains the official archives, extracted 4K images, four resized copies, environments, and model runs. GitHub blocks regular Git files larger than 100 MiB and recommends keeping repositories small. The official Fairdata record is the canonical source for the image binaries.

## Known label-quality findings

At confidence 0.25, seven empty-label images triggered the 320x180 detector. Visual review found visible smoke in six of them:

- `evoDJI_0001_frame23.jpg`
- `karkkila_DJI_0003_frame39.jpg`
- `karkkila_DJI_0004_frame132.jpg`
- `ruokolahti_DJI_0087_frame46.jpg`
- `ruokolahti_DJI_0087_frame249.jpg`
- `ruokolahti_DJI_0087_frame259.jpg`

The remaining image, `heinola_DJI_0048_frame277.jpg`, is the clearest genuine false alarm: bright, slightly hazy sky above treetops.

