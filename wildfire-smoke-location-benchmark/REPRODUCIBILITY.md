# Reproducing the benchmark

## Recorded environment

- Windows
- NVIDIA GeForce RTX 4060, 8 GB VRAM
- Python 3.12
- PyTorch 2.8.0 with CUDA 12.9 build
- torchvision 0.23.0
- Ultralytics 8.4.106
- OpenCV 5.0.0
- Deterministic random seed: `20260728`

Install the Python packages in `requirements.txt`. Install the PyTorch build appropriate for your CUDA or CPU environment from the official PyTorch instructions.

## 1. Audit the official subset

Point `--source` at the extracted `Boreal-Forest-Fire-Subset-A` directory containing the location image/label folders and `Empty-Images`/`Empty-Labels`.

```powershell
python scripts/audit_dataset.py `
  --source D:\path\to\Boreal-Forest-Fire-Subset-A `
  --output D:\benchmark\analysis
```

This verifies paired files, class IDs, normalized coordinates, image decodability, counts, and dimensions. It also produces the ground-truth montage.

## 2. Prepare location folds

The original run used a stratified 10% validation split drawn only from the three training locations. The fourth location was never used for training or validation.

```powershell
python scripts/prepare_yolo.py `
  --index D:\benchmark\analysis\image_index.csv `
  --dataset D:\benchmark\yolo_dataset `
  --output D:\benchmark\experiment `
  --seed 20260728
```

The exact membership from the completed experiment is preserved under `data/manifests/splits/`.

| Held-out location | Train | Validation | Test |
|---|---:|---:|---:|
| Evo | 3,620 | 403 | 931 |
| Heinola | 3,448 | 383 | 1,123 |
| Karkkila | 3,439 | 382 | 1,133 |
| Ruokolahti | 2,867 | 320 | 1,767 |

## 3. Train YOLO11n

Place `yolo11n.pt` at the benchmark root or adjust `train_folds.py`. The completed run used 15 epochs, a 640-pixel canvas, batch 8, AMP, cosine learning-rate scheduling, and close-mosaic for the final three epochs.

```powershell
python scripts/train_folds.py --root D:\benchmark --epochs 15 --batch 8 --workers 4
```

The four completed best checkpoints are included under `models/`. Sanitized training arguments, epoch histories, curves, and confusion matrices are under `results/training/`.

## 4. Create exact-resolution datasets

```powershell
python scripts/prepare_resolutions.py --root D:\benchmark --quality 88
```

The completed experiment created exact 1920x1080, 1280x720, 640x360, and 320x180 JPEG copies while retaining normalized labels.

## 5. Evaluate all 16 combinations

```powershell
python scripts/evaluate_resolutions.py --root D:\benchmark
```

Ultralytics validation used the larger image dimension as its square inference canvas. This matters when interpreting the high-resolution degradation: inference scale changed along with stored image resolution.

## 6. Audit errors

```powershell
python scripts/analyze_errors.py `
  --root D:\benchmark `
  --resolution 320x180 `
  --confidence 0.25 `
  --iou 0.5
```

Predictions were matched greedily to unmatched ground-truth boxes at IoU 0.50. Unmatched predictions were counted as false-positive boxes; unmatched labels were counted as false-negative boxes.

## Reliability rule

The lowest-resolution conclusion uses a declared operational rule:

- Macro recall at least 0.80
- Macro mAP@0.50 at least 0.80
- Recall at least 0.70 in every held-out location

Both 640x360 and 320x180 passed; therefore 320x180 is the lowest tested reliable setting. This criterion emphasizes smoke alerting. The lower mAP@0.50-0.95 values show that precise plume localization remains harder.

