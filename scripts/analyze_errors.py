import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")
FIELDS = (
    "location",
    "image",
    "gt_count",
    "pred_count",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "max_false_positive_confidence",
    "background_false_alarm",
)


def load_ground_truth(label_path: Path, width: int, height: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def match_predictions(pred_boxes, pred_conf, gt_boxes, threshold=0.5):
    order = np.argsort(-pred_conf)
    used_gt = set()
    matched = set()
    for pred_index in order:
        ious = iou_one_to_many(pred_boxes[pred_index], gt_boxes)
        if not len(ious):
            continue
        gt_index = int(np.argmax(ious))
        if ious[gt_index] >= threshold and gt_index not in used_gt:
            used_gt.add(gt_index)
            matched.add(int(pred_index))
    fp_indices = [i for i in range(len(pred_boxes)) if i not in matched]
    return matched, fp_indices, len(gt_boxes) - len(used_gt)


def draw_annotated(record, output_path: Path, highres_root: Path):
    source = highres_root / record["location"] / Path(record["image"]).name
    image = cv2.imread(str(source))
    if image is None:
        return None
    height, width = image.shape[:2]
    sx = width / record["source_width"]
    sy = height / record["source_height"]

    for box in record["gt_boxes"]:
        x1, y1, x2, y2 = [int(v * s) for v, s in zip(box, (sx, sy, sx, sy))]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 4)
        cv2.putText(image, "GT smoke", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)

    for index in record["fp_indices"]:
        box = record["pred_boxes"][index]
        conf = record["pred_conf"][index]
        x1, y1, x2, y2 = [int(v * s) for v, s in zip(box, (sx, sy, sx, sy))]
        cv2.rectangle(image, (x1, y1), (x2, y2), (20, 20, 235), 5)
        cv2.putText(image, f"FP smoke {conf:.2f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 235), 2)

    title = f"{record['location']} | {Path(record['image']).name} | red=FP, yellow=GT"
    cv2.rectangle(image, (0, 0), (width, 48), (12, 18, 28), -1)
    cv2.putText(image, title, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def make_montage(paths, output_path: Path, columns=2, cell=(960, 540)):
    if not paths:
        return
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (cell[0] * columns, cell[1] * rows), (21, 28, 38))
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail(cell, Image.Resampling.LANCZOS)
        x = (index % columns) * cell[0] + (cell[0] - image.width) // 2
        y = (index // columns) * cell[1] + (cell[1] - image.height) // 2
        canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resolution", default="320x180")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))
    dataset = args.root / "resolution_datasets" / args.resolution
    highres = args.root / "resolution_datasets" / "1920x1080" / "images"
    output_dir = args.root / "analysis" / f"false_positives_{args.resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for location in LOCATIONS:
        weights = args.root / "runs" / "training" / f"holdout_{location.lower()}" / "weights" / "best.pt"
        image_dir = dataset / "images" / location
        label_dir = dataset / "labels" / location
        model = YOLO(str(weights))
        results = model.predict(
            source=str(image_dir),
            imgsz=max(width, height),
            conf=args.confidence,
            iou=0.6,
            max_det=300,
            device=0,
            batch=32,
            stream=True,
            verbose=False,
        )
        for result in results:
            image_path = Path(result.path)
            shape_height, shape_width = result.orig_shape
            gt_boxes = load_ground_truth(label_dir / f"{image_path.stem}.txt", shape_width, shape_height)
            if result.boxes is None or len(result.boxes) == 0:
                pred_boxes = np.empty((0, 4), dtype=np.float32)
                pred_conf = np.empty(0, dtype=np.float32)
            else:
                pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                pred_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            matched, fp_indices, fn_count = match_predictions(pred_boxes, pred_conf, gt_boxes, args.iou)
            records.append(
                {
                    "location": location,
                    "image": str(image_path.relative_to(dataset / "images")),
                    "gt_count": len(gt_boxes),
                    "pred_count": len(pred_boxes),
                    "tp_count": len(matched),
                    "fp_count": len(fp_indices),
                    "fn_count": fn_count,
                    "max_fp_conf": max((float(pred_conf[i]) for i in fp_indices), default=0.0),
                    "background_false_alarm": len(gt_boxes) == 0 and len(fp_indices) > 0,
                    "source_width": shape_width,
                    "source_height": shape_height,
                    "gt_boxes": gt_boxes,
                    "pred_boxes": pred_boxes,
                    "pred_conf": pred_conf,
                    "fp_indices": fp_indices,
                }
            )
        del model
        torch.cuda.empty_cache()
        print(f"Analyzed {location}", flush=True)

    csv_path = args.root / "experiment" / f"errors_{args.resolution}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "location": r["location"],
                    "image": r["image"],
                    "gt_count": r["gt_count"],
                    "pred_count": r["pred_count"],
                    "true_positive_count": r["tp_count"],
                    "false_positive_count": r["fp_count"],
                    "false_negative_count": r["fn_count"],
                    "max_false_positive_confidence": f"{r['max_fp_conf']:.6f}",
                    "background_false_alarm": r["background_false_alarm"],
                }
            )

    selected = []
    for location in LOCATIONS:
        candidates = [r for r in records if r["location"] == location and r["fp_count"]]
        backgrounds = sorted(
            [r for r in candidates if r["background_false_alarm"]],
            key=lambda r: r["max_fp_conf"],
            reverse=True,
        )
        smoky = sorted(
            [r for r in candidates if not r["background_false_alarm"]],
            key=lambda r: r["max_fp_conf"],
            reverse=True,
        )
        selected.extend((backgrounds + smoky)[:8])

    annotated_by_location = defaultdict(list)
    top_rows = []
    for rank, record in enumerate(selected, start=1):
        stem = Path(record["image"]).stem
        output = output_dir / record["location"] / f"{rank:02d}_{stem}.jpg"
        if draw_annotated(record, output, highres):
            annotated_by_location[record["location"]].append(output)
        top_rows.append(
            {
                "location": record["location"],
                "image": record["image"],
                "gt_count": record["gt_count"],
                "false_positive_count": record["fp_count"],
                "max_false_positive_confidence": f"{record['max_fp_conf']:.6f}",
                "background_false_alarm": record["background_false_alarm"],
                "annotated_image": str(output),
            }
        )

    top_csv = output_dir / "top_false_positives.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=top_rows[0].keys() if top_rows else [])
        if top_rows:
            writer.writeheader()
            writer.writerows(top_rows)

    for location, paths in annotated_by_location.items():
        make_montage(paths, output_dir / f"{location}_false_positives_montage.jpg")
    make_montage([p for paths in annotated_by_location.values() for p in paths], output_dir / "all_false_positives_montage.jpg", columns=2)

    summary = defaultdict(lambda: {"images": 0, "fp_images": 0, "fp_boxes": 0, "fn_boxes": 0, "background_images": 0, "background_fp_images": 0})
    for r in records:
        s = summary[r["location"]]
        s["images"] += 1
        s["fp_images"] += int(r["fp_count"] > 0)
        s["fp_boxes"] += r["fp_count"]
        s["fn_boxes"] += r["fn_count"]
        s["background_images"] += int(r["gt_count"] == 0)
        s["background_fp_images"] += int(r["background_false_alarm"])
    summary_path = args.root / "experiment" / f"error_summary_{args.resolution}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ("location", "images", "fp_images", "fp_boxes", "fn_boxes", "background_images", "background_fp_images")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for location in LOCATIONS:
            writer.writerow({"location": location, **summary[location]})
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
