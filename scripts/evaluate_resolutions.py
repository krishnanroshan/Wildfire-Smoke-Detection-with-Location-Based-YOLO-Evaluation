import argparse
import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")
RESOLUTIONS = ((1920, 1080, 1), (1280, 720, 2), (640, 360, 8), (320, 180, 16))
FIELDS = ("location", "resolution", "width", "height", "images", "precision", "recall", "map50", "map50_95", "minutes")


def load_existing(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "experiment" / "resolution_metrics.csv"
    rows = load_existing(output)
    done = {(r["location"], r["resolution"]) for r in rows}

    for width, height, batch in RESOLUTIONS:
        resolution = f"{width}x{height}"
        for location in LOCATIONS:
            if (location, resolution) in done:
                print(f"Skipping {location} {resolution}", flush=True)
                continue
            name = f"holdout_{location.lower()}"
            weights = args.root / "runs" / "training" / name / "weights" / "best.pt"
            data = args.root / "experiment" / "resolutions" / resolution / name / "data.yaml"
            test_txt = data.parent / "test.txt"
            image_count = len(test_txt.read_text(encoding="utf-8").splitlines())
            model = YOLO(str(weights))
            started = time.time()
            metrics = model.val(
                data=str(data),
                split="test",
                imgsz=(height, width),
                batch=batch,
                device=0,
                workers=4,
                conf=0.001,
                iou=0.6,
                max_det=300,
                plots=False,
                save_json=False,
                project=str(args.root / "runs" / "evaluation" / resolution),
                name=name,
                exist_ok=True,
                verbose=True,
            )
            row = {
                "location": location,
                "resolution": resolution,
                "width": width,
                "height": height,
                "images": image_count,
                "precision": f"{float(metrics.box.mp):.8f}",
                "recall": f"{float(metrics.box.mr):.8f}",
                "map50": f"{float(metrics.box.map50):.8f}",
                "map50_95": f"{float(metrics.box.map):.8f}",
                "minutes": f"{(time.time() - started) / 60:.3f}",
            }
            rows.append(row)
            save(output, rows)
            print(f"RESULT {row}", flush=True)
            del model, metrics
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
