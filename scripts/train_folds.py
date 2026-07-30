import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--locations", nargs="*", default=list(LOCATIONS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    project = args.root / "runs" / "training"
    project.mkdir(parents=True, exist_ok=True)
    summary_path = args.root / "experiment" / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    for location in args.locations:
        key = location.lower()
        name = f"holdout_{key}"
        run_dir = project / name
        best = run_dir / "weights" / "best.pt"
        if summary.get(location, {}).get("status") == "complete" and best.exists():
            print(f"Skipping completed fold {location}: {best}", flush=True)
            continue

        data = args.root / "experiment" / "folds" / name / "data.yaml"
        model = YOLO(str(args.root / "yolo11n.pt"))
        started = time.time()
        summary[location] = {"status": "running", "started": started}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        result = model.train(
            data=str(data),
            epochs=args.epochs,
            patience=5,
            imgsz=640,
            batch=args.batch,
            device=0,
            workers=args.workers,
            project=str(project),
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            seed=20260728,
            deterministic=True,
            amp=True,
            cache=False,
            cos_lr=True,
            close_mosaic=3,
            plots=True,
            verbose=True,
        )
        elapsed = time.time() - started
        metrics = {k: float(v) for k, v in getattr(result, "results_dict", {}).items() if isinstance(v, (int, float))}
        summary[location] = {
            "status": "complete",
            "elapsed_minutes": round(elapsed / 60, 2),
            "best_weights": str(best),
            "validation_metrics": metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Completed {location} in {elapsed / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
