import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")
RESOLUTIONS = ("1920x1080", "1280x720", "640x360", "320x180")


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    return float(row[key])


def pct(value):
    return f"{100 * value:.1f}%"


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    analysis = root / "analysis"
    experiment = root / "experiment"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    audit = json.loads((analysis / "dataset_audit.json").read_text(encoding="utf-8"))
    splits = json.loads((experiment / "split_summary.json").read_text(encoding="utf-8"))
    metrics = load_csv(experiment / "resolution_metrics.csv")
    errors = load_csv(experiment / "error_summary_320x180.csv")
    lookup = {(row["location"], row["resolution"]): row for row in metrics}

    aggregate = []
    for resolution in RESOLUTIONS:
        rows = [lookup[(location, resolution)] for location in LOCATIONS]
        aggregate.append(
            {
                "resolution": resolution,
                "macro_precision": np.mean([as_float(r, "precision") for r in rows]),
                "macro_recall": np.mean([as_float(r, "recall") for r in rows]),
                "macro_map50": np.mean([as_float(r, "map50") for r in rows]),
                "macro_map50_95": np.mean([as_float(r, "map50_95") for r in rows]),
                "worst_location_recall": min(as_float(r, "recall") for r in rows),
            }
        )
    write_csv(
        experiment / "resolution_aggregate.csv",
        [
            {key: (f"{value:.8f}" if isinstance(value, float) else value) for key, value in row.items()}
            for row in aggregate
        ],
        ("resolution", "macro_precision", "macro_recall", "macro_map50", "macro_map50_95", "worst_location_recall"),
    )

    location_mean = []
    for location in LOCATIONS:
        rows = [lookup[(location, resolution)] for resolution in RESOLUTIONS]
        location_mean.append(
            {
                "location": location,
                "mean_map50": float(np.mean([as_float(r, "map50") for r in rows])),
                "mean_map50_95": float(np.mean([as_float(r, "map50_95") for r in rows])),
            }
        )
    hardest = min(location_mean, key=lambda row: row["mean_map50_95"])["location"]
    reliable = [
        row
        for row in aggregate
        if row["macro_recall"] >= 0.80 and row["macro_map50"] >= 0.80 and row["worst_location_recall"] >= 0.70
    ]
    lowest_reliable = reliable[-1]["resolution"] if reliable else "None"

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})
    colors = ("#0ea5e9", "#f59e0b", "#10b981", "#8b5cf6")

    baseline = "640x360"
    baseline_rows = [lookup[(location, baseline)] for location in LOCATIONS]
    measures = (("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP@0.50"), ("map50_95", "mAP@0.50–0.95"))
    x = np.arange(len(LOCATIONS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for index, (key, label) in enumerate(measures):
        values = [as_float(row, key) for row in baseline_rows]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=colors[index])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2, fontsize=8, rotation=90)
    ax.set_title("Held-out location performance at 640×360")
    ax.set_ylabel("Score")
    ax.set_xticks(x, LOCATIONS)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.19), frameon=False)
    fig.savefig(report_dir / "accuracy_by_location.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, (("recall", "Recall"), ("map50", "mAP@0.50"))):
        for location, color in zip(LOCATIONS, colors):
            values = [as_float(lookup[(location, resolution)], key) for resolution in RESOLUTIONS]
            ax.plot(RESOLUTIONS, values, marker="o", linewidth=2, label=location, color=color)
        macro_values = [next(row[key.replace("map50", "macro_map50") if key == "map50" else "macro_recall"] for row in aggregate if row["resolution"] == resolution) for resolution in RESOLUTIONS]
        ax.plot(RESOLUTIONS, macro_values, marker="D", linewidth=3, linestyle="--", color="#111827", label="Macro mean")
        ax.axhline(0.8, color="#6b7280", linestyle=":", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Exact source resolution")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.suptitle("Cross-location smoke detection across image resolutions", fontsize=14)
    fig.savefig(report_dir / "accuracy_by_resolution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    error_lookup = {row["location"]: row for row in errors}
    report = []
    report.append("# Wildfire smoke detection benchmark\n")
    report.append("## Dataset inventory\n")
    report.append(f"- {audit['total_images']:,} labeled UAV images; videos excluded.")
    report.append(f"- {audit['smoke_images']:,} smoke images, {audit['no_smoke_images']:,} no-smoke images, and {audit['smoke_boxes']:,} smoke boxes.")
    report.append("- One YOLO detection class: `0 smoke`; each `.txt` row stores normalized `class x_center y_center width height`.")
    report.append("- Source folders: one image folder and one label folder for each of Evo, Heinola, Karkkila, and Ruokolahti, plus `Empty-Images` and `Empty-Labels`.\n")
    report.append("| Location | Images | Smoke | No smoke | Boxes |")
    report.append("|---|---:|---:|---:|---:|")
    for location in LOCATIONS:
        row = audit["per_location"][location]
        report.append(f"| {location} | {row['images']:,} | {row['smoke_images']:,} | {row['no_smoke_images']:,} | {row['smoke_boxes']:,} |")

    report.append("\n## Training protocol\n")
    report.append("YOLO11n (2.58M fused parameters), initialized from pretrained weights and fine-tuned for 15 epochs at 640-pixel input, batch 8, automatic mixed precision, deterministic seed 20260728. Each fold held out one complete location for testing. A stratified 10% validation subset was drawn only from the other three locations.\n")
    report.append("| Held-out test location | Train | Validation | Test |")
    report.append("|---|---:|---:|---:|")
    for location in LOCATIONS:
        row = splits[location]
        report.append(f"| {location} | {row['train']:,} | {row['validation']:,} | {row['test']:,} |")

    report.append("\n## Held-out results at 640×360\n")
    report.append("| Location | Precision | Recall | mAP@0.50 | mAP@0.50–0.95 |")
    report.append("|---|---:|---:|---:|---:|")
    for location in LOCATIONS:
        row = lookup[(location, baseline)]
        report.append(f"| {location} | {pct(as_float(row, 'precision'))} | {pct(as_float(row, 'recall'))} | {pct(as_float(row, 'map50'))} | {pct(as_float(row, 'map50_95'))} |")
    report.append(f"\nHardest location: **{hardest}**, based on the lowest mean mAP@0.50–0.95 across all four tested resolutions; it is also lowest on mAP at the 640×360 operating point.")

    report.append("\n## Resolution comparison (macro average over four held-out folds)\n")
    report.append("| Resolution | Precision | Recall | mAP@0.50 | mAP@0.50–0.95 | Worst-site recall |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for row in aggregate:
        report.append(f"| {row['resolution']} | {pct(row['macro_precision'])} | {pct(row['macro_recall'])} | {pct(row['macro_map50'])} | {pct(row['macro_map50_95'])} | {pct(row['worst_location_recall'])} |")
    report.append("\nOperational reliability rule: macro recall ≥80%, macro mAP@0.50 ≥80%, and every held-out location recall ≥70%.")
    report.append(f"Lowest tested reliable resolution: **{lowest_reliable}**. Bounding-box localization remains moderate (see mAP@0.50–0.95), so this supports detection/alerting more strongly than precise plume outlining.")
    report.append("The non-monotonic result is expected for this particular setup: the network was trained at 640-pixel scale, and Ultralytics validation used the larger image dimension as the square inference canvas. Feeding 1920-pixel inputs directly creates a severe scale mismatch; it is not evidence that high-resolution source imagery is intrinsically worse.")

    report.append("\n## Incorrect detections at 320×180\n")
    report.append("Audit threshold: confidence ≥0.25; IoU ≥0.50 for a correct match.\n")
    report.append("| Location | Images with FP box | FP boxes | Missed boxes | Labeled no-smoke images | No-smoke images flagged |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for location in LOCATIONS:
        row = error_lookup[location]
        report.append(f"| {location} | {int(row['fp_images']):,} | {int(row['fp_boxes']):,} | {int(row['fn_boxes']):,} | {int(row['background_images']):,} | {int(row['background_fp_images']):,} |")
    report.append("\nSeven labeled no-smoke images were flagged. Visual review showed six contain obvious real smoke despite empty label files (annotation omissions). The only genuine environmental false alarm was `heinola_DJI_0048_frame277.jpg`, where a broad box covered bright, slightly hazy sky above treetops. No clear cloud, fog, or water-reflection false alarm was observed at this threshold. Many unmatched boxes in smoke-positive frames were duplicate or oversized plume boxes, so they are localization errors rather than confusion with a different object.\n")
    report.append("## Files\n")
    report.append("- `analysis/sample_smoke_boxes.jpg`: ground-truth examples.")
    report.append("- `analysis/false_positives_320x180/`: annotated error images and montages.")
    report.append("- `experiment/resolution_metrics.csv`: all 16 held-out evaluations.")
    report.append("- `experiment/errors_320x180.csv`: per-image error audit.")
    report.append("- `report/accuracy_by_location.png` and `report/accuracy_by_resolution.png`: comparison graphs.")
    report.append("- `runs/training/holdout_*/weights/best.pt`: the four trained model checkpoints.")
    (report_dir / "wildfire_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    key_results = {
        "hardest_location": hardest,
        "lowest_reliable_resolution": lowest_reliable,
        "reliability_rule": "macro recall >= 0.80, macro mAP50 >= 0.80, worst-location recall >= 0.70",
        "aggregate_by_resolution": aggregate,
        "mean_by_location": location_mean,
    }
    (report_dir / "key_results.json").write_text(json.dumps(key_results, indent=2), encoding="utf-8")
    print(report_dir / "wildfire_smoke_report.md")
    print(report_dir / "accuracy_by_location.png")
    print(report_dir / "accuracy_by_resolution.png")
    print(json.dumps(key_results, indent=2))


if __name__ == "__main__":
    main()
