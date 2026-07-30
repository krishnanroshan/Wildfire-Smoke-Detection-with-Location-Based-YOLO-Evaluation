import argparse
import json
import shutil
from pathlib import Path

import cv2


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")
RESOLUTIONS = ((1920, 1080), (1280, 720), (640, 360), (320, 180))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    source = args.root / "yolo_dataset"
    output = args.root / "resolution_datasets"
    experiment = args.root / "experiment" / "resolutions"
    counts = {}

    for width, height in RESOLUTIONS:
        label = f"{width}x{height}"
        counts[label] = {}
        for location in LOCATIONS:
            src_images = source / "images" / location
            src_labels = source / "labels" / location
            dst_images = output / label / "images" / location
            dst_labels = output / label / "labels" / location
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)
            images = sorted(src_images.glob("*.jpg"))
            paths = []
            for idx, image_path in enumerate(images, 1):
                out_image = dst_images / image_path.name
                if not out_image.exists():
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Could not read {image_path}")
                    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(out_image), resized, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        raise ValueError(f"Could not write {out_image}")
                out_label = dst_labels / f"{image_path.stem}.txt"
                if not out_label.exists():
                    shutil.copyfile(src_labels / f"{image_path.stem}.txt", out_label)
                paths.append(str(out_image))
                if idx % 250 == 0:
                    print(f"{label} {location}: {idx}/{len(images)}", flush=True)
            fold = experiment / label / f"holdout_{location.lower()}"
            fold.mkdir(parents=True, exist_ok=True)
            test_txt = fold / "test.txt"
            test_txt.write_text("\n".join(paths) + "\n", encoding="utf-8")
            yaml = (
                f"path: {(output / label).as_posix()}\n"
                f"train: {test_txt.as_posix()}\n"
                f"val: {test_txt.as_posix()}\n"
                f"test: {test_txt.as_posix()}\n"
                "names:\n  0: smoke\n"
            )
            (fold / "data.yaml").write_text(yaml, encoding="utf-8")
            counts[label][location] = len(paths)
    (experiment / "resolution_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
