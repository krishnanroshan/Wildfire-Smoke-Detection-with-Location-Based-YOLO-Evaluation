import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_label(path: Path):
    boxes = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")
        cls, x, y, w, h = map(float, parts)
        if cls != 0:
            raise ValueError(f"{path}:{line_no}: unexpected class {cls}")
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            raise ValueError(f"{path}:{line_no}: coordinates outside [0,1]")
        boxes.append((int(cls), x, y, w, h))
    return boxes


def image_size(path: Path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        raise ValueError(f"Could not decode image: {path}: {exc}") from exc


def infer_empty_location(name: str):
    lower = name.lower()
    for loc in LOCATIONS:
        if lower.startswith(loc.lower()):
            return loc
    raise ValueError(f"Cannot infer location from empty-image filename: {name}")


def collect_records(source: Path):
    records = []
    validation = {"missing_labels": [], "missing_images": [], "duplicate_stems": []}
    seen = set()

    pairs = [(loc, source / f"{loc}-Images", source / f"{loc}-Labels", "location") for loc in LOCATIONS]
    pairs.append((None, source / "Empty-Images", source / "Empty-Labels", "dedicated_empty"))

    for fixed_loc, image_dir, label_dir, kind in pairs:
        images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        validation["missing_labels"].extend(str(images[s]) for s in sorted(images.keys() - labels.keys()))
        validation["missing_images"].extend(str(labels[s]) for s in sorted(labels.keys() - images.keys()))
        for stem in sorted(images.keys() & labels.keys()):
            loc = fixed_loc or infer_empty_location(images[stem].name)
            key = (loc, stem)
            if key in seen:
                validation["duplicate_stems"].append(f"{loc}:{stem}")
            seen.add(key)
            boxes = read_label(labels[stem])
            w, h = image_size(images[stem])
            records.append({
                "location": loc,
                "stem": stem,
                "image_path": str(images[stem]),
                "label_path": str(labels[stem]),
                "source_kind": kind,
                "width": w,
                "height": h,
                "smoke": bool(boxes),
                "box_count": len(boxes),
                "boxes": boxes,
            })
    if any(validation.values()):
        raise ValueError(json.dumps(validation, indent=2))
    return records


def draw_montage(records, output: Path):
    selected = []
    for loc in LOCATIONS:
        smoke = [r for r in records if r["location"] == loc and r["smoke"]]
        if smoke:
            selected.extend([smoke[len(smoke) // 3], smoke[(2 * len(smoke)) // 3]])
    tiles = []
    for r in selected:
        image = Image.open(r["image_path"]).convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)
        for _, xc, yc, bw, bh in r["boxes"]:
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w - 1, int((xc + bw / 2) * w))
            y2 = min(h - 1, int((yc + bh / 2) * h))
            draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=max(4, w // 900))
        tile = image.resize((640, 360), Image.Resampling.LANCZOS)
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rectangle((0, 0, 640, 34), fill=(0, 0, 0))
        tile_draw.text((10, 10), f"{r['location']} | {Path(r['image_path']).name}", fill=(255, 255, 255))
        tiles.append(tile)
    montage = Image.new("RGB", (1280, 360 * ((len(tiles) + 1) // 2)), (0, 0, 0))
    for i, tile in enumerate(tiles):
        montage.paste(tile, ((i % 2) * 640, (i // 2) * 360))
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.source)
    per_location = {}
    for loc in LOCATIONS:
        rows = [r for r in records if r["location"] == loc]
        per_location[loc] = {
            "images": len(rows),
            "smoke_images": sum(r["smoke"] for r in rows),
            "no_smoke_images": sum(not r["smoke"] for r in rows),
            "smoke_boxes": sum(r["box_count"] for r in rows),
            "dedicated_empty_images": sum(r["source_kind"] == "dedicated_empty" for r in rows),
        }

    dimensions = Counter(f"{r['width']}x{r['height']}" for r in records)
    summary = {
        "total_images": len(records),
        "smoke_images": sum(r["smoke"] for r in records),
        "no_smoke_images": sum(not r["smoke"] for r in records),
        "smoke_boxes": sum(r["box_count"] for r in records),
        "classes": {"0": "smoke"},
        "per_location": per_location,
        "dimensions": dict(dimensions.most_common()),
    }
    (args.output / "dataset_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.output / "image_index.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["location", "stem", "image_path", "label_path", "source_kind", "width", "height", "smoke", "box_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fields})

    draw_montage(records, args.output / "sample_smoke_boxes.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
