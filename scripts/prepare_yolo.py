import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


LOCATIONS = ("Evo", "Heinola", "Karkkila", "Ruokolahti")


def copy_verified(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise ValueError(f"Destination collision: {dst}")
        return
    shutil.copy2(src, dst)


def stratified_split(rows, seed, val_fraction=0.1):
    train, val = [], []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["location"], row["smoke"])].append(row)
    for key, group in sorted(groups.items()):
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction) if len(group) >= 10 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def write_paths(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(r["prepared_image"]) for r in rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    rows = []
    with args.index.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["smoke"] = row["smoke"].lower() == "true"
            image_src = Path(row["image_path"])
            label_src = Path(row["label_path"])
            image_dst = args.dataset / "images" / row["location"] / image_src.name
            label_dst = args.dataset / "labels" / row["location"] / label_src.name
            copy_verified(image_src, image_dst)
            copy_verified(label_src, label_dst)
            row["prepared_image"] = image_dst
            row["prepared_label"] = label_dst
            rows.append(row)

    folds_root = args.output / "folds"
    split_summary = {}
    for heldout in LOCATIONS:
        training_pool = [r for r in rows if r["location"] != heldout]
        test = [r for r in rows if r["location"] == heldout]
        train, val = stratified_split(training_pool, args.seed)
        fold = folds_root / f"holdout_{heldout.lower()}"
        write_paths(fold / "train.txt", train)
        write_paths(fold / "val.txt", val)
        write_paths(fold / "test.txt", test)
        yaml = (
            f"path: {args.dataset.as_posix()}\n"
            f"train: {(fold / 'train.txt').as_posix()}\n"
            f"val: {(fold / 'val.txt').as_posix()}\n"
            f"test: {(fold / 'test.txt').as_posix()}\n"
            "names:\n  0: smoke\n"
        )
        (fold / "data.yaml").write_text(yaml, encoding="utf-8")
        split_summary[heldout] = {
            "train": len(train), "validation": len(val), "test": len(test),
            "train_locations": sorted({r["location"] for r in train}),
            "test_location": heldout,
            "train_smoke": sum(r["smoke"] for r in train),
            "validation_smoke": sum(r["smoke"] for r in val),
            "test_smoke": sum(r["smoke"] for r in test),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(json.dumps(split_summary, indent=2))


if __name__ == "__main__":
    main()
