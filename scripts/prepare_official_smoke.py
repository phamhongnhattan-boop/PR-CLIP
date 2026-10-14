"""Build small image-disjoint training/validation CSVs for a pipeline check."""

import argparse
import csv
from pathlib import Path


def first_images(path: Path, image_count: int, one_caption_per_image: bool):
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    filenames = list(dict.fromkeys(row["filename"] for row in rows))[:image_count]
    if len(filenames) != image_count:
        raise ValueError(f"Only found {len(filenames)} images in {path}")
    selected = []
    used = set()
    for row in rows:
        filename = row["filename"]
        if filename not in filenames or (one_caption_per_image and filename in used):
            continue
        selected.append({"filename": filename, "title": row["title"]})
        used.add(filename)
    return selected


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=["filename", "title"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=2)
    parser.add_argument("--val-images", type=int, default=12)
    args = parser.parse_args()
    train = first_images(args.train_csv, args.train_images, one_caption_per_image=True)
    val = first_images(args.val_csv, args.val_images, one_caption_per_image=False)
    write_csv(args.train_output, train)
    write_csv(args.val_output, val)
    print(f"train: {len(train)} pairs; val: {len(val)} pairs")


if __name__ == "__main__":
    main()
