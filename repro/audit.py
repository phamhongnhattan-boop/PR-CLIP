"""Audit author-provided RET-2 and test CSVs against the released image archive."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


FILES = ("train/ret2.csv", "test/rsitmd_test.csv", "test/rsicd_test.csv",
         "test/rsitmd_val.csv", "test/rsicd_val.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()
    counts = {}
    names = {}
    for relative in FILES:
        csv_path = args.dataset / relative
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        c = Counter(row["filename"] for row in rows)
        names[relative] = set(c)
        counts[relative] = {"pairs": len(rows), "images": len(c),
                            "captions_per_image": dict(Counter(c.values())),
                            "missing_images": sum(not (args.images / name).is_file() for name in c),
                            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    train = names["train/ret2.csv"]
    test = names["test/rsitmd_test.csv"] | names["test/rsicd_test.csv"]
    report = {"splits": counts,
              "train_test_filename_overlap": len(train & test),
              "rsitmd_rsicd_test_filename_overlap": len(names["test/rsitmd_test.csv"] &
                                                         names["test/rsicd_test.csv"]),
              "rsitmd_val_test_filename_overlap": len(names["test/rsitmd_val.csv"] &
                                                       names["test/rsitmd_test.csv"]),
              "rsicd_val_test_filename_overlap": len(names["test/rsicd_val.csv"] &
                                                      names["test/rsicd_test.csv"]),
              "rsicd_val_and_test_csv_identical": counts["test/rsicd_val.csv"]["csv_sha256"] ==
                                                   counts["test/rsicd_test.csv"]["csv_sha256"]}
    if args.hash_images:
        digest_to_train = {}
        for name in train:
            digest_to_train.setdefault(hashlib.sha256((args.images / name).read_bytes()).hexdigest(),
                                       []).append(name)
        collisions = []
        for name in test:
            digest = hashlib.sha256((args.images / name).read_bytes()).hexdigest()
            if digest in digest_to_train:
                collisions.append({"test": name, "train": digest_to_train[digest]})
        report["train_test_exact_content_overlap"] = collisions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
