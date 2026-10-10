"""Load the author's retrieval CSV files without changing their test split."""

from collections import OrderedDict
import csv
from pathlib import Path


def load_retrieval_csv(csv_path: str | Path, image_dir: str | Path):
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    images = OrderedDict()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"filename", "title"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain filename and title columns")
        for row in reader:
            filename, caption = row["filename"], row["title"]
            if not filename or not caption:
                raise ValueError("Empty filename or caption")
            images.setdefault(filename, []).append(caption)
    if not images:
        raise ValueError("Retrieval CSV contains no pairs")
    missing = [name for name in images if not (image_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[:5]}")
    paths = [image_dir / name for name in images]
    captions = []
    text_to_image = []
    for image_index, image_captions in enumerate(images.values()):
        captions.extend(image_captions)
        text_to_image.extend([image_index] * len(image_captions))
    return paths, captions, text_to_image
