"""Evaluate OpenCLIP or a locally trained PR-CLIP backbone on author CSV splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import open_clip

from .data import load_retrieval_csv
from .metrics import retrieval_metrics


def encode_batches(model, preprocess, paths, captions, batch_size, device):
    image_vectors = []
    text_vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    batch.append(preprocess(image.convert("RGB")))
            tensor = torch.stack(batch).to(device)
            image_vectors.append(model.encode_image(tensor, normalize=True).float().cpu().numpy())
            print(f"images {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
        for start in range(0, len(captions), batch_size * 4):
            ids = open_clip.tokenize(captions[start:start + batch_size * 4]).to(device)
            text_vectors.append(model.encode_text(ids, normalize=True).float().cpu().numpy())
            print(f"captions {min(start + batch_size * 4, len(captions))}/{len(captions)}", flush=True)
    return np.concatenate(image_vectors), np.concatenate(text_vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/weights/open_clip"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths, captions, mapping = load_retrieval_csv(args.csv, args.images)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=args.pretrained, cache_dir=str(args.cache_dir))
    checkpoint_kind = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "clip" in checkpoint:
            weights = checkpoint["clip"]
            checkpoint_kind = "repro training checkpoint"
        elif "state_dict" in checkpoint:
            weights = checkpoint["state_dict"]
            checkpoint_kind = "OpenCLIP state_dict checkpoint"
        else:
            weights = checkpoint
            checkpoint_kind = "OpenCLIP state_dict checkpoint"
        model.load_state_dict(weights, strict=True)
    model.to(device)
    image_vectors, text_vectors = encode_batches(model, preprocess, paths, captions, args.batch_size, device)
    metrics = retrieval_metrics(image_vectors, text_vectors, mapping)
    report = {
        "kind": "fine-tuned OpenCLIP backbone" if args.checkpoint else "OpenAI CLIP baseline",
        "checkpoint_kind": checkpoint_kind,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "pretrained": args.pretrained,
        "device": device,
        "csv": str(args.csv.resolve()),
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "images": len(paths),
        "captions": len(captions),
        "metrics_percent": {key: round(value, 4) for key, value in metrics.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
