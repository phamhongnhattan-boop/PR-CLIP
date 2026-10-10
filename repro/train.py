"""Train paper objective on RET-2, with explicit resource-limited smoke options."""

import argparse
import csv
import json
import math
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip

from .openclip_tokens import encode_with_tokens
from .prclip import PositionalReconstruction


class Pairs(Dataset):
    def __init__(self, csv_path, image_dir, transform, max_pairs, seed):
        self.image_dir = image_dir
        self.transform = transform
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            self.rows = [(row["filename"], row["title"]) for row in csv.DictReader(stream)]
        if max_pairs:
            rng = random.Random(seed)
            self.rows = rng.sample(self.rows, min(max_pairs, len(self.rows)))
        missing = [name for name, _ in self.rows if not (image_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} training images missing; first: {missing[:3]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        filename, caption = self.rows[index]
        with Image.open(self.image_dir / filename) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Optional OpenCLIP/RET-2 state_dict to initialize both encoders")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--extraction-layers", type=int, default=2)
    parser.add_argument("--reconstruction-layers", type=int, default=8)
    parser.add_argument("--lambda-cmc", type=float, default=0.01)
    parser.add_argument("--lambda-prc", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/weights/open_clip"))
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip, preprocess_train, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=args.pretrained, cache_dir=str(args.cache_dir))
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        weights = initial.get("state_dict", initial)
        clip.load_state_dict(weights, strict=True)
    clip.to(device)
    clip.set_grad_checkpointing(True)
    for parameter in clip.parameters():
        parameter.requires_grad_(not args.freeze_backbone)
    aux = PositionalReconstruction(
        extraction_layers=args.extraction_layers,
        reconstruction_layers=args.reconstruction_layers,
        lambda_cmc=args.lambda_cmc,
        lambda_prc=args.lambda_prc,
        temperature=args.temperature).to(device)
    pairs = Pairs(args.csv, args.images, preprocess_train, args.max_pairs, args.seed)
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        drop_last=True)
    if not len(loader):
        raise ValueError("Training split must contain at least one full batch")
    parameters = [p for p in list(clip.parameters()) + list(aux.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps or args.epochs * len(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    with args.log.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"configuration": vars(args) | {"csv": str(args.csv),
                                                         "images": str(args.images),
                                                         "output": str(args.output),
                                                         "log": str(args.log),
                                                         "cache_dir": str(args.cache_dir)},
                              "device": device, "pairs": len(pairs)}) + "\n")
        for epoch in range(args.epochs):
            clip.train()
            aux.train()
            for images, captions in loader:
                if step >= total_steps:
                    break
                if args.warmup and step < args.warmup:
                    scale = (step + 1) / args.warmup
                else:
                    progress = (step - args.warmup) / max(total_steps - args.warmup, 1)
                    scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * scale
                images = images.to(device)
                ids = open_clip.tokenize(list(captions)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    image_tokens, text_tokens, image_global, text_global, text_mask = \
                        encode_with_tokens(clip, images, ids)
                    losses = aux(image_tokens, text_tokens, image_global, text_global, text_mask)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                record = {"epoch": epoch + 1, "step": step, "lr": group["lr"],
                          "loss": float(losses["loss"].detach()),
                          "cmc": float(losses["cmc"].detach()),
                          "prc": float(losses["prc"].detach())}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)
            if step >= total_steps:
                break
    torch.save({"clip": clip.cpu().state_dict(), "aux": aux.cpu().state_dict(),
                "steps": step, "configuration": vars(args)}, args.output)
    print(f"saved {args.output} after {step} steps", flush=True)


if __name__ == "__main__":
    main()
