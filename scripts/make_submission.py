#!/usr/bin/env python3
"""Generate a valid submission.csv for val or test images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cse164cv.constants import NUM_CLASSES  # noqa: E402
from cse164cv.data import ImageOnlyDataset, ImageResizeTransform  # noqa: E402
from cse164cv.models import build_model_from_config, split_model_output  # noqa: E402
from cse164cv.rle import encode_segmentation_rle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--constant-class-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


@torch.no_grad()
def predict_outputs(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_size: torch.Tensor,
    device: torch.device,
    constant_class_id: int,
) -> tuple[np.ndarray, int]:
    image = image.to(device, non_blocking=True)
    logits, class_logits = split_model_output(model(image))
    height = int(original_size[0, 0].item())
    width = int(original_size[0, 1].item())
    if logits.shape[-2:] != (height, width):
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint16)
    class_id = constant_class_id if class_logits is None else int(class_logits.argmax(dim=1).item())
    return mask, class_id


def main() -> None:
    args = parse_args()
    if not (0 <= args.constant_class_id < NUM_CLASSES):
        raise ValueError("--constant-class-id must be in 0..299")
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1; masks are resized and encoded at each image's original size")

    device = resolve_device(args.device)
    model = None
    checkpoint_config: dict = {}
    if args.checkpoint:
        model, checkpoint_config = load_model(args.checkpoint, device)
        print(f"Loaded model checkpoint {args.checkpoint} on {device}")
    else:
        print("No checkpoint supplied; writing all-background masks.")

    image_size = args.image_size or int(checkpoint_config.get("image_size", 256))
    base_channels = args.base_channels or int(checkpoint_config.get("base_channels", 32))
    if model is None and args.base_channels:
        print(f"Ignoring --base-channels={base_channels} because no checkpoint was supplied.")

    dataset = ImageOnlyDataset(
        args.data_root,
        args.split,
        transform=ImageResizeTransform(image_size),
        max_items=args.max_items,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "segmentation_rle"])
        writer.writeheader()
        for batch in loader:
            if model is not None:
                if batch["image"].size(0) != 1:
                    raise ValueError("Model prediction currently expects --batch-size 1")
                mask, class_id = predict_outputs(
                    model,
                    batch["image"],
                    batch["original_size"],
                    device,
                    args.constant_class_id,
                )
            else:
                height = int(batch["original_size"][0, 0].item())
                width = int(batch["original_size"][0, 1].item())
                mask = np.zeros((height, width), dtype=np.uint16)
                class_id = args.constant_class_id
            writer.writerow(
                {
                    "image": batch["image_name"][0],
                    "class_id": class_id,
                    "segmentation_rle": encode_segmentation_rle(mask),
                }
            )

    print(f"Wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
