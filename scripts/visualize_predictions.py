#!/usr/bin/env python3
"""Visualize validation predictions from a trained segmentation checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cse164cv.constants import IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from cse164cv.data import ImageResizeTransform, ValidationSegmentationDataset  # noqa: E402
from cse164cv.models import build_model_from_config, split_model_output  # noqa: E402

try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    BILINEAR = Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/seg_weighted_256/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seg_weighted_256/visualizations"))
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--display-width", type=int, default=320)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--no-overlay", action="store_true", help="Do not include the prediction overlay panel.")
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


def color_for_id(seg_id: int) -> tuple[int, int, int]:
    """Deterministic colors for mask ids without relying on external assets."""
    if seg_id == 0:
        return (0, 0, 0)
    if seg_id == IGNORE_INDEX:
        return (150, 150, 150)
    value = int(seg_id)
    return (
        (37 * value + 53) % 256,
        (91 * value + 29) % 256,
        (17 * value + 191) % 256,
    )


def colorize_mask(mask: np.ndarray) -> Image.Image:
    mask = np.asarray(mask, dtype=np.int64)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for seg_id in np.unique(mask):
        color[mask == seg_id] = color_for_id(int(seg_id))
    return Image.fromarray(color, mode="RGB")


def prediction_overlay(original: Image.Image, pred_mask: np.ndarray, alpha: float) -> Image.Image:
    original_arr = np.asarray(original.convert("RGB"), dtype=np.float32)
    color_arr = np.asarray(colorize_mask(pred_mask), dtype=np.float32)
    foreground = pred_mask != 0
    blended = original_arr.copy()
    blended[foreground] = (1.0 - alpha) * original_arr[foreground] + alpha * color_arr[foreground]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def add_title(image: Image.Image, title: str, height: int = 34) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + height), color=(245, 245, 245))
    result.paste(image, (0, height))
    draw = ImageDraw.Draw(result)
    draw.text((8, 9), title, fill=(0, 0, 0))
    return result


def make_text_panel(width: int, height: int, lines: list[str]) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(panel)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0))
        y += 18
    return panel


def resize_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), BILINEAR)


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image: torch.Tensor, original_size: torch.Tensor, device: torch.device) -> np.ndarray:
    image = image.to(device, non_blocking=True)
    logits, _ = split_model_output(model(image))
    height = int(original_size[0, 0].item())
    width = int(original_size[0, 1].item())
    if logits.shape[-2:] != (height, width):
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint16)
    if np.any(mask > NUM_CLASSES):
        raise ValueError("Predicted mask contains ids outside 0..300")
    return mask


def save_visualization(
    output_path: Path,
    original: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    image_name: str,
    class_id: int,
    display_width: int,
    include_overlay: bool,
    overlay_alpha: float,
) -> None:
    display_height = max(1, round(original.height * display_width / original.width))
    panels = [
        add_title(resize_panel(original.convert("RGB"), display_width, display_height), "original"),
        add_title(resize_panel(colorize_mask(gt_mask), display_width, display_height), "ground truth"),
        add_title(resize_panel(colorize_mask(pred_mask), display_width, display_height), "prediction"),
    ]
    if include_overlay:
        overlay = prediction_overlay(original, pred_mask, overlay_alpha)
        panels.append(add_title(resize_panel(overlay, display_width, display_height), "prediction overlay"))

    unique_pred = np.unique(pred_mask)
    foreground_fraction = float(np.mean(pred_mask != 0))
    shown_ids = " ".join(str(int(value)) for value in unique_pred[:30])
    if len(unique_pred) > 30:
        shown_ids += f" ... ({len(unique_pred)} total)"
    text_lines = [
        f"image: {image_name}",
        f"class_id: {class_id}",
        f"pred foreground fraction: {foreground_fraction:.4f}",
        f"unique predicted ids: {shown_ids}",
    ]

    panel_width = display_width * len(panels)
    panel_height = max(panel.height for panel in panels)
    text_panel = make_text_panel(panel_width, 90, text_lines)
    canvas = Image.new("RGB", (panel_width, panel_height + text_panel.height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += display_width
    canvas.paste(text_panel, (0, panel_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.num_examples < 1:
        raise ValueError("--num-examples must be positive")

    device = resolve_device(args.device)
    model, checkpoint_config = load_model(args.checkpoint, device)
    image_size = args.image_size or int(checkpoint_config.get("image_size", 256))
    if args.base_channels is not None and args.base_channels != int(checkpoint_config.get("base_channels", 32)):
        print("--base-channels is ignored because the checkpoint config defines the loaded model.")

    dataset = ValidationSegmentationDataset(
        args.data_root,
        image_transform=ImageResizeTransform(image_size),
        max_items=args.num_examples,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print(f"Loaded {args.checkpoint} on {device}; saving {len(dataset)} examples to {args.output_dir}")
    for index, batch in enumerate(tqdm(loader, total=len(loader), desc="visualizing"), start=1):
        image_name = batch["image_name"][0]
        image_path = args.data_root / "val" / "images" / image_name
        with Image.open(image_path) as image_file:
            original = image_file.convert("RGB")

        gt_mask = batch["mask"].squeeze(0).numpy()
        pred_mask = predict_mask(model, batch["image"], batch["original_size"], device)
        class_id = int(batch["class_id"][0].item())
        output_path = args.output_dir / f"{index:03d}_{Path(image_name).stem}.png"
        save_visualization(
            output_path=output_path,
            original=original,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            image_name=image_name,
            class_id=class_id,
            display_width=args.display_width,
            include_overlay=not args.no_overlay,
            overlay_alpha=args.overlay_alpha,
        )
        unique_pred = np.unique(pred_mask)
        print(
            f"{output_path}: foreground_fraction={float(np.mean(pred_mask != 0)):.4f}, "
            f"unique_pred_ids={unique_pred[:30].astype(int).tolist()}"
        )


if __name__ == "__main__":
    main()
