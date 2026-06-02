#!/usr/bin/env python3
"""Train the first legal supervised segmentation baseline from scratch."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cse164cv.constants import IGNORE_INDEX, NUM_SEG_CLASSES  # noqa: E402
from cse164cv.data import (  # noqa: E402
    ImageResizeTransform,
    SegmentationTrainTransform,
    TrainSegmentationDataset,
    ValidationSegmentationDataset,
)
from cse164cv.metrics import foreground_miou, update_hist_from_logits  # noqa: E402
from cse164cv.models import SmallUNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seg_baseline"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--background-loss-weight",
        type=float,
        default=0.05,
        help="Cross-entropy weight for background id 0; foreground ids 1..300 keep weight 1.0.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=164)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, float]:
    model.eval()
    hist = np.zeros((NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64)
    for batch in tqdm(loader, total=len(loader), desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        hist = update_hist_from_logits(hist, logits, masks)
    miou, _ = foreground_miou(hist)
    valid_pixels = max(1, int(hist.sum()))
    return {
        "val_foreground_mIoU": miou,
        "val_gt_foreground_fraction": float(hist[1:, :].sum() / valid_pixels),
        "val_pred_foreground_fraction": float(hist[:, 1:].sum() / valid_pixels),
        "val_pred_foreground_classes": int(np.count_nonzero(hist[:, 1:].sum(axis=0))),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_miou: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "val_miou": val_miou,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "num_seg_classes": NUM_SEG_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "background_loss_weight": args.background_loss_weight,
        },
    }
    torch.save(checkpoint, path)


def write_metrics(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Persist completed epoch metrics in CSV and JSON formats."""
    fieldnames = [
        "epoch",
        "train_loss",
        "val_foreground_mIoU",
        "val_gt_foreground_fraction",
        "val_pred_foreground_fraction",
        "val_pred_foreground_classes",
        "learning_rate",
        "best_checkpoint_path",
    ]
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    train_dataset = TrainSegmentationDataset(
        args.data_root,
        transform=SegmentationTrainTransform(args.image_size),
        max_items=args.max_train_samples,
    )
    val_dataset = ValidationSegmentationDataset(
        args.data_root,
        image_transform=ImageResizeTransform(args.image_size),
        max_items=args.max_val_samples,
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    # Validation keeps original masks and upsamples logits to their exact size,
    # so use batch size 1 to avoid variable-size mask collation issues.
    val_loader = make_loader(val_dataset, 1, False, args.num_workers, device)

    model = SmallUNet(num_classes=NUM_SEG_CLASSES, base_channels=args.base_channels).to(device)
    class_weights = torch.ones(NUM_SEG_CLASSES, dtype=torch.float32, device=device)
    class_weights[0] = args.background_loss_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_miou = -1.0
    print(
        f"Training from scratch on {len(train_dataset)} train masks; "
        f"validating on {len(val_dataset)} public masks; device={device}; "
        f"background_loss_weight={args.background_loss_weight}"
    )
    metric_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        train_bar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"epoch {epoch}/{args.epochs} train",
            leave=False,
        )
        for step, batch in train_bar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            train_bar.set_postfix(train_loss=f"{running_loss / max(1, seen):.4f}")

        train_loss = running_loss / max(1, seen)
        val_metrics = validate(model, val_loader, device, desc=f"epoch {epoch}/{args.epochs} val")
        val_miou = val_metrics["val_foreground_mIoU"]
        learning_rate = float(optimizer.param_groups[0]["lr"])

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, val_miou, args)
        best_checkpoint_path = ""
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = args.output_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, val_miou, args)
            best_checkpoint_path = str(best_path)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "learning_rate": learning_rate,
            "best_checkpoint_path": best_checkpoint_path,
        }
        metric_rows.append(epoch_metrics)
        write_metrics(args.output_dir, metric_rows)
        print(f"epoch_metrics: {json.dumps(epoch_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
