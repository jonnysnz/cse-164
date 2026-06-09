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
from torch.utils.data import DataLoader, Dataset, Subset
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


DEFAULT_CONFIG = {
    "data_root": "data",
    "output_dir": "outputs/seg_baseline",
    "epochs": 5,
    "batch_size": 8,
    "image_size": 256,
    "base_channels": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "background_loss_weight": 0.05,
    "dice_weight": 0.0,
    "num_workers": 2,
    "seed": 164,
    "device": "auto",
    "max_train_samples": None,
    "max_val_samples": None,
    "log_interval": 25,
    "overwrite_output": False,
}

RUN_ARTIFACT_NAMES = {
    "args.json",
    "best.pt",
    "config_used.json",
    "last.pt",
    "metrics.csv",
    "metrics.json",
    "resolved_config.yaml",
    "summary.json",
}


def parse_yaml_scalar(value: str) -> object:
    value = value.strip()
    if value == "" or value.lower() in {"null", "none"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def load_flat_yaml(path: Path) -> dict[str, object]:
    """Load this project's flat key-value YAML config without extra deps."""
    config: dict[str, object] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'key: value'")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty config key")
        config[key] = parse_yaml_scalar(value)
    return config


def dump_flat_yaml(path: Path, config: dict[str, object]) -> None:
    lines = []
    for key, value in config.items():
        if value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def jsonable_config(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def important_hyperparameters(args: argparse.Namespace) -> dict[str, object]:
    keys = [
        "data_root",
        "output_dir",
        "epochs",
        "batch_size",
        "image_size",
        "base_channels",
        "learning_rate",
        "weight_decay",
        "background_loss_weight",
        "dice_weight",
        "num_workers",
        "seed",
        "device",
        "max_train_samples",
        "max_val_samples",
        "log_interval",
        "overwrite_output",
        "config",
    ]
    all_values = jsonable_config(args)
    return {key: all_values[key] for key in keys}


def make_parser(defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Flat YAML config file for segmentation training.")

    def default(name: str) -> object:
        return defaults.get(name, argparse.SUPPRESS)

    parser.add_argument("--data-root", type=Path, default=default("data_root"))
    parser.add_argument("--output-dir", type=Path, default=default("output_dir"))
    parser.add_argument("--epochs", type=int, default=default("epochs"))
    parser.add_argument("--batch-size", type=int, default=default("batch_size"))
    parser.add_argument("--image-size", type=int, default=default("image_size"))
    parser.add_argument("--base-channels", type=int, default=default("base_channels"))
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=default("learning_rate"))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay"))
    parser.add_argument(
        "--background-loss-weight",
        type=float,
        default=default("background_loss_weight"),
        help="Cross-entropy weight for background id 0; foreground ids 1..300 keep weight 1.0.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=default("dice_weight"),
        help="Weight for optional foreground Dice loss. Default 0.0 disables Dice loss.",
    )
    parser.add_argument("--num-workers", type=int, default=default("num_workers"))
    parser.add_argument("--seed", type=int, default=default("seed"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=default("device"))
    parser.add_argument("--max-train-samples", type=int, default=default("max_train_samples"))
    parser.add_argument("--max-val-samples", type=int, default=default("max_val_samples"))
    parser.add_argument("--log-interval", type=int, default=default("log_interval"))
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=default("overwrite_output"),
        help="Allow replacing checkpoints and metrics already present in --output-dir.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()

    config_values: dict[str, object] = {}
    if config_args.config:
        config_values = load_flat_yaml(config_args.config)
        unknown = sorted(set(config_values) - set(DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown config keys in {config_args.config}: {unknown}")

    merged_defaults = {**DEFAULT_CONFIG, **config_values}
    parser = make_parser(merged_defaults)
    cli_args = vars(parser.parse_args())
    resolved = {**merged_defaults, **cli_args}
    resolved["config"] = str(config_args.config) if config_args.config else None
    resolved["data_root"] = Path(resolved["data_root"])
    resolved["output_dir"] = Path(resolved["output_dir"])
    return argparse.Namespace(**resolved)


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
    dataset: Dataset,
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


def apply_deterministic_limit(dataset: Dataset, max_samples: int | None, seed: int, name: str) -> Dataset:
    """Use a deterministic seed-selected subset for quick/debug runs."""
    if max_samples is None:
        return dataset
    if max_samples < 1:
        raise ValueError(f"--max-{name}-samples must be positive or null")
    limit = min(max_samples, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def foreground_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss over foreground target classes, ignoring id 1000.

    CE still handles all classes, including background. This auxiliary term only
    compares channels 1..300 for foreground classes present in the current batch.
    Keeping the Dice target set batch-local avoids building a huge 300-channel
    one-hot tensor for every batch.
    """
    valid = target != IGNORE_INDEX
    foreground = valid & (target > 0) & (target < NUM_SEG_CLASSES)
    present_ids = torch.unique(target[foreground])
    if present_ids.numel() == 0:
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=1)
    fg_probs = probs[:, present_ids, :, :]
    fg_target = (target.unsqueeze(1) == present_ids.view(1, -1, 1, 1)).to(dtype=fg_probs.dtype)
    valid_mask = valid.unsqueeze(1).to(dtype=fg_probs.dtype)

    fg_probs = fg_probs * valid_mask
    fg_target = fg_target * valid_mask
    dims = (0, 2, 3)
    intersection = (fg_probs * fg_target).sum(dim=dims)
    denominator = fg_probs.sum(dim=dims) + fg_target.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


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
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "background_loss_weight": args.background_loss_weight,
            "dice_weight": args.dice_weight,
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


def write_summary(output_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    if not rows:
        return
    best_row = max(rows, key=lambda row: float(row["val_foreground_mIoU"]))
    final_row = rows[-1]
    summary = {
        "best_epoch": int(best_row["epoch"]),
        "best_val_foreground_mIoU": float(best_row["val_foreground_mIoU"]),
        "best_checkpoint_path": str(output_dir / "best.pt"),
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_val_foreground_mIoU": float(final_row["val_foreground_mIoU"]),
        "hyperparameters": important_hyperparameters(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def ensure_output_is_available(output_dir: Path, overwrite_output: bool) -> None:
    """Prevent separate experiments from silently sharing checkpoints and metrics."""
    if overwrite_output or not output_dir.exists():
        return
    existing_artifacts = sorted(path.name for path in output_dir.iterdir() if path.name in RUN_ARTIFACT_NAMES)
    if existing_artifacts:
        artifacts = ", ".join(existing_artifacts)
        raise FileExistsError(
            f"Output directory {output_dir} already contains run artifacts: {artifacts}. "
            "Choose a new --output-dir, or pass --overwrite-output to replace this run."
        )


def main() -> None:
    args = parse_args()
    if args.dice_weight < 0:
        raise ValueError("--dice-weight must be non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    ensure_output_is_available(args.output_dir, args.overwrite_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = jsonable_config(args)
    (args.output_dir / "args.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True))
    (args.output_dir / "config_used.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True))
    dump_flat_yaml(args.output_dir / "resolved_config.yaml", resolved_config)

    full_train_dataset = TrainSegmentationDataset(
        args.data_root,
        transform=SegmentationTrainTransform(args.image_size),
    )
    full_val_dataset = ValidationSegmentationDataset(
        args.data_root,
        image_transform=ImageResizeTransform(args.image_size),
    )
    train_dataset = apply_deterministic_limit(
        full_train_dataset,
        args.max_train_samples,
        args.seed,
        "train",
    )
    val_dataset = apply_deterministic_limit(
        full_val_dataset,
        args.max_val_samples,
        args.seed + 1,
        "val",
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    # Validation keeps original masks and upsamples logits to their exact size,
    # so use batch size 1 to avoid variable-size mask collation issues.
    val_loader = make_loader(val_dataset, 1, False, args.num_workers, device)

    model = SmallUNet(num_classes=NUM_SEG_CLASSES, base_channels=args.base_channels).to(device)
    class_weights = torch.ones(NUM_SEG_CLASSES, dtype=torch.float32, device=device)
    class_weights[0] = args.background_loss_weight
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_miou = -1.0
    print(
        f"Using {len(train_dataset)}/{len(full_train_dataset)} train samples and "
        f"{len(val_dataset)}/{len(full_val_dataset)} validation samples"
    )
    print(
        f"Training from scratch on {len(train_dataset)} train masks; "
        f"validating on {len(val_dataset)} public masks; device={device}; "
        f"background_loss_weight={args.background_loss_weight}; "
        f"dice_weight={args.dice_weight}"
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
            ce_loss = ce_criterion(logits, masks)
            if args.dice_weight > 0:
                dice_loss = foreground_dice_loss(logits, masks)
                loss = ce_loss + args.dice_weight * dice_loss
            else:
                loss = ce_loss
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
        write_summary(args.output_dir, metric_rows, args)
        print(f"epoch_metrics: {json.dumps(epoch_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
