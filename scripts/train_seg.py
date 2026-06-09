#!/usr/bin/env python3
"""Train legal supervised segmentation or segmentation-heavy multi-task U-Nets."""

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

from cse164cv.constants import IGNORE_INDEX, NUM_CLASSES, NUM_SEG_CLASSES  # noqa: E402
from cse164cv.data import (  # noqa: E402
    ClassificationTrainTransform,
    ImageResizeTransform,
    SegmentationTrainTransform,
    TrainClassificationDataset,
    TrainSegmentationDataset,
    ValidationSegmentationDataset,
)
from cse164cv.metrics import foreground_miou, update_hist_from_logits  # noqa: E402
from cse164cv.models import MultiTaskUNet, build_model_from_config, split_model_output  # noqa: E402


DEFAULT_CONFIG = {
    "data_root": "data",
    "output_dir": "outputs/seg_baseline",
    "model_type": "small_unet",
    "epochs": 5,
    "batch_size": 8,
    "image_size": 256,
    "base_channels": 32,
    "classification_dropout": 0.2,
    "learning_rate": 1e-3,
    "min_learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "background_loss_weight": 0.05,
    "dice_weight": 0.0,
    "classification_weight": 0.0,
    "scheduler": "none",
    "hflip_prob": 0.5,
    "color_jitter": 0.0,
    "crop_scale_min": 1.0,
    "num_workers": 2,
    "seed": 164,
    "device": "auto",
    "max_train_samples": None,
    "max_classification_samples": None,
    "max_val_samples": None,
    "log_interval": 25,
    "overwrite_output": False,
    "resume": None,
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
        "model_type",
        "epochs",
        "batch_size",
        "image_size",
        "base_channels",
        "classification_dropout",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "background_loss_weight",
        "dice_weight",
        "classification_weight",
        "scheduler",
        "hflip_prob",
        "color_jitter",
        "crop_scale_min",
        "num_workers",
        "seed",
        "device",
        "max_train_samples",
        "max_classification_samples",
        "max_val_samples",
        "log_interval",
        "overwrite_output",
        "resume",
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
    parser.add_argument(
        "--model-type",
        choices=["small_unet", "multitask_unet"],
        default=default("model_type"),
    )
    parser.add_argument("--epochs", type=int, default=default("epochs"))
    parser.add_argument("--batch-size", type=int, default=default("batch_size"))
    parser.add_argument("--image-size", type=int, default=default("image_size"))
    parser.add_argument("--base-channels", type=int, default=default("base_channels"))
    parser.add_argument("--classification-dropout", type=float, default=default("classification_dropout"))
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=default("learning_rate"))
    parser.add_argument("--min-learning-rate", type=float, default=default("min_learning_rate"))
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
    parser.add_argument(
        "--classification-weight",
        type=float,
        default=default("classification_weight"),
        help="Weight for image classification CE. Requires model_type=multitask_unet.",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=default("scheduler"))
    parser.add_argument("--hflip-prob", type=float, default=default("hflip_prob"))
    parser.add_argument("--color-jitter", type=float, default=default("color_jitter"))
    parser.add_argument("--crop-scale-min", type=float, default=default("crop_scale_min"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers"))
    parser.add_argument("--seed", type=int, default=default("seed"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=default("device"))
    parser.add_argument("--max-train-samples", type=int, default=default("max_train_samples"))
    parser.add_argument("--max-classification-samples", type=int, default=default("max_classification_samples"))
    parser.add_argument("--max-val-samples", type=int, default=default("max_val_samples"))
    parser.add_argument("--log-interval", type=int, default=default("log_interval"))
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=default("overwrite_output"),
        help="Allow replacing checkpoints and metrics already present in --output-dir.",
    )
    parser.add_argument("--resume", type=Path, default=default("resume"), help="Resume from a saved last.pt checkpoint.")
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
    if resolved["resume"] is not None:
        resolved["resume"] = Path(resolved["resume"])
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
def validate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, float | int]:
    model.eval()
    hist = np.zeros((NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64)
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for batch in tqdm(loader, total=len(loader), desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        seg_logits, class_logits = split_model_output(model(images))
        hist = update_hist_from_logits(hist, seg_logits, masks)
        if class_logits is not None:
            predicted_classes = class_logits.argmax(dim=1)
            for target, predicted in zip(class_ids.cpu().numpy(), predicted_classes.cpu().numpy()):
                class_total[int(target)] += 1
                class_correct[int(target)] += int(target == predicted)
    miou, _ = foreground_miou(hist)
    valid_pixels = max(1, int(hist.sum()))
    classes_present = class_total > 0
    class_accuracy = float(class_correct.sum() / max(1, class_total.sum()))
    macro_accuracy = (
        float(np.mean(class_correct[classes_present] / class_total[classes_present]))
        if np.any(classes_present)
        else 0.0
    )
    return {
        "val_foreground_mIoU": miou,
        "val_gt_foreground_fraction": float(hist[1:, :].sum() / valid_pixels),
        "val_pred_foreground_fraction": float(hist[:, 1:].sum() / valid_pixels),
        "val_pred_foreground_classes": int(np.count_nonzero(hist[:, 1:].sum(axis=0))),
        "val_classification_accuracy": class_accuracy,
        "val_classification_macro_accuracy": macro_accuracy,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
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
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": {
            **jsonable_config(args),
            "num_classes": NUM_CLASSES,
            "num_seg_classes": NUM_SEG_CLASSES,
            "ignore_index": IGNORE_INDEX,
        },
    }
    torch.save(checkpoint, path)


def write_metrics(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Persist completed epoch metrics in CSV and JSON formats."""
    fieldnames = [
        "epoch",
        "train_loss",
        "train_seg_ce_loss",
        "train_dice_loss",
        "train_classification_loss",
        "val_foreground_mIoU",
        "val_gt_foreground_fraction",
        "val_pred_foreground_fraction",
        "val_pred_foreground_classes",
        "val_classification_accuracy",
        "val_classification_macro_accuracy",
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
        "final_val_classification_accuracy": float(final_row["val_classification_accuracy"]),
        "final_val_classification_macro_accuracy": float(final_row["val_classification_macro_accuracy"]),
        "hyperparameters": important_hyperparameters(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def ensure_output_is_available(output_dir: Path, overwrite_output: bool, resume: Path | None) -> None:
    """Prevent separate experiments from silently sharing checkpoints and metrics."""
    if overwrite_output or resume is not None or not output_dir.exists():
        return
    existing_artifacts = sorted(path.name for path in output_dir.iterdir() if path.name in RUN_ARTIFACT_NAMES)
    if existing_artifacts:
        artifacts = ", ".join(existing_artifacts)
        raise FileExistsError(
            f"Output directory {output_dir} already contains run artifacts: {artifacts}. "
            "Choose a new --output-dir, or pass --overwrite-output to replace this run."
        )


def load_existing_metrics(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "metrics.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of epoch metrics in {path}")
    return rows


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("val_miou", -1.0))


def main() -> None:
    args = parse_args()
    if args.dice_weight < 0 or args.classification_weight < 0:
        raise ValueError("--dice-weight and --classification-weight must be non-negative")
    if args.classification_weight > 0 and args.model_type != "multitask_unet":
        raise ValueError("--classification-weight requires --model-type multitask_unet")
    if not 0 <= args.classification_dropout < 1:
        raise ValueError("--classification-dropout must be in [0, 1)")
    if not 0 <= args.hflip_prob <= 1:
        raise ValueError("--hflip-prob must be in [0, 1]")
    if args.color_jitter < 0:
        raise ValueError("--color-jitter must be non-negative")
    if not 0 < args.crop_scale_min <= 1:
        raise ValueError("--crop-scale-min must be in (0, 1]")
    if args.min_learning_rate < 0:
        raise ValueError("--min-learning-rate must be non-negative")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    ensure_output_is_available(args.output_dir, args.overwrite_output, args.resume)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = jsonable_config(args)
    (args.output_dir / "args.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True))
    (args.output_dir / "config_used.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True))
    dump_flat_yaml(args.output_dir / "resolved_config.yaml", resolved_config)

    full_train_dataset = TrainSegmentationDataset(
        args.data_root,
        transform=SegmentationTrainTransform(
            args.image_size,
            hflip_prob=args.hflip_prob,
            color_jitter=args.color_jitter,
            crop_scale_min=args.crop_scale_min,
        ),
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
    full_classification_dataset = None
    classification_dataset = None
    classification_loader = None
    if args.classification_weight > 0:
        full_classification_dataset = TrainClassificationDataset(
            args.data_root,
            transform=ClassificationTrainTransform(
                args.image_size,
                hflip_prob=args.hflip_prob,
                color_jitter=args.color_jitter,
                crop_scale_min=args.crop_scale_min,
            ),
        )
        classification_dataset = apply_deterministic_limit(
            full_classification_dataset,
            args.max_classification_samples,
            args.seed + 2,
            "classification",
        )
        classification_loader = make_loader(
            classification_dataset,
            args.batch_size,
            True,
            args.num_workers,
            device,
        )
    # Validation keeps original masks and upsamples logits to their exact size,
    # so use batch size 1 to avoid variable-size mask collation issues.
    val_loader = make_loader(val_dataset, 1, False, args.num_workers, device)

    model_config = {
        **resolved_config,
        "num_classes": NUM_CLASSES,
        "num_seg_classes": NUM_SEG_CLASSES,
    }
    model = build_model_from_config(model_config).to(device)
    class_weights = torch.ones(NUM_SEG_CLASSES, dtype=torch.float32, device=device)
    class_weights[0] = args.background_loss_weight
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_learning_rate,
        )

    metric_rows = load_existing_metrics(args.output_dir) if args.resume is not None else []
    best_miou = max(
        (float(row["val_foreground_mIoU"]) for row in metric_rows),
        default=-1.0,
    )
    start_epoch = 1
    if args.resume is not None:
        start_epoch, checkpoint_miou = load_resume_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
        )
        best_miou = max(best_miou, checkpoint_miou)

    print(
        f"Using {len(train_dataset)}/{len(full_train_dataset)} train samples and "
        f"{len(val_dataset)}/{len(full_val_dataset)} validation samples"
    )
    if classification_dataset is not None and full_classification_dataset is not None:
        print(
            f"Using {len(classification_dataset)}/{len(full_classification_dataset)} "
            "classification-labeled samples"
        )
    print(
        f"Training from scratch on {len(train_dataset)} train masks; "
        f"validating on {len(val_dataset)} public masks; device={device}; "
        f"model_type={args.model_type}; "
        f"background_loss_weight={args.background_loss_weight}; "
        f"dice_weight={args.dice_weight}; "
        f"classification_weight={args.classification_weight}"
    )
    if args.resume is not None:
        print(f"Resuming from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_seg_ce_loss = 0.0
        running_dice_loss = 0.0
        running_classification_loss = 0.0
        seen = 0
        classification_iterator = iter(classification_loader) if classification_loader is not None else None
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
            seg_logits, seg_class_logits = split_model_output(model(images))
            ce_loss = ce_criterion(seg_logits, masks)
            if args.dice_weight > 0:
                dice_loss = foreground_dice_loss(seg_logits, masks)
            else:
                dice_loss = seg_logits.sum() * 0.0

            classification_loss = seg_logits.sum() * 0.0
            seg_objective = ce_loss + args.dice_weight * dice_loss
            if classification_iterator is not None:
                if seg_class_logits is None or not isinstance(model, MultiTaskUNet):
                    raise RuntimeError("Multi-task classification requires MultiTaskUNet outputs")
                seg_class_ids = batch["class_id"].to(device, non_blocking=True)
                seg_classification_loss = classification_criterion(seg_class_logits, seg_class_ids)
                seg_objective = seg_objective + 0.5 * args.classification_weight * seg_classification_loss
                try:
                    classification_batch = next(classification_iterator)
                except StopIteration:
                    classification_iterator = iter(classification_loader)
                    classification_batch = next(classification_iterator)
            else:
                seg_classification_loss = classification_loss
                classification_batch = None

            # Backpropagate the segmentation-heavy objective first so the
            # decoder graph can be freed before the image-only forward pass.
            seg_objective.backward()

            labeled_classification_loss = classification_loss
            if classification_batch is not None:
                classification_images = classification_batch["image"].to(device, non_blocking=True)
                classification_ids = classification_batch["class_id"].to(device, non_blocking=True)
                labeled_logits = model.forward_classification(classification_images)
                labeled_classification_loss = classification_criterion(labeled_logits, classification_ids)
                (0.5 * args.classification_weight * labeled_classification_loss).backward()

            if classification_iterator is not None:
                classification_loss = 0.5 * (seg_classification_loss + labeled_classification_loss)
            loss = ce_loss + args.dice_weight * dice_loss + args.classification_weight * classification_loss
            optimizer.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            running_seg_ce_loss += float(ce_loss.item()) * batch_size
            running_dice_loss += float(dice_loss.item()) * batch_size
            running_classification_loss += float(classification_loss.item()) * batch_size
            seen += batch_size
            train_bar.set_postfix(
                loss=f"{running_loss / max(1, seen):.4f}",
                seg_ce=f"{running_seg_ce_loss / max(1, seen):.4f}",
                cls=f"{running_classification_loss / max(1, seen):.4f}",
            )

        train_loss = running_loss / max(1, seen)
        val_metrics = validate(model, val_loader, device, desc=f"epoch {epoch}/{args.epochs} val")
        val_miou = val_metrics["val_foreground_mIoU"]
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, scheduler, epoch, val_miou, args)
        best_checkpoint_path = ""
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = args.output_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_miou, args)
            best_checkpoint_path = str(best_path)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seg_ce_loss": running_seg_ce_loss / max(1, seen),
            "train_dice_loss": running_dice_loss / max(1, seen),
            "train_classification_loss": running_classification_loss / max(1, seen),
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
