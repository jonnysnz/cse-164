#!/usr/bin/env python3
"""Compare segmentation training runs by best validation foreground mIoU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to print; 0 means all runs.")
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def read_metrics(run_dir: Path) -> list[dict[str, object]]:
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    if metrics_json.is_file():
        data = read_json(metrics_json)
        if isinstance(data, list):
            return [dict(row) for row in data]
    if metrics_csv.is_file():
        with metrics_csv.open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return []


def read_hyperparameters(run_dir: Path) -> dict[str, object]:
    for name in ("config_used.json", "args.json"):
        path = run_dir / name
        if path.is_file():
            data = read_json(path)
            if isinstance(data, dict):
                return data
    return {}


def summarize_from_metrics(run_dir: Path, metrics: list[dict[str, object]]) -> dict[str, object] | None:
    if not metrics:
        return None
    best_row = max(metrics, key=lambda row: float(row.get("val_foreground_mIoU", 0.0)))
    final_row = metrics[-1]
    hyperparameters = read_hyperparameters(run_dir)
    return {
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "best_val_foreground_mIoU": float(best_row.get("val_foreground_mIoU", 0.0)),
        "best_checkpoint_path": str(run_dir / "best.pt"),
        "final_epoch": int(float(final_row.get("epoch", 0))),
        "final_train_loss": float(final_row.get("train_loss", 0.0)),
        "final_val_foreground_mIoU": float(final_row.get("val_foreground_mIoU", 0.0)),
        "hyperparameters": hyperparameters,
    }


def load_run_summary(run_dir: Path) -> dict[str, object] | None:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        data = read_json(summary_path)
        if isinstance(data, dict):
            return data
    return summarize_from_metrics(run_dir, read_metrics(run_dir))


def format_float(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    args = parse_args()
    summaries = []
    for run_dir in sorted(path for path in args.outputs_root.glob("*") if path.is_dir()):
        summary = load_run_summary(run_dir)
        if summary is None:
            continue
        hparams = summary.get("hyperparameters", {})
        if not isinstance(hparams, dict):
            hparams = {}
        summaries.append(
            {
                "output_dir": str(run_dir),
                "best_mIoU": float(summary.get("best_val_foreground_mIoU", 0.0)),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "background_loss_weight": hparams.get("background_loss_weight", "-"),
                "dice_weight": hparams.get("dice_weight", "-"),
                "learning_rate": hparams.get("learning_rate", hparams.get("lr", "-")),
            }
        )

    summaries.sort(key=lambda row: row["best_mIoU"], reverse=True)
    if args.limit > 0:
        summaries = summaries[: args.limit]

    if not summaries:
        print(f"No runs with summary.json, metrics.json, or metrics.csv found under {args.outputs_root}")
        return

    print(
        f"{'rank':>4}  {'output_dir':<36}  {'best_mIoU':>10}  "
        f"{'best_epoch':>10}  {'bg_weight':>9}  {'dice_weight':>11}  {'lr':>10}"
    )
    for rank, row in enumerate(summaries, start=1):
        print(
            f"{rank:>4}  {row['output_dir']:<36}  {format_float(row['best_mIoU']):>10}  "
            f"{row['best_epoch']:>10}  {format_float(row['background_loss_weight'], 4):>9}  "
            f"{format_float(row['dice_weight'], 4):>11}  {format_float(row['learning_rate'], 6):>10}"
        )


if __name__ == "__main__":
    main()

