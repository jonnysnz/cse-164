#!/usr/bin/env python3
"""Inspect the local CSE 164 data release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cse164cv.data import (  # noqa: E402
    decode_rgb_mask,
    list_images,
    load_class_map,
    load_train_labeled_metadata,
    load_train_seg_metadata,
    load_val_classification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=3)
    return parser.parse_args()


def count_files(data_root: Path) -> None:
    dirs = [
        "train_seg/images",
        "train_seg/masks",
        "train_labeled/images",
        "train_unlabeled/images",
        "val/images",
        "val/masks",
        "test/images",
    ]
    print("File counts:")
    for rel in dirs:
        path = data_root / rel
        count = sum(1 for child in path.iterdir() if child.is_file())
        print(f"  {rel}: {count}")


def check_metadata(data_root: Path) -> None:
    class_map = load_class_map(data_root)
    train_labeled = load_train_labeled_metadata(data_root)
    train_seg = load_train_seg_metadata(data_root)
    val_rows = load_val_classification(data_root)

    print("\nMetadata rows:")
    print(f"  class_map.json: {len(class_map)}")
    print(f"  train_labeled.json: {len(train_labeled)}")
    print(f"  train_seg.json: {len(train_seg)}")
    print(f"  val/classification.json: {len(val_rows)}")

    missing = 0
    bad_labels = 0
    for row in train_labeled:
        missing += not (data_root / row["image"]).is_file()
        bad_labels += not (0 <= int(row["class_id"]) <= 299)
    for row in train_seg:
        missing += not (data_root / row["image"]).is_file()
        missing += not (data_root / row["mask"]).is_file()
        bad_labels += not (0 <= int(row["class_id"]) <= 299)
        bad_labels += int(row["segmentation_id"]) != int(row["class_id"]) + 1
    for row in val_rows:
        image_path = data_root / "val" / "images" / row["image"]
        mask_path = data_root / "val" / "masks" / Path(row["image"]).with_suffix(".png").name
        missing += not image_path.is_file()
        missing += not mask_path.is_file()
        bad_labels += not (0 <= int(row["class_id"]) <= 299)

    print("\nMetadata consistency:")
    print(f"  missing referenced files: {missing}")
    print(f"  invalid labels or segmentation ids: {bad_labels}")


def inspect_masks(data_root: Path, samples: int) -> None:
    image_paths = list_images(data_root / "train_seg" / "images")[:samples]
    print("\nSample train_seg masks:")
    for image_path in image_paths:
        mask_path = data_root / "train_seg" / "masks" / image_path.with_suffix(".png").name
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            ids = decode_rgb_mask(mask_file)
            values = np.unique(ids)
            print(
                f"  {image_path.name}: image={image_file.size}, mask={mask_file.size}, "
                f"unique={len(values)}, min={int(values.min())}, max={int(values.max())}, "
                f"has_ignore={bool(np.any(ids == 1000))}, first_values={values[:12].tolist()}"
            )


def main() -> None:
    args = parse_args()
    count_files(args.data_root)
    check_metadata(args.data_root)
    inspect_masks(args.data_root, args.samples)


if __name__ == "__main__":
    main()

