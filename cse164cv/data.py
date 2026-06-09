"""Datasets and transforms for the CSE 164 competition data."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .constants import IMAGE_EXTENSIONS

try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_class_map(data_root: Path) -> list[dict[str, Any]]:
    return load_json(data_root / "metadata" / "class_map.json")


def load_train_labeled_metadata(data_root: Path) -> list[dict[str, Any]]:
    return load_json(data_root / "metadata" / "train_labeled.json")


def load_train_seg_metadata(data_root: Path) -> list[dict[str, Any]]:
    return load_json(data_root / "metadata" / "train_seg.json")


def load_val_classification(data_root: Path) -> list[dict[str, Any]]:
    return load_json(data_root / "val" / "classification.json")


def decode_rgb_mask(mask_image: Image.Image) -> np.ndarray:
    """Decode a competition RGB mask into integer segmentation ids.

    Ground truth masks store ids with:

        segmentation_id = R + G * 256

    The decoded array keeps ignore pixels as id 1000. Training loss must use
    CrossEntropyLoss(ignore_index=1000), and predictions must never emit 1000.
    """
    arr = np.asarray(mask_image.convert("RGB"), dtype=np.uint16)
    return (arr[:, :, 0].astype(np.int64) + arr[:, :, 1].astype(np.int64) * 256)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def mask_to_tensor(mask_image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(decode_rgb_mask(mask_image)).long()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def random_crop_box(width: int, height: int, scale_min: float) -> tuple[int, int, int, int]:
    """Sample a crop that preserves the source aspect ratio."""
    if not 0 < scale_min <= 1:
        raise ValueError("crop_scale_min must be in (0, 1]")
    scale = random.uniform(scale_min, 1.0)
    crop_width = max(1, round(width * scale**0.5))
    crop_height = max(1, round(height * scale**0.5))
    left = random.randint(0, max(0, width - crop_width))
    top = random.randint(0, max(0, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def apply_color_jitter(image: Image.Image, strength: float) -> Image.Image:
    """Apply mild image-only brightness, contrast, and color augmentation."""
    if strength <= 0:
        return image
    transforms = [
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ]
    random.shuffle(transforms)
    for transform in transforms:
        factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
        image = transform(image).enhance(factor)
    return image


class SegmentationTrainTransform:
    """Apply paired-safe geometry and image-only color augmentation."""

    def __init__(
        self,
        image_size: int = 256,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.0,
        crop_scale_min: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.crop_scale_min = crop_scale_min

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.crop_scale_min < 1.0:
            box = random_crop_box(image.width, image.height, self.crop_scale_min)
            image = image.crop(box)
            mask = mask.crop(box)
        size = (self.image_size, self.image_size)
        image = image.resize(size, BILINEAR)
        # Masks must use nearest-neighbor interpolation so class ids and ignore
        # pixels remain exact integers after geometric transforms.
        mask = mask.resize(size, NEAREST)
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image = apply_color_jitter(image, self.color_jitter)
        return image_to_tensor(image), mask_to_tensor(mask)


class ClassificationTrainTransform:
    """Image-only counterpart to the segmentation training augmentation."""

    def __init__(
        self,
        image_size: int = 256,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.0,
        crop_scale_min: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.crop_scale_min = crop_scale_min

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.crop_scale_min < 1.0:
            image = image.crop(random_crop_box(image.width, image.height, self.crop_scale_min))
        image = image.resize((self.image_size, self.image_size), BILINEAR)
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = apply_color_jitter(image, self.color_jitter)
        return image_to_tensor(image)


class ImageResizeTransform:
    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), BILINEAR)
        return image_to_tensor(image)


class TrainSegmentationDataset(Dataset):
    """Segmentation-labeled training data from train_seg/images and masks."""

    def __init__(
        self,
        data_root: Path,
        transform: SegmentationTrainTransform | None = None,
        max_items: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.rows = load_train_seg_metadata(self.data_root)
        if max_items is not None:
            self.rows = self.rows[:max_items]
        self.transform = transform or SegmentationTrainTransform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.data_root / row["image"]
        mask_path = self.data_root / row["mask"]
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            image, mask = self.transform(image_file.convert("RGB"), mask_file.convert("RGB"))
        return {
            "image": image,
            "mask": mask,
            "class_id": int(row["class_id"]),
            "segmentation_id": int(row["segmentation_id"]),
            "image_name": image_path.name,
        }


class TrainClassificationDataset(Dataset):
    """Image-level labeled data from train_labeled/images and metadata JSON."""

    def __init__(
        self,
        data_root: Path,
        transform: ImageResizeTransform | None = None,
        max_items: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.rows = load_train_labeled_metadata(self.data_root)
        if max_items is not None:
            self.rows = self.rows[:max_items]
        self.transform = transform or ImageResizeTransform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.data_root / row["image"]
        with Image.open(image_path) as image_file:
            image = self.transform(image_file.convert("RGB"))
        return {"image": image, "class_id": int(row["class_id"]), "image_name": image_path.name}


class ValidationSegmentationDataset(Dataset):
    """Public validation images, masks, and classification labels."""

    def __init__(
        self,
        data_root: Path,
        image_transform: ImageResizeTransform | None = None,
        max_items: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.rows = load_val_classification(self.data_root)
        if max_items is not None:
            self.rows = self.rows[:max_items]
        self.image_transform = image_transform or ImageResizeTransform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_name = row["image"]
        image_path = self.data_root / "val" / "images" / image_name
        mask_path = self.data_root / "val" / "masks" / Path(image_name).with_suffix(".png").name
        with Image.open(image_path) as image_file:
            original_width, original_height = image_file.size
            image = self.image_transform(image_file.convert("RGB"))
        with Image.open(mask_path) as mask_file:
            mask = mask_to_tensor(mask_file.convert("RGB"))
        return {
            "image": image,
            "mask": mask,
            "class_id": int(row["class_id"]),
            "image_name": image_name,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.long),
        }


class ImageOnlyDataset(Dataset):
    """Image-only splits: train_unlabeled/images, val/images, or test/images."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        transform: ImageResizeTransform | None = None,
        max_items: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        if split not in {"train_unlabeled", "val", "test"}:
            raise ValueError("split must be one of train_unlabeled, val, or test")
        self.split = split
        self.image_dir = self.data_root / split / "images"
        self.image_paths = list_images(self.image_dir)
        if max_items is not None:
            self.image_paths = self.image_paths[:max_items]
        self.transform = transform or ImageResizeTransform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image_file:
            original_width, original_height = image_file.size
            image = self.transform(image_file.convert("RGB"))
        return {
            "image": image,
            "image_name": image_path.name,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.long),
        }
