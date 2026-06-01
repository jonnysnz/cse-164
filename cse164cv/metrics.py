"""Validation metrics for segmentation baselines."""

from __future__ import annotations

import numpy as np
import torch

from .constants import IGNORE_INDEX, NUM_CLASSES


def confusion_matrix_for_masks(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Build a confusion matrix over ids 0..300 while ignoring target 1000."""
    pred = np.asarray(pred, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    pred = np.where((pred >= 0) & (pred <= num_classes), pred, 0)

    valid = target != ignore_index
    valid &= target >= 0
    valid &= target <= num_classes
    labels = (num_classes + 1) * target[valid] + pred[valid]
    hist = np.bincount(labels, minlength=(num_classes + 1) ** 2)
    return hist.reshape(num_classes + 1, num_classes + 1)


def foreground_miou(hist: np.ndarray) -> tuple[float, dict[int, float]]:
    """Compute mean IoU over foreground segmentation ids 1..300.

    Background id 0 is deliberately excluded. Classes absent from the current
    validation split are also excluded from the average, matching the starter
    metric behavior.
    """
    per_class: dict[int, float] = {}
    for class_id in range(1, NUM_CLASSES + 1):
        true_positive = hist[class_id, class_id]
        false_positive = hist[:, class_id].sum() - true_positive
        false_negative = hist[class_id, :].sum() - true_positive
        denom = true_positive + false_positive + false_negative
        if denom > 0:
            per_class[class_id] = float(true_positive / denom)
    if not per_class:
        return 0.0, per_class
    return float(np.mean(list(per_class.values()))), per_class


@torch.no_grad()
def update_hist_from_logits(
    hist: np.ndarray,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    """Upsample logits to target size, argmax, and update the mIoU histogram."""
    if logits.shape[-2:] != target.shape[-2:]:
        logits = torch.nn.functional.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pred = logits.argmax(dim=1).cpu().numpy()
    target_np = target.cpu().numpy()
    for pred_mask, target_mask in zip(pred, target_np):
        hist += confusion_matrix_for_masks(pred_mask, target_mask)
    return hist

