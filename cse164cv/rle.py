"""Run-length encoding helpers for Kaggle submissions."""

from __future__ import annotations

import numpy as np

from .constants import NUM_CLASSES


def encode_segmentation_rle(mask: np.ndarray) -> str:
    """Encode a segmentation-id mask as row-major 1-indexed RLE triples.

    Only foreground ids are written. Each run is:

        start length value

    where `start` is 1-indexed after row-major flattening and `value` must be
    a foreground segmentation id in 1..300. Background id 0 is omitted. The
    competition expects an all-background prediction to be encoded as "0".
    """
    flat = np.asarray(mask, dtype=np.int64).reshape(-1)
    if np.any(flat < 0) or np.any(flat > NUM_CLASSES):
        raise ValueError("Predicted masks may contain only ids 0..300")

    foreground = flat != 0
    if not np.any(foreground):
        return "0"

    idx = np.flatnonzero(foreground)
    values = flat[idx]
    if np.any(values < 1) or np.any(values > NUM_CLASSES):
        raise ValueError("RLE values must be foreground segmentation ids 1..300")

    run_break = np.ones(len(idx), dtype=bool)
    run_break[1:] = (idx[1:] != idx[:-1] + 1) | (values[1:] != values[:-1])
    starts = np.flatnonzero(run_break)
    ends = np.r_[starts[1:], len(idx)]

    parts: list[str] = []
    for start_pos, end_pos in zip(starts, ends):
        start = int(idx[start_pos]) + 1
        length = int(end_pos - start_pos)
        value = int(values[start_pos])
        parts.extend((str(start), str(length), str(value)))
    return " ".join(parts)

