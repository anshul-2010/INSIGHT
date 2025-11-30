"""
Attention-weighted superpixel-to-patch decomposition (Section 4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PatchDescriptor:
    patch_img: np.ndarray
    weight: float
    sp_id: int
    coords: Tuple[int, int, int, int]
    activation: float


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def subdivide_bbox(
    bbox: Tuple[int, int, int, int], patch_size: int, stride: Optional[int] = None
):
    y1, y2, x1, x2 = bbox
    stride = stride or patch_size
    patches = []
    max_y = y2 - patch_size + 1
    max_x = x2 - patch_size + 1
    if max_y < y1 or max_x < x1:
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ph = patch_size // 2
        yy1 = max(y1, cy - ph)
        xx1 = max(x1, cx - ph)
        yy2 = min(y2, yy1 + patch_size - 1)
        xx2 = min(x2, xx1 + patch_size - 1)
        return [(yy1, yy2 + 1, xx1, xx2 + 1)]
    for yy in range(y1, max_y + 1, stride):
        for xx in range(x1, max_x + 1, stride):
            patches.append((yy, yy + patch_size, xx, xx + patch_size))
    return patches


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def extract_attention_weighted_patches(
    rgb_img: np.ndarray,
    gradcam: np.ndarray,
    superpixel_masks: Dict[int, np.ndarray],
    patch_size: int = 32,
    tau: float = 8.0,
    stride: Optional[int] = None,
    min_area: int = 10,
    parent_temp: float = 5.0,
    parent_bias: float = 0.5,
) -> List[PatchDescriptor]:
    """
    Args:
        rgb_img: H x W x 3 float image in [0, 1]
        gradcam: H x W activation map in [0, 1]
        superpixel_masks: dict {id: mask}
    Returns:
        List of PatchDescriptor objects sorted by superpixel traversal order.
    """

    eps = 1e-9
    results: List[PatchDescriptor] = []

    for spid, mask in superpixel_masks.items():
        if mask.sum() < min_area:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        Ask = float((gradcam * mask).sum() / (mask.sum() + eps))
        patches_coords = subdivide_bbox(bbox, patch_size, stride=stride)
        if not patches_coords:
            continue

        raw_scores: List[float] = []
        patch_meta: List[Tuple[int, int, int, int]] = []
        for py1, py2, px1, px2 in patches_coords:
            pmask = mask[py1:py2, px1:px2]
            if pmask.size == 0:
                continue
            raw_scores.append(float((gradcam[py1:py2, px1:px2] * pmask).sum()))
            patch_meta.append((py1, py2, px1, px2))
        if len(raw_scores) == 0:
            continue

        raw = np.array(raw_scores, dtype=np.float32)
        if raw.sum() == 0:
            probs = np.ones_like(raw) / len(raw)
        else:
            scaled = raw * tau
            exp_scores = np.exp(scaled - scaled.max())
            probs = exp_scores / (exp_scores.sum() + eps)

        sigma_ask = float(sigmoid(parent_temp * (Ask - parent_bias)))

        for idx, (py1, py2, px1, px2) in enumerate(patch_meta):
            weight = float(probs[idx]) * sigma_ask
            patch_img = rgb_img[py1:py2, px1:px2, :].copy()
            results.append(
                PatchDescriptor(
                    patch_img=patch_img,
                    weight=weight,
                    sp_id=int(spid),
                    coords=(py1, py2, px1, px2),
                    activation=float(raw[idx]),
                )
            )

    total_weight = sum(p.weight for p in results) + eps
    for patch in results:
        patch.weight /= total_weight
    return results
