"""
Superpixel-aware region proposals powering the hierarchical voting system.

Outputs a `SuperpixelHierarchy` that stores:
 - dense label maps per granularity
 - binary masks per region for quick aggregation
 - parent-child relationships between consecutive levels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import slic


@dataclass
class SuperpixelHierarchy:
    levels: Tuple[int, ...]
    segmentations: Dict[int, np.ndarray]
    masks: Dict[int, Dict[int, np.ndarray]]
    parent_map: Dict[Tuple[int, int], Dict[int, int]]

    def finest_level(self) -> int:
        return max(self.levels)

    def get_masks(self, level: int) -> Dict[int, np.ndarray]:
        return self.masks[level]


def slic_segments(
    rgb_img: np.ndarray, n_segments: int = 150, compactness: float = 10.0
) -> np.ndarray:
    lab = rgb2lab(rgb_img)
    return slic(lab, n_segments=n_segments, compactness=compactness, start_label=0)


def masks_from_segments(segments: np.ndarray) -> Dict[int, np.ndarray]:
    masks = {}
    for sid in np.unique(segments):
        masks[int(sid)] = (segments == sid).astype(np.float32)
    return masks


def parent_child_map(parent_seg: np.ndarray, child_seg: np.ndarray) -> Dict[int, int]:
    mapping = {}
    for cid in np.unique(child_seg):
        mask = child_seg == cid
        parent_ids, counts = np.unique(parent_seg[mask], return_counts=True)
        mapping[int(cid)] = (
            int(parent_ids[np.argmax(counts)]) if len(parent_ids) else -1
        )
    return mapping


def activation_per_region(
    heatmap: np.ndarray, masks: Dict[int, np.ndarray]
) -> Dict[int, float]:
    eps = 1e-9
    activations = {}
    for sid, mask in masks.items():
        denom = mask.sum() + eps
        activations[int(sid)] = float((heatmap * mask).sum() / denom)
    return activations


def build_hierarchy(
    rgb_img: np.ndarray,
    levels: Iterable[int] = (50, 150, 300),
    compactness: float = 10.0,
) -> SuperpixelHierarchy:
    segs = {}
    masks = {}
    parent_maps = {}
    sorted_levels = tuple(sorted(levels))
    prev_level = None
    for level in sorted_levels:
        seg = slic_segments(rgb_img, n_segments=level, compactness=compactness)
        segs[level] = seg
        masks[level] = masks_from_segments(seg)
        if prev_level is not None:
            parent_maps[(prev_level, level)] = parent_child_map(segs[prev_level], seg)
        prev_level = level
    return SuperpixelHierarchy(sorted_levels, segs, masks, parent_maps)
