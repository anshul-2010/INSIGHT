"""
Stage-1 workflow: DRCT super-resolution + GradCAM localization + superpixel-aware patch extraction.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from drct_sr import load_drct
from gradcam import GradCAM, auto_select_last_conv
from patch_extractor import PatchDescriptor, extract_attention_weighted_patches
from superpixel import SuperpixelHierarchy, activation_per_region, build_hierarchy
from utils_image import read_image_bgr, resize_numpy, to_tensor


@dataclass
class StageOneOutputs:
    lr_img: np.ndarray
    sr_img: np.ndarray
    gradcam: np.ndarray
    hierarchy: SuperpixelHierarchy
    superpixel_activations: Dict[int, float]
    patches: Sequence[PatchDescriptor]


class DummyArtifactClassifier(nn.Module):
    """Fallback classifier so the pipeline can run even without pretrained weights."""

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        feat = self.feature(x)
        pooled = self.pool(feat).flatten(1)
        return self.fc(pooled)


def _prepare_lr_tensor(img: np.ndarray, target_hw=(32, 32), device="cpu") -> torch.Tensor:
    lr_img = resize_numpy(img, target_hw)
    return to_tensor(lr_img).to(device), lr_img


def run_stage1(
    image_path: str,
    device: str = "cpu",
    scale: int = 4,
    sp_levels: Iterable[int] = (150,),
    patch_size: int = 32,
    tau: float = 8.0,
    classifier: Optional[nn.Module] = None,
    target_layer: Optional[nn.Module] = None,
    class_idx: int = 1,
) -> StageOneOutputs:
    img = read_image_bgr(image_path, to_rgb=True)
    lr_tensor, lr_img = _prepare_lr_tensor(img, target_hw=(32, 32), device=device)

    sr_model = load_drct(scale=scale, device=device).eval()
    with torch.no_grad():
        sr_out = sr_model(lr_tensor)
    sr_tensor = sr_out.detach().clone().requires_grad_(True)
    sr_float = sr_out.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

    classifier = classifier.to(device) if classifier is not None else DummyArtifactClassifier().to(device)
    classifier.eval()
    target_layer = target_layer or auto_select_last_conv(classifier)
    gradcam = GradCAM(classifier, target_layer)
    cam = gradcam.compute_cam(sr_tensor.to(device), class_idx=class_idx, upsample_to=sr_float.shape[:2])

    hierarchy = build_hierarchy(sr_float, levels=sp_levels)
    finest_masks = hierarchy.get_masks(hierarchy.finest_level())
    activations = activation_per_region(cam, finest_masks)

    patches = extract_attention_weighted_patches(
        rgb_img=sr_float,
        gradcam=cam,
        superpixel_masks=finest_masks,
        patch_size=patch_size,
        tau=tau,
        stride=patch_size // 2,
    )

    return StageOneOutputs(
        lr_img=lr_img,
        sr_img=sr_float,
        gradcam=cam,
        hierarchy=hierarchy,
        superpixel_activations=activations,
        patches=patches,
    )


def save_stage1_visuals(outputs: StageOneOutputs, out_dir: str = "stage1_out", max_patches: int = 10):
    os.makedirs(out_dir, exist_ok=True)
    lr = (outputs.lr_img * 255).astype("uint8")
    sr = (outputs.sr_img * 255).astype("uint8")
    cam_vis = (outputs.gradcam * 255).astype("uint8")
    cv2.imwrite(os.path.join(out_dir, "lr.png"), cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "sr.png"), cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "cam.png"), cam_vis)
    for i, patch in enumerate(outputs.patches[:max_patches]):
        patch_img = (patch.patch_img * 255).astype("uint8")
        cv2.imwrite(
            os.path.join(out_dir, f"patch_{i}_w{patch.weight:.4f}.png"),
            cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=8.0)
    parser.add_argument("--sp_levels", type=int, nargs="+", default=[150])
    args = parser.parse_args()

    outputs = run_stage1(
        image_path=args.image,
        device=args.device,
        scale=args.scale,
        sp_levels=tuple(args.sp_levels),
        patch_size=args.patch_size,
        tau=args.tau,
    )
    save_stage1_visuals(outputs)
    print(f"Saved outputs to stage1_out. Extracted {len(outputs.patches)} patches.")
