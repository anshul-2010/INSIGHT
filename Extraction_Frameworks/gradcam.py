"""
GradCAM utilities for attention-guided artifact localization.

Hook usage:
    gradcam = GradCAM(model, target_layer=auto_select_last_conv(model))
    cam = gradcam.compute_cam(input_tensor, class_idx=1, upsample_to=(Hs, Ws))
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, out):
            self.activations = out
            out.register_hook(self._save_gradients)

        self.target_layer.register_forward_hook(forward_hook)

    def _save_gradients(self, grad):
        self.gradients = grad

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        upsample_to: Optional[Tuple[int, int]] = None,
        retain_graph: bool = False,
    ) -> np.ndarray:
        """
        Args:
            input_tensor: tensor shaped [1, C, H, W].
            class_idx: optional target class; inferred from logits argmax if None.
            upsample_to: optional spatial size for final CAM (H, W).
            retain_graph: keep computation graph for repeated calls.
        """

        device = next(self.model.parameters()).device
        logits = self.model(input_tensor.to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if class_idx is None:
            probs = F.softmax(logits, dim=1)
            class_idx = int(probs.argmax(dim=1).item())

        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=retain_graph)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "GradCAM hooks failed. Ensure the target layer received activations."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_tensor = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = cam_tensor[0, 0].detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-9)
        if upsample_to is not None:
            cam = cv2.resize(
                cam, (upsample_to[1], upsample_to[0]), interpolation=cv2.INTER_LINEAR
            )
            cam = np.clip(cam, 0.0, 1.0)
        return cam


def auto_select_last_conv(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer in the model to serve as a GradCAM target."""

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(
            "GradCAM target layer could not be inferred; pass it explicitly."
        )
    return last_conv


def overlay_heatmap(
    image: np.ndarray, cam: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Utility for visualization: overlays the CAM on top of an RGB image."""

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(overlay, 1 - alpha, heatmap, alpha, 0)
    return blended
