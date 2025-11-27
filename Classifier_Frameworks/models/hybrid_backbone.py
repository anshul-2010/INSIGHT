"""Hybrid backbone combining spatial CNN/ResNet cues with frequency reasoning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - allow running inside/outside package
    from ..blocks import SEBlock, SimpleHighPass, SpatialAttn7x7  # type: ignore
    from ..utils import compute_dct_batch  # type: ignore
    from .resnet_like import BasicBlock  # type: ignore
except ImportError:  # pragma: no cover
    from blocks import SEBlock, SimpleHighPass, SpatialAttn7x7
    from utils import compute_dct_batch
    from models.resnet_like import BasicBlock


class SpatialEncoder(nn.Module):
    """Compact ResNet-style encoder tuned for 32x32 spatial reasoning."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.in_ch = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.high_pass = SimpleHighPass(base_channels * 4)
        self.se = SEBlock(base_channels * 4)
        self.sa = SpatialAttn7x7()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, out_ch: int, blocks: int, stride: int):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_ch, out_ch, stride=s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_map = self.layer3(x)
        feat_map = self.high_pass(feat_map)
        feat_map = self.se(feat_map)
        feat_map = self.sa(feat_map)
        vec = torch.flatten(self.pool(feat_map), 1)
        return feat_map, vec


class FrequencyEncoder(nn.Module):
    """DCT-driven encoder that captures aliasing and spectral artifacts."""

    def __init__(self, in_ch: int = 3, hidden: int = 48):
        super().__init__()
        conv_blocks = [
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor):
        dct_x = compute_dct_batch(x)
        feat_map = self.conv(dct_x)
        vec = torch.flatten(self.pool(feat_map), 1)
        return feat_map, vec


class HybridBackbone(nn.Module):
    """Final forensic classifier used throughout INSIGHT."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 256, spatial_base: int = 32):
        super().__init__()
        self.spatial = SpatialEncoder(base_channels=spatial_base)
        self.frequency = FrequencyEncoder()
        fused_dim = spatial_base * 4 + self.frequency.out_dim
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        spatial_map, spatial_vec = self.spatial(x)
        freq_map, freq_vec = self.frequency(x)
        fused = torch.cat([spatial_vec, freq_vec], dim=1)
        embed = self.proj(fused)
        logits = self.classifier(embed)
        # Provide GradCAM-friendly activation map (spatial branch) as auxiliary output
        return logits, embed, {"spatial_map": spatial_map, "frequency_map": freq_map}

