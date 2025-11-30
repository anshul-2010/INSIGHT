"""
Degradation-Robust Convolutional Transformer (DRCT) super-resolution backbone.

The implementation mirrors the structure-preserving variant described in Section 4.1:
 - convolutional stages ingest the low-resolution input and estimate a degradation prior
 - hybrid conv/transformer refiners propagate local + global cues while being gated by the prior
 - progressive pixel-shuffle upsampling produces hallucination-averse outputs in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class LocalConvBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)


class TinyTransformer(nn.Module):
    def __init__(self, dim: int, nhead: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = seq + self.attn(self.norm1(seq), self.norm1(seq), self.norm1(seq))[0]
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).view(b, c, h, w)


class DegradationAwareGate(nn.Module):
    """Predicts per-channel degradation priors that modulate refinement blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = torch.sigmoid(self.proj(self.estimator(x)))
        return x * prior


class DRCTStage(nn.Module):
    """One refinement stage = conv residual + transformer + feed-forward, gated by degradation prior."""

    def __init__(self, dim: int):
        super().__init__()
        self.local = LocalConvBlock(dim)
        self.trans = TinyTransformer(dim, nhead=4)
        self.ffn = ConvFFN(dim)
        self.gate = DegradationAwareGate(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        x = self.gate(x)
        x = self.trans(x)
        return self.ffn(x)


class DRCT_SR(nn.Module):
    def __init__(
        self, in_ch: int = 3, base_dim: int = 64, num_blocks: int = 4, scale: int = 4
    ):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("scale should be 2 or 4 for the current DRCT prototype.")
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([DRCTStage(base_dim) for _ in range(num_blocks)])
        self.ups_layers = nn.ModuleList()
        for _ in range(1 if scale == 2 else 2):
            self.ups_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        base_dim, base_dim * 4, kernel_size=3, padding=1, bias=False
                    ),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                )
            )
        self.tail = nn.Conv2d(base_dim, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.head(x)
        for blk in self.blocks:
            feats = feats + blk(feats)
        for up in self.ups_layers:
            feats = up(feats)
        return torch.sigmoid(self.tail(feats))


def load_drct(scale: int = 4, device: str = "cpu") -> DRCT_SR:
    """Convenience loader used by Stage 1."""

    model = DRCT_SR(scale=scale).to(device)
    return model
