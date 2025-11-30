import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # allow usage both as package and standalone script
    from ..blocks import SEBlock, SpatialAttn7x7, SimpleHighPass  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed from repo root
    from blocks import SEBlock, SpatialAttn7x7, SimpleHighPass


class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, width=32, use_se=True, use_spatial=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.highpass = SimpleHighPass(width * 4)
        self.use_se = use_se
        self.use_spatial = use_spatial
        if use_se:
            self.se = SEBlock(width * 4)
        if use_spatial:
            self.sa = SpatialAttn7x7()
        # split head to expose features before final linear layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(width * 4, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.highpass(x)
        if self.use_se:
            x = self.se(x)
        if self.use_spatial:
            x = self.sa(x)
        pooled = self.pool(x)
        feat = self.flatten(pooled)
        embed = self.relu(self.fc1(feat))
        logits = self.fc2(embed)
        return logits, feat
