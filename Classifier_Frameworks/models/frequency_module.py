import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # local import fallback
    from ..utils import compute_dct_batch  # type: ignore
except ImportError:  # pragma: no cover
    from utils import compute_dct_batch


class DCTBackbone(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Accepts torch tensor, compute DCT coefficients on CPU for now
        dct_x = (
            compute_dct_batch(x)
            if x.is_cuda == False
            else compute_dct_batch(x.cpu()).to(x.device)
        )
        feat = torch.flatten(self.conv(dct_x), 1)
        logits = self.fc(feat)
        return logits, feat
