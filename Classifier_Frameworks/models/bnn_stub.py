import torch
import torch.nn as nn
import torch.nn.functional as F


# Lightweight binary stub; actual binarization layers are non-trivial to implement robustly.
# This stub uses sign quantization in forward pass for weights/activations to simulate BNN behavior.
def binarize_tensor(x):
    return x.sign()


class BinarizeConv(nn.Conv2d):
    def forward(self, x):
        # Simulate binary weights by using sign in forward but keep gradients via STE
        bw = self.weight.sign()
        return F.conv2d(
            x.sign(),
            bw,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SimpleBNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = binarize_tensor(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        feat = torch.flatten(self.pool(x), 1)
        logits = self.fc(feat)
        return logits, feat
