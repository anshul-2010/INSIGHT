import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.shape
        s = self.fc(x).view(b,c,1,1)
        return x * s

class SpatialAttn7x7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, mx], dim=1)
        m = self.sig(self.conv(cat))
        return x * m

class SimpleHighPass(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # learnable residual high-pass
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.conv.weight, 0, 1e-3)
    def forward(self,x):
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) + self.conv(x)