import torch
import numpy as np
from scipy.fftpack import dct


def to_device(batch, device):
    """Move batch (imgs, labels) to device.

    Accepts either a tuple (imgs, labels) or nested ((img1,img2), labels).
    """
    imgs, labels = batch
    if isinstance(imgs, (list, tuple)):
        imgs = tuple(i.to(device) for i in imgs)
    else:
        imgs = imgs.to(device)
    return imgs, labels.to(device)


def topk_accuracy(logits, labels, k=1):
    _, idx = logits.topk(k, dim=1)
    correct = idx.eq(labels.view(-1, 1).expand_as(idx))
    return float(correct[:, :k].any(dim=1).float().mean().item())


def compute_dct_batch(x: torch.Tensor) -> torch.Tensor:
    """Compute a 2D DCT per channel, preferring torch.fft for GPU support."""

    if hasattr(torch.fft, "dct"):
        # Apply separable DCT over the last two spatial dims
        dct_h = torch.fft.dct(x, type=2, norm="ortho", dim=-1)
        dct_hw = torch.fft.dct(dct_h, type=2, norm="ortho", dim=-2)
        return dct_hw

    return _compute_dct_numpy(x)


def _compute_dct_numpy(x: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    B, C, H, W = x_np.shape
    out = np.zeros_like(x_np)
    for b in range(B):
        for c in range(C):
            out[b, c, :, :] = dct(
                dct(x_np[b, c, :, :], axis=0, norm="ortho"), axis=1, norm="ortho"
            )
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def l2_normalize(x, dim=1, eps=1e-10):
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)
