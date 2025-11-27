import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEnsemble(nn.Module):
    """A simple ensemble that averages logits (not probabilities) and concatenates features.

    Important: We average logits instead of probabilities because the CrossEntropyLoss expects
    raw logits (it applies log-softmax internally). Returning probabilities would break training.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = []
        feats = []
        for m in self.models:
            l, f = m(x)
            logits.append(l)
            feats.append(f)
        avg_logits = torch.stack(logits, dim=0).mean(dim=0)  # mean across models
        fused_feat = torch.cat(feats, dim=1)
        return avg_logits, fused_feat

class StackingMeta(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.meta = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, stacked_feats):
        """stacked_feats is expected with shape [B, in_dim]."""
        return self.meta(stacked_feats)