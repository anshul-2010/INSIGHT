import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=128, depth=4, heads=4, mlp_dim=256, num_classes=2):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size)**2
        patch_dim = in_ch * patch_size * patch_size
        # simple linear patch embedding via conv
        self.to_patch = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(dim, heads, batch_first=True),
                'mlp': nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)),
                'ln1': nn.LayerNorm(dim),
                'ln2': nn.LayerNorm(dim)
            }) for _ in range(depth)
        ])
        self.to_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self,x):
        # x: [B,C,H,W]
        B = x.shape[0]
        # produce patch embeddings
        proj = self.to_patch(x)  # [B, dim, N]
        proj = proj.flatten(2).transpose(1,2)  # [B, N, dim]
        proj = proj + self.pos_emb
        cls = self.cls_token.expand(B,-1,-1)
        tokens = torch.cat([cls, proj], dim=1)
        for block in self.transformer:
            x_ = block['ln1'](tokens)
            attn_out,_ = block['attn'](x_, x_, x_)
            tokens = tokens + attn_out
            tokens = tokens + block['mlp'](block['ln2'](tokens))
        cls_tok = tokens[:,0]
        logits = self.to_head(cls_tok)
        return logits, cls_tok