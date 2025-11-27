# Lightweight script sketch for contrastive pretraining using SimCLR skeleton.
import torch
from torch import optim
from torch.utils.data import DataLoader

try:  # pragma: no cover
    from .insight_dataset import InsightDataset  # type: ignore
    from .models.contrastive_simclr import ProjectionHead, SimCLRTrainer  # type: ignore
    from .models.resnet_like import ResNetSmall  # type: ignore
except ImportError:  # pragma: no cover
    from insight_dataset import InsightDataset
    from models.contrastive_simclr import ProjectionHead, SimCLRTrainer
    from models.resnet_like import ResNetSmall

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = ResNetSmall()  # default produces (logits, feat) where feat is 128-d
    proj = ProjectionHead(in_dim=128, out_dim=64)
    trainer = SimCLRTrainer(enc, proj, device=device)
    # For SimCLR we need two augmented views per sample - use augment_twice flag
    ds = InsightDataset('data', split='train', img_size=32, augment_twice=True)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    optimizer = optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)
    for epoch in range(10):
        for batch in loader:
            (x1, x2), _ = batch
            loss = trainer.train_step(x1, x2, optimizer)
        print(f"Epoch {epoch} SimCLR pretrain loss: {loss:.4f}")

if __name__=='__main__':
    main()
