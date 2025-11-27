"""Unified training script for INSIGHT classifier research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:  # pragma: no cover
    from .attacks import pgd_attack  # type: ignore
    from .ensemble import SimpleEnsemble  # type: ignore
    from .insight_dataset import InsightDataset  # type: ignore
    from .models import (
        AutoencoderDetector,
        DCTBackbone,
        HybridBackbone,
        ResNetSmall,
        SimpleBNN,
        SmallCNN,
        TinyViT,
    )
    from .utils import to_device
except ImportError:  # pragma: no cover
    from attacks import pgd_attack
    from ensemble import SimpleEnsemble
    from insight_dataset import InsightDataset
    from models import (
        AutoencoderDetector,
        DCTBackbone,
        HybridBackbone,
        ResNetSmall,
        SimpleBNN,
        SmallCNN,
        TinyViT,
    )
    from utils import to_device


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n


def unpack_model_outputs(output):
    if isinstance(output, tuple):
        logits = output[0]
        features = output[1] if len(output) > 1 else None
        extras = output[2:] if len(output) > 2 else ()
    else:
        logits = output
        features = None
        extras = ()
    return logits, features, extras


def build_single_model(arch: str, num_classes: int, latent_dim: int):
    if arch == "resnet":
        return ResNetSmall(num_classes=num_classes)
    if arch == "cnn":
        return SmallCNN(num_classes=num_classes)
    if arch == "vit":
        return TinyViT(num_classes=num_classes)
    if arch == "dct":
        return DCTBackbone(num_classes=num_classes)
    if arch == "bnn":
        return SimpleBNN(num_classes=num_classes)
    if arch == "autoencoder":
        return AutoencoderDetector(latent_dim=latent_dim, num_classes=num_classes)
    if arch == "hybrid":
        return HybridBackbone(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def maybe_build_ensemble(cfg):
    if not cfg.ensemble_archs:
        return build_single_model(cfg.arch, cfg.num_classes, cfg.latent_dim)
    members = [build_single_model(name, cfg.num_classes, cfg.latent_dim) for name in cfg.ensemble_archs]
    return SimpleEnsemble(members)


def maybe_adv_examples(cfg, model, imgs, labels):
    if not cfg.adv:
        return imgs
    if cfg.adv_prob > 0 and torch.rand(1).item() > cfg.adv_prob:
        return imgs
    return pgd_attack(model, imgs, labels, eps=cfg.adv_eps, alpha=cfg.adv_step, iters=cfg.adv_iters)


def train_epoch(model, loader, optimizer, device, cfg, scaler=None):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="train", leave=False):
        imgs, labels = to_device(batch, device)
        imgs = maybe_adv_examples(cfg, model, imgs, labels)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            logits, _, _ = unpack_model_outputs(model(imgs))
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        preds = logits.argmax(1)
        loss_meter.update(loss.item(), imgs.size(0))
        acc_meter.update((preds == labels).float().sum().item(), imgs.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        imgs, labels = to_device(batch, device)
        logits, _, _ = unpack_model_outputs(model(imgs))
        loss = criterion(logits, labels)
        preds = logits.argmax(1)
        loss_meter.update(loss.item(), imgs.size(0))
        acc_meter.update((preds == labels).float().sum().item(), imgs.size(0))
    return loss_meter.avg, acc_meter.avg


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_acc: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_acc", 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description="INSIGHT low-res classifier training")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--arch", type=str, default="hybrid", choices=["hybrid", "resnet", "cnn", "vit", "dct", "autoencoder", "bnn"])
    parser.add_argument("--ensemble-archs", type=str, nargs="*", help="Optional list of backbones to ensemble")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=2, dest="num_classes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent-dim", type=int, default=128, dest="latent_dim")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--adv", action="store_true", help="Enable adversarial training (PGD)")
    parser.add_argument("--adv-prob", type=float, default=1.0, help="Probability of applying adversarial augmentation per batch")
    parser.add_argument("--adv-eps", type=float, default=0.02)
    parser.add_argument("--adv-step", type=float, default=0.005)
    parser.add_argument("--adv-iters", type=int, default=7)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_ds = InsightDataset(cfg.data, split="train", img_size=32)
    val_ds = InsightDataset(cfg.data, split="val", img_size=32)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = maybe_build_ensemble(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=cfg.amp)

    start_epoch = 0
    best_val = 0.0
    if cfg.resume:
        ckpt_path = Path(cfg.resume)
        start_epoch, best_val = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch} (best_val={best_val:.3f})")

    patience_counter = 0
    history = []

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, cfg, scaler if cfg.amp else None)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_val)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("Early stopping triggered.")
                break

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()