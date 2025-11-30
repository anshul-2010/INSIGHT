"""Module providing helper functions for adversarial training.

This wraps the PGD attack inside the training loop for simple adversarial training.
"""
import torch

try:  # pragma: no cover
    from .attacks import pgd_attack  # type: ignore
    from .utils import to_device  # type: ignore
except ImportError:  # pragma: no cover
    from attacks import pgd_attack
    from utils import to_device


def adv_train_epoch(model, loader, optim, device, eps=0.03, alpha=0.007, iters=7):
    """Perform one epoch of adversarial training using PGD.

    The model receives adversarial examples generated on-the-fly using the PGD inner loop.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = to_device((imgs, labels), device)
        # generate adversarial examples for current batch
        x_adv = pgd_attack(model, imgs, labels, eps=eps, alpha=alpha, iters=iters)
        logits, _ = model(x_adv)
        loss = criterion(logits, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * imgs.size(0)
        total_acc += (logits.argmax(1) == labels).float().sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)
