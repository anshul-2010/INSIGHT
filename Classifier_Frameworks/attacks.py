"""Collection of white-box adversarial attacks used for INSIGHT robustness studies.

The implementations are adapted from canonical open-source references (Madry Lab PGD,
Kurakin et al. FGSM/BIM, Moosavi-Dezfooli et al. DeepFool, Carlini & Wagner L2) and are
kept lightweight so they can run directly inside research notebooks without extra deps.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:  # optional dependency for AutoAttack
    from autoattack import AutoAttack  # type: ignore
except Exception:  # pragma: no cover
    AutoAttack = None


__all__ = [
    "fgsm_attack",
    "bim_attack",
    "pgd_attack",
    "mifgsm_attack",
    "deepfool_attack",
    "carlini_wagner_l2",
    "autoattack_runner",
]


def _clamp(x: torch.Tensor, lower: float = 0.0, upper: float = 1.0) -> torch.Tensor:
    return torch.max(torch.min(x, torch.tensor(upper, device=x.device)), torch.tensor(lower, device=x.device))


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.01,
    loss_fn: Callable = F.cross_entropy,
) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al.)."""

    x_adv = x.clone().detach().requires_grad_(True)
    logits, *_ = model(x_adv)
    loss = loss_fn(logits, y)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return _clamp(x_adv.detach())


def bim_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    alpha: float = 0.005,
    iters: int = 5,
    loss_fn: Callable = F.cross_entropy,
) -> torch.Tensor:
    """Basic Iterative Method (a.k.a. projected FGSM)."""

    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits, *_ = model(x_adv)
        loss = loss_fn(logits, y)
        loss.backward()
        grad = x_adv.grad.sign()
        x_adv = x_adv + alpha * grad
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = _clamp(x + delta).detach()
    return x_adv


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    alpha: float = 0.007,
    iters: int = 10,
    restarts: int = 1,
    loss_fn: Callable = F.cross_entropy,
) -> torch.Tensor:
    """Madry et al. Projected Gradient Descent with multiple restarts."""

    best_adv = None
    best_loss = None
    for _ in range(restarts):
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)
        for _ in range(iters):
            x_adv.requires_grad_(True)
            logits, *_ = model(x_adv)
            loss = loss_fn(logits, y)
            loss.backward()
            grad = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = _clamp(x + delta).detach()
        with torch.no_grad():
            logits, *_ = model(x_adv)
            loss = loss_fn(logits, y)
            if best_loss is None or loss > best_loss:
                best_loss = loss
                best_adv = x_adv
    return best_adv if best_adv is not None else x


def mifgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    alpha: float = 0.005,
    iters: int = 10,
    decay: float = 1.0,
    loss_fn: Callable = F.cross_entropy,
) -> torch.Tensor:
    """Momentum Iterative FGSM (Dong et al., CVPR'18)."""

    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)
    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits, *_ = model(x_adv)
        loss = loss_fn(logits, y)
        loss.backward()
        grad = x_adv.grad / (torch.mean(torch.abs(x_adv.grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = decay * momentum + grad
        x_adv = x_adv + alpha * momentum.sign()
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = _clamp(x + delta).detach()
    return x_adv


def deepfool_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    max_iter: int = 20,
    overshoot: float = 0.02,
) -> torch.Tensor:
    """DeepFool (Moosavi-Dezfooli et al.). Works for binary or multi-class classifiers."""

    x_adv = x.clone().detach()
    batch = x_adv.shape[0]
    logits, *_ = model(x_adv)
    num_classes = logits.shape[1]

    for b in range(batch):
        xi = x_adv[b : b + 1].detach()
        r_tot = torch.zeros_like(xi)
        for _ in range(max_iter):
            xi.requires_grad_(True)
            logits_i, *_ = model(xi)
            if y is None:
                target = logits_i.argmax(dim=1)
            else:
                target = y[b : b + 1]
            target_idx = target.item()
            if logits_i.argmax().item() != target_idx:
                break
            gradients = []
            for k in range(num_classes):
                model.zero_grad()
                if xi.grad is not None:
                    xi.grad.zero_()
                logits_i[0, k].backward(retain_graph=True)
                gradients.append(xi.grad.detach().clone())
            target_grad = gradients[target_idx]
            perturb = None
            min_ratio = None
            for k in range(num_classes):
                if k == target_idx:
                    continue
                w_k = gradients[k] - target_grad
                f_k = logits_i[0, k] - logits_i[0, target_idx]
                ratio = torch.abs(f_k) / (torch.norm(w_k.flatten(), p=2) + 1e-12)
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    perturb = (ratio * w_k / (torch.norm(w_k.flatten(), p=2) + 1e-12))
            r_tot += perturb
            xi = xi + (1 + overshoot) * perturb
            xi = _clamp(xi).detach()
        x_adv[b] = xi.squeeze(0)
    return x_adv


def carlini_wagner_l2(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    confidence: float = 0.0,
    c: float = 1e-3,
    steps: int = 200,
    lr: float = 1e-2,
) -> torch.Tensor:
    """Carlini & Wagner L2 attack (simplified)."""

    device = x.device
    w = torch.atanh((x * 2 - 1) * 0.999999).detach()
    w.requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    def _to_img(w_tensor):
        return torch.tanh(w_tensor) * 0.5 + 0.5

    for _ in range(steps):
        adv_img = _to_img(w)
        logits, *_ = model(adv_img)
        one_hot = torch.nn.functional.one_hot(y, num_classes=logits.shape[1]).float()
        real = torch.sum(one_hot * logits, dim=1)
        other = torch.max((1 - one_hot) * logits - one_hot * 1e4, dim=1)[0]
        f_loss = torch.clamp(real - other + confidence, min=0)
        l2_loss = torch.sum((adv_img - x) ** 2, dim=(1, 2, 3))
        loss = c * f_loss.mean() + l2_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return _to_img(w).detach()


def autoattack_runner(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str = "Linf",
    eps: float = 0.03,
    version: str = "standard",
) -> torch.Tensor:
    """Wrapper around the AutoAttack library (Croce & Hein, 2020)."""

    if AutoAttack is None:
        raise ImportError("AutoAttack is not installed. Run `pip install autoattack` to enable this feature.")
    attacker = AutoAttack(model, norm=norm, eps=eps, version=version)
    attacker.attacks_to_run = ["apgd-ce", "apgd-t", "fab", "square"]
    x_adv = attacker.run_standard_evaluation(x, y, bs=x.shape[0])
    return x_adv
