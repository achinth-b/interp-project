"""VPD loss functions.

Adapted from Goodfire's nano_param_decomp/run.py (Section E).

The four loss terms that drive the decomposition:
  1. Faithfulness:    subcomponents must sum to the original weight (Δ → 0)
  2. Minimality:      as few subcomponents active per input as possible
  3. Stochastic:      model output stable under random masking
  4. Adversarial:     model output stable under worst-case masking (via PPGD)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .component_linear import ComponentLinear, clear_wrappers, set_component_mode


def faithfulness_loss(wrappers: dict[str, ComponentLinear]) -> Tensor:
    """Mean squared weight-delta error across all decomposed modules."""
    sum_sq = torch.zeros((), device=next(iter(wrappers.values())).V.device)
    numel = 0
    for w in wrappers.values():
        delta = w.weight_delta()
        sum_sq = sum_sq + delta.pow(2).sum()
        numel += delta.numel()
    return sum_sq / numel


def importance_minimality_loss(
    ci_upper: dict[str, Tensor], p: float, eps: float, beta: float
) -> Tensor:
    """Encourage sparsity in causal importance values.

    Uses annealed L_p norm (p decreases from p_start to p_end over training)
    plus a logarithmic frequency penalty.
    """
    total = torch.zeros((), device=next(iter(ci_upper.values())).device)
    for v in ci_upper.values():
        vals = (v + eps).pow(p)  # [B, S, C]
        batch_seq_dims = tuple(range(vals.ndim - 1))
        sum_c = vals.sum(dim=batch_seq_dims)
        n = math.prod(vals.shape[:-1])
        mean_c = sum_c / n
        total = total + (mean_c + beta * mean_c * torch.log2(1 + sum_c)).sum()
    return total


def anneal_p(step: int, total_steps: int, p_start: float, p_end: float) -> float:
    """Linear interpolation of the L_p exponent over training."""
    t = min(max(step / total_steps, 0.0), 1.0)
    return p_start + (p_end - p_start) * t


def kl_logits(pred: Tensor, target: Tensor) -> Tensor:
    """KL(softmax(target) || softmax(pred)), averaged over all positions."""
    log_q = F.log_softmax(pred, dim=-1)
    p = F.softmax(target.detach(), dim=-1)
    return F.kl_div(log_q, p, reduction="none").sum(dim=-1).mean()


def sample_continuous_masks(
    ci_lower: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Sample stochastic masks: mask = ci + (1-ci) * U(0,1)."""
    masks: dict[str, Tensor] = {}
    delta_masks: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        u = torch.rand_like(ci)
        masks[name] = ci + (1 - ci) * u
        delta_masks[name] = torch.rand(
            *ci.shape[:-1], device=ci.device, dtype=ci.dtype
        )
    return masks, delta_masks


def sample_routing(
    module_names: list[str], batch_dims: tuple[int, ...], device: torch.device
) -> dict[str, Tensor]:
    """Uniform k-subset routing: for each position, route to a random k-subset of modules."""
    M = len(module_names)
    k = torch.randint(1, M + 1, batch_dims, device=device)
    noise = torch.rand(M, *batch_dims, device=device)
    ranks = noise.argsort(dim=0).argsort(dim=0)
    return {name: ranks[i] < k for i, name in enumerate(module_names)}


def stochastic_recon_loss(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_lower: dict[str, Tensor],
    target_logits: Tensor,
    forward_fn: callable,
) -> Tensor:
    """One-sample stochastic-mask reconstruction loss with routing."""
    first_ci = next(iter(ci_lower.values()))
    B, S = first_ci.shape[:2]
    device = first_ci.device

    masks, delta_masks = sample_continuous_masks(ci_lower)
    routing = sample_routing(list(wrappers), (B, S), device)
    set_component_mode(wrappers, masks, delta_masks, routing)
    try:
        pred_logits = forward_fn()
    finally:
        clear_wrappers(wrappers)
    return kl_logits(pred_logits, target_logits)


def cosine_lr(
    step: int, total: int, start: float, final_frac: float, warmup_pct: float = 0.0
) -> float:
    """Linear warmup + cosine decay schedule."""
    warmup_steps = int(warmup_pct * total)
    decay_steps = total - warmup_steps
    if warmup_steps > 0 and step < warmup_steps:
        return start * (step / warmup_steps)
    if decay_steps <= 1:
        return start
    progress = (step - warmup_steps) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    final = start * final_frac
    return final + 0.5 * (start - final) * (1 + math.cos(math.pi * progress))
