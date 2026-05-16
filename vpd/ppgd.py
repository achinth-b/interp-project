"""Persistent Projected Gradient Descent (PPGD) for adversarial ablation.

Adapted from Goodfire's nano_param_decomp/run.py (Section F).

PPGD maintains adversarial ablation masks that persist across training steps.
These masks are optimized to find the worst-case masking pattern that maximizes
reconstruction loss — forcing the decomposition to be robust against adversarial
removal of subcomponents.

Each training step:
  1. warmup:       2 inner PGD steps on the current batch
  2. recon_loss:   forward once with current adversarial masks (for the main loss)
  3. external_step: Adam update on the masks using gradients from the total loss
"""

import torch
import torch.nn as nn
from torch import Tensor

from .component_linear import ComponentLinear, clear_wrappers, set_component_mode
from .config import VPDConfig
from .losses import kl_logits


class PersistentPGD:
    """Per-module adversarial sources with Adam state, persisted across steps."""

    def __init__(
        self,
        wrappers: dict[str, ComponentLinear],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        cfg: VPDConfig,
    ) -> None:
        self.cfg = cfg
        self.sources: dict[str, Tensor] = {}
        self.m: dict[str, Tensor] = {}
        self.v: dict[str, Tensor] = {}

        for name, w in wrappers.items():
            shape = (batch_size, seq_len, w.C + 1)  # +1 for delta mask
            src = torch.rand(shape, device=device).requires_grad_(True)
            self.sources[name] = src
            self.m[name] = torch.zeros(shape, device=device)
            self.v[name] = torch.zeros(shape, device=device)
        self.t = 0

    def _masks_from_sources(
        self, ci_lower: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Convert raw adversarial sources into masks using CI values."""
        masks: dict[str, Tensor] = {}
        delta_masks: dict[str, Tensor] = {}
        for name, ci in ci_lower.items():
            s = self.sources[name]
            masks[name] = ci + (1 - ci) * s[..., :ci.shape[-1]]
            delta_masks[name] = s[..., -1]
        return masks, delta_masks

    def recon_loss(
        self,
        wrappers: dict[str, ComponentLinear],
        ci_lower: dict[str, Tensor],
        target_logits: Tensor,
        forward_fn: callable,
    ) -> Tensor:
        """Compute reconstruction loss under current adversarial masks."""
        masks, delta_masks = self._masks_from_sources(ci_lower)
        set_component_mode(wrappers, masks, delta_masks, routing=None)
        try:
            pred = forward_fn()
        finally:
            clear_wrappers(wrappers)
        return kl_logits(pred, target_logits)

    def warmup(
        self,
        wrappers: dict[str, ComponentLinear],
        ci_lower: dict[str, Tensor],
        target_logits: Tensor,
        forward_fn: callable,
        lr: float,
    ) -> None:
        """Run inner PGD steps to find adversarial masks."""
        for _ in range(self.cfg.ppgd_inner_steps):
            loss = self.recon_loss(wrappers, ci_lower, target_logits, forward_fn)
            grads = torch.autograd.grad(
                loss, list(self.sources.values()), retain_graph=False
            )
            self._adam_step(dict(zip(self.sources, grads, strict=True)), lr)

    def external_step(self, grads: dict[str, Tensor], lr: float) -> None:
        """Adam update using gradients extracted from the total loss."""
        self._adam_step(grads, lr)

    def _adam_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self.t += 1
        bc1 = 1 - self.cfg.ppgd_beta1 ** self.t
        bc2 = 1 - self.cfg.ppgd_beta2 ** self.t
        with torch.no_grad():
            for name, src in self.sources.items():
                g = grads[name]
                m, v = self.m[name], self.v[name]
                m.mul_(self.cfg.ppgd_beta1).add_(g, alpha=1 - self.cfg.ppgd_beta1)
                v.mul_(self.cfg.ppgd_beta2).addcmul_(g, g, value=1 - self.cfg.ppgd_beta2)
                src.add_(lr * (m / bc1) / ((v / bc2).sqrt() + self.cfg.ppgd_eps))
                src.clamp_(0.0, 1.0)
