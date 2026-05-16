"""Causal Importance (CI) Transformer.

Adapted from Goodfire's nano_param_decomp/run.py (Section D).

The CI transformer is the learned function that predicts which subcomponents
are "needed" for a given input. It takes the pre-weight activations from all
decomposed layers, concatenates them, runs them through a small transformer,
and outputs per-subcomponent importance values in [0, 1].

The output passes through two variants of leaky-hard sigmoid:
  - lower_leaky: clamp(x, 0, 1) with asymmetric backward (can resurrect dead components)
  - upper_leaky: linear continuation above 1 (allows importance > 1 during training)
"""

import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import VPDConfig


# --- Leaky-hard sigmoids (Section B of Goodfire's implementation) ---

class _LowerLeakyHardSigmoid(torch.autograd.Function):
    """Forward: clamp(x, 0, 1). Backward: pass-through in (0,1); in x<=0 region,
    only pass alpha * grad when grad < 0 (to resurrect dead components)."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        alpha: float = ctx.alpha
        zero = torch.zeros_like(grad_output)
        grad = torch.where(
            x <= 0,
            torch.where(grad_output < 0, alpha * grad_output, zero),
            torch.where(x <= 1, grad_output, zero),
        )
        return grad, None


def lower_leaky(x: Tensor, alpha: float) -> Tensor:
    return cast(Tensor, _LowerLeakyHardSigmoid.apply(x, alpha))


def upper_leaky(x: Tensor, alpha: float) -> Tensor:
    """For x > 1: return 1 + alpha*(x-1). Otherwise: clamp(x, 0, 1)."""
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))


# --- RoPE utilities ---

def precompute_rope(
    seq_len: int, head_dim: int, base: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Split-in-half RoPE. x: [B, H, S, head_dim]; cos/sin: [S, half]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# --- CI Transformer blocks ---

class CIAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class CIBlock(nn.Module):
    def __init__(self, cfg: VPDConfig) -> None:
        super().__init__()
        self.attn = CIAttention(cfg.ci_d_model, cfg.ci_n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.ci_d_model, cfg.ci_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.ci_mlp_hidden, cfg.ci_d_model),
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(F.rms_norm(x, (x.shape[-1],)), cos, sin)
        x = x + self.mlp(F.rms_norm(x, (x.shape[-1],)))
        return x


class CITransformer(nn.Module):
    """The causal importance function: a shared transformer over all decomposed layers.

    Input: dict of pre-weight activations {module_path: [B, S, d_in]}.
    Output: three dicts of per-module tensors:
      - ci_lower: clamp(x, 0, 1) with resurrection gradient
      - ci_upper: linear continuation above 1
      - ci_raw: pre-sigmoid logits
    """

    def __init__(
        self,
        d_in_per_module: dict[str, int],
        c_per_module: dict[str, int],
        cfg: VPDConfig,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.module_order = sorted(d_in_per_module.keys())
        self.cfg = cfg

        total_in = sum(d_in_per_module.values())
        total_C = sum(c_per_module[n] for n in self.module_order)

        self.proj_in = nn.Linear(total_in, cfg.ci_d_model)
        self.blocks = nn.ModuleList([CIBlock(cfg) for _ in range(cfg.ci_n_blocks)])
        self.proj_out = nn.Linear(cfg.ci_d_model, total_C)
        self.c_splits: list[int] = [c_per_module[n] for n in self.module_order]

        # Initial RoPE buffer.
        head_dim = cfg.ci_d_model // cfg.ci_n_heads
        cos, sin = precompute_rope(max_seq_len, head_dim, cfg.ci_rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_rope(self, S: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Lazy-buffer RoPE: return cos/sin for sequence length S."""
        if S <= self.rope_cos.shape[0]:
            return self.rope_cos[:S], self.rope_sin[:S]

        # Extend RoPE buffer.
        head_dim = self.cfg.ci_d_model // self.cfg.ci_n_heads
        cos, sin = precompute_rope(S, head_dim, self.cfg.ci_rope_base, device)
        # Manually assign as we are extending an existing buffer.
        self.rope_cos = cos
        self.rope_sin = sin
        return cos, sin

    def forward(
        self, acts: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        # RMS-norm each module's activations and concatenate.
        normed = [F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order]
        x = torch.cat(normed, dim=-1)

        x = self.proj_in(x)
        S = x.shape[1]
        cos, sin = self._get_rope(S, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)

        logits = self.proj_out(x)  # [B, S, total_C]
        per_module = dict(
            zip(self.module_order, logits.split(self.c_splits, dim=-1), strict=True)
        )

        alpha = self.cfg.leaky_alpha
        ci_lower = {n: lower_leaky(v, alpha) for n, v in per_module.items()}
        ci_upper = {n: upper_leaky(v, alpha) for n, v in per_module.items()}
        return ci_lower, ci_upper, per_module
