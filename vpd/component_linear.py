"""ComponentLinear: the core VPD wrapper for nn.Linear.

Adapted from Goodfire's nano_param_decomp/run.py (Section C).

Each ComponentLinear replaces one nn.Linear in the target model and decomposes
its weight W_target into C rank-1 subcomponents:

    W_target ≈ (V @ U)^T + Δ

where V: [d_in, C], U: [C, d_out], and Δ = W_target - (V @ U)^T is the
residual "delta" that absorbs what V @ U cannot yet explain.

Two forward modes:
  - "target":    original behavior, caches input for the CI function
  - "component": masked reconstruction via per-subcomponent importance masks
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ComponentLinear(nn.Module):
    """Replaces one nn.Linear in the target model with a decomposed version."""

    def __init__(self, linear: nn.Linear, C: int) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = C

        # Frozen original weight and bias.
        self.register_buffer("W_target", linear.weight.detach().clone())
        bias = linear.bias
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None
        )

        # Learnable component parameters.
        self.V = nn.Parameter(
            torch.empty(d_in, C, device=linear.weight.device, dtype=linear.weight.dtype)
            .normal_(0.0, 1.0 / math.sqrt(d_in))
        )
        self.U = nn.Parameter(
            torch.empty(C, d_out, device=linear.weight.device, dtype=linear.weight.dtype)
            .normal_(0.0, 1.0 / math.sqrt(C))
        )

        # Transient per-forward state (set by the training loop).
        self.mode: Literal["target", "component"] = "target"
        self.mask: Tensor | None = None          # [B, S, C]
        self.delta_mask: Tensor | None = None    # [B, S]
        self.routing_mask: Tensor | None = None  # [B, S] bool
        self.last_input: Tensor | None = None

    def weight_delta(self) -> Tensor:
        """Residual: what the subcomponents don't yet explain."""
        return self.W_target - (self.V @ self.U).T

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "target":
            self.last_input = x.detach()
            return F.linear(x, self.W_target, self.bias)

        assert self.mask is not None and self.delta_mask is not None

        # Component activations: project input into component space, mask, project out.
        comp_acts = x @ self.V                    # [B, S, C]
        comp_out = (comp_acts * self.mask) @ self.U  # [B, S, d_out]
        if self.bias is not None:
            comp_out = comp_out + self.bias

        # Delta contribution (scaled per position).
        delta_out = F.linear(x, self.weight_delta())
        comp_out = comp_out + self.delta_mask.unsqueeze(-1) * delta_out

        # Optional routing: use target output at non-routed positions.
        if self.routing_mask is not None:
            target_out = F.linear(x, self.W_target, self.bias)
            comp_out = torch.where(self.routing_mask.unsqueeze(-1), comp_out, target_out)

        return comp_out


def install_components(
    model: nn.Module, module_to_c: dict[str, int]
) -> dict[str, ComponentLinear]:
    """Freeze the target model and replace each listed nn.Linear in-place."""
    for p in model.parameters():
        p.requires_grad_(False)

    wrappers: dict[str, ComponentLinear] = {}
    for path, C in module_to_c.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        linear = model.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is {type(linear)}, expected nn.Linear"
        wrapper = ComponentLinear(linear, C)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper

    return wrappers


def clear_wrappers(wrappers: dict[str, ComponentLinear]) -> None:
    """Reset all wrappers to target (pass-through) mode."""
    for w in wrappers.values():
        w.mode = "target"
        w.mask = None
        w.delta_mask = None
        w.routing_mask = None


def set_component_mode(
    wrappers: dict[str, ComponentLinear],
    masks: dict[str, Tensor],
    delta_masks: dict[str, Tensor],
    routing: dict[str, Tensor] | None = None,
) -> None:
    """Switch all wrappers to component (masked) mode."""
    for name, w in wrappers.items():
        w.mode = "component"
        w.mask = masks[name]
        w.delta_mask = delta_masks[name]
        w.routing_mask = None if routing is None else routing[name]
