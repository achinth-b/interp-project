"""VPD decomposition training loop for SmolVLM-256M.

Adapted from Goodfire's nano_param_decomp/run.py (Sections H-I).

This is the core training function that takes a loaded VLM model and data,
installs ComponentLinear wrappers on the target layers, builds the CI
transformer, and runs the full VPD optimization: faithfulness warmup followed
by the four-term loss main loop.

Key adaptation from Goodfire's LLM-only implementation:
  - The model takes (input_ids, pixel_values, attention_mask) instead of just input_ids
  - We only decompose a subset of layers (first 4 of 30)
  - Data comes from image-text pairs rather than pure text
"""

import os
import time
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from .ci_transformer import CITransformer
from .component_linear import ComponentLinear, clear_wrappers, install_components
from .config import VPDConfig
from .losses import (
    anneal_p,
    cosine_lr,
    faithfulness_loss,
    importance_minimality_loss,
    kl_logits,
    stochastic_recon_loss,
)
from .ppgd import PersistentPGD


def decompose(
    model: nn.Module,
    cfg: VPDConfig,
    train_iter: Iterator[dict[str, Tensor]],
    device: torch.device,
    save_path: str | None = None,
) -> dict[str, ComponentLinear]:
    """Run VPD decomposition on the target model.

    Args:
        model: The full VLM (Idefics3ForConditionalGeneration), already on device.
        cfg: VPD hyperparameters.
        train_iter: Yields dicts with keys 'input_ids', 'attention_mask', 'pixel_values'.
        device: CUDA device.
        save_path: Optional path to save the decomposition checkpoint.

    Returns:
        Dictionary of ComponentLinear wrappers (the decomposition result).
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model.eval()
    c_per_module = cfg.build_c_per_module()
    total_C = sum(c_per_module.values())

    # --- Install ComponentLinear wrappers ---
    wrappers = install_components(model, c_per_module)
    print(f"[vpd] Installed {len(wrappers)} ComponentLinear wrappers ({total_C} total components)")

    # --- Build CI transformer ---
    d_in_per_module = {name: int(w.W_target.shape[1]) for name, w in wrappers.items()}
    ci_fn = CITransformer(d_in_per_module, c_per_module, cfg, max_seq_len=2048)
    ci_params_count = sum(p.numel() for p in ci_fn.parameters())
    print(f"[vpd] CI transformer: {ci_params_count:,} parameters")
    ci_fn.to(device)

    # --- Resume logic ---
    start_step = 0
    if save_path and os.path.exists(save_path):
        print(f"[vpd] Found existing checkpoint at {save_path}. Resuming...")
        checkpoint = torch.load(save_path, map_location="cpu")
        start_step = checkpoint.get("step", 0)
        
        # Load wrappers
        for name, w_state in checkpoint["wrappers"].items():
            if name in wrappers:
                wrappers[name].V.data.copy_(w_state["V"].to(device))
                wrappers[name].U.data.copy_(w_state["U"].to(device))
        
        # Load CI transformer
        ci_fn.load_state_dict(checkpoint["ci_fn_state_dict"])
        print(f"[vpd] Resumed from step {start_step}")

    # --- Faithfulness warmup ---
    # Only run warmup if starting from scratch.
    component_params = [p for w in wrappers.values() for p in (w.V, w.U)]
    if start_step == 0:
        warmup_opt = torch.optim.AdamW(
            component_params, lr=cfg.faithfulness_warmup_lr, weight_decay=0.0
        )
        print(f"[vpd] Faithfulness warmup ({cfg.faithfulness_warmup_steps} steps)...")
        for step in range(cfg.faithfulness_warmup_steps):
            warmup_opt.zero_grad()
            loss = faithfulness_loss(wrappers)
            loss.backward()
            warmup_opt.step()
            if step % 100 == 0:
                print(f"  step {step:4d}  faith_loss={loss.item():.6f}")
        print(f"  final faith_loss={loss.item():.6f}")
    else:
        print("[vpd] Skipping faithfulness warmup (resuming from checkpoint)")

    # --- Main optimizer ---
    ci_params = list(ci_fn.parameters())
    opt = torch.optim.AdamW(
        component_params + ci_params, lr=cfg.main_lr, weight_decay=0.0
    )

    ppgd: PersistentPGD | None = None
    ppgd_seq_len = -1
    print(f"[vpd] Starting main loop ({start_step} to {cfg.n_steps})...")

    for step in range(start_step, cfg.n_steps):
        t0 = time.time()

        main_lr = cosine_lr(step, cfg.n_steps, cfg.main_lr, cfg.main_lr_final_frac)
        ppgd_lr = cosine_lr(
            step, cfg.n_steps, cfg.ppgd_lr, cfg.ppgd_lr_final_frac, cfg.ppgd_warmup_pct
        )
        for g in opt.param_groups:
            g["lr"] = main_lr

        batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        # Re-create PPGD if sequence length changed (due to different padding).
        cur_seq_len = input_ids.shape[1]
        cur_batch_size = input_ids.shape[0]
        if ppgd is None or cur_seq_len != ppgd_seq_len:
            ppgd = PersistentPGD(wrappers, cur_batch_size, cur_seq_len, device, cfg)
            ppgd_seq_len = cur_seq_len

        # Closure for forward pass (reused by stochastic and PPGD losses).
        def forward_fn() -> Tensor:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            return out.logits if hasattr(out, "logits") else out

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 1. Target forward: get ground-truth logits + cache pre-weight activations.
            clear_wrappers(wrappers)
            target_logits = forward_fn()
            acts = {n: w.last_input for n, w in wrappers.items()}

            # 2. CI transformer: predict subcomponent importance.
            ci_lower, ci_upper, _ = ci_fn(acts)

            # 3. PPGD inner warmup (adversarial mask optimization).
            ppgd.warmup(wrappers, ci_lower, target_logits, forward_fn, lr=ppgd_lr)

            # 4. Compute all four losses.
            loss_faith = faithfulness_loss(wrappers)
            loss_imp = importance_minimality_loss(
                ci_upper,
                anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end),
                cfg.imp_eps,
                cfg.imp_beta,
            )
            loss_stoch = stochastic_recon_loss(
                model, wrappers, ci_lower, target_logits, forward_fn
            )
            loss_ppgd = ppgd.recon_loss(wrappers, ci_lower, target_logits, forward_fn)

        # Total loss (outside autocast for fp32 precision).
        total = (
            cfg.coeff_faith * loss_faith
            + cfg.coeff_imp * loss_imp
            + cfg.coeff_stoch * loss_stoch
            + cfg.coeff_ppgd * loss_ppgd
        )

        # Extract PPGD gradients before main backward.
        ppgd_grads = torch.autograd.grad(
            loss_ppgd, list(ppgd.sources.values()), retain_graph=True
        )
        ppgd_grads_dict = dict(zip(ppgd.sources, ppgd_grads, strict=True))

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(component_params, cfg.grad_clip_components)
        opt.step()
        ppgd.external_step(ppgd_grads_dict, ppgd_lr)

        dt = time.time() - t0

        # --- Logging ---
        if step % cfg.log_freq == 0:
            print(
                f"  step={step:5d}  "
                f"faith={loss_faith.item():.4g}  "
                f"imp={loss_imp.item():.4g}  "
                f"stoch={loss_stoch.item():.4g}  "
                f"ppgd={loss_ppgd.item():.4g}  "
                f"lr={main_lr:.2e}  "
                f"dt={dt:.2f}s"
            )

        # --- Eval ---
        if step % cfg.eval_freq == 0 and step > 0:
            with torch.no_grad():
                # Quick eval: L0 (number of active components).
                total_l0 = 0.0
                for ci in ci_lower.values():
                    total_l0 += (ci > 0.0).float().sum(-1).mean().item()
                alive = sum(
                    (ci.mean(dim=(0, 1)) > 0.0).sum().item()
                    for ci in ci_lower.values()
                )
                print(
                    f"  [eval] step={step}  L0={total_l0:.1f}/{total_C}  "
                    f"alive={alive}/{total_C}"
                )

        # --- Periodic Checkpoint ---
        if save_path and step % 100 == 0 and step > 0:
            checkpoint = {
                "step": step,
                "config": cfg,
                "wrappers": {
                    name: {"V": w.V.detach().cpu(), "U": w.U.detach().cpu()}
                    for name, w in wrappers.items()
                },
                "ci_fn_state_dict": ci_fn.state_dict(),
            }
            # Save to a temporary path then move to avoid partial writes
            torch.save(checkpoint, save_path + ".tmp")
            import os
            os.replace(save_path + ".tmp", save_path)
            print(f"  [ckpt] Saved intermediate checkpoint to {save_path}")

    # --- Save checkpoint ---
    if save_path:
        checkpoint = {
            "config": cfg,
            "wrappers": {
                name: {"V": w.V.detach().cpu(), "U": w.U.detach().cpu()}
                for name, w in wrappers.items()
            },
            "ci_fn_state_dict": ci_fn.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"[vpd] Saved checkpoint to {save_path}")

    return wrappers
