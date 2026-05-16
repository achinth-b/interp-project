"""VPD configuration for SmolVLM-256M decomposition.

Hyperparameters adapted from Goodfire's nano_param_decomp, scaled for
the SmolLM2-135M backbone (30 layers, d_model=576, 9 attention heads).
"""

from dataclasses import dataclass, field


@dataclass
class VPDConfig:
    """All hyperparameters for the VPD decomposition run."""

    # --- Target model ---
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    decompose_layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # --- Component counts per module type ---
    # Following Goodfire's ratios from the SimpleStories-2L config.
    c_q_proj: int = 192
    c_k_proj: int = 128
    c_v_proj: int = 128
    c_o_proj: int = 192
    c_gate_proj: int = 512
    c_up_proj: int = 512
    c_down_proj: int = 384

    # --- Training schedule ---
    n_steps: int = 5_000
    batch_size: int = 8
    seed: int = 42

    # --- Main optimizer (AdamW over component params + CI transformer) ---
    main_lr: float = 5e-5
    main_lr_final_frac: float = 0.1

    # --- Faithfulness warmup (AdamW over component params only) ---
    faithfulness_warmup_steps: int = 400
    faithfulness_warmup_lr: float = 1e-3

    # --- Loss coefficients ---
    coeff_faith: float = 1e7
    coeff_imp: float = 2e-4
    coeff_stoch: float = 0.5
    coeff_ppgd: float = 0.5

    # --- Importance minimality (L_p with linear p-anneal) ---
    p_start: float = 2.0
    p_end: float = 0.4
    imp_eps: float = 1e-12
    imp_beta: float = 0.5

    # --- Leaky-hard sigmoid ---
    leaky_alpha: float = 0.01

    # --- CI transformer ---
    ci_d_model: int = 512
    ci_n_blocks: int = 4
    ci_n_heads: int = 8
    ci_mlp_hidden: int = 2048
    ci_rope_base: float = 10000.0

    # --- Persistent PGD ---
    ppgd_lr: float = 0.01
    ppgd_lr_final_frac: float = 1.0
    ppgd_warmup_pct: float = 0.025
    ppgd_beta1: float = 0.5
    ppgd_beta2: float = 0.99
    ppgd_eps: float = 1e-8
    ppgd_inner_steps: int = 2

    # --- Gradient clipping ---
    grad_clip_components: float = 0.01

    # --- Evaluation + logging ---
    eval_freq: int = 200
    log_freq: int = 50

    # --- Data ---
    coco_n_images: int = 2000
    prompt: str = "Describe this image in detail."

    def build_c_per_module(self) -> dict[str, int]:
        """Build the {module_path: n_components} map for all decomposed layers."""
        c_map: dict[str, int] = {}
        module_cs = {
            "self_attn.q_proj": self.c_q_proj,
            "self_attn.k_proj": self.c_k_proj,
            "self_attn.v_proj": self.c_v_proj,
            "self_attn.o_proj": self.c_o_proj,
            "mlp.gate_proj": self.c_gate_proj,
            "mlp.up_proj": self.c_up_proj,
            "mlp.down_proj": self.c_down_proj,
        }
        for layer_idx in self.decompose_layers:
            for suffix, n_c in module_cs.items():
                c_map[f"model.text_model.layers.{layer_idx}.{suffix}"] = n_c
        return c_map

    @property
    def total_components(self) -> int:
        return sum(self.build_c_per_module().values())
