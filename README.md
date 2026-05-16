# VPD on SmolVLM-256M

Applying [Goodfire's adVersarial Parameter Decomposition (VPD)](https://github.com/goodfire-ai/param-decomp) 
to the first 4 transformer layers of SmolVLM-256M's LLM backbone.

## What this does

VPD decomposes each `nn.Linear` weight matrix into rank-1 subcomponents that sum back 
to the original weight. A learned Causal Importance function predicts which subcomponents 
are needed for each input. Adversarial ablation training ensures the decomposition is 
mechanistically faithful.

## Quick start

```bash
# Probe the model structure
uv run modal run vpd/probe_model.py

# Run the decomposition (A10G GPU, ~1-2 hours)
uv run modal run vpd/modal_entrypoint.py
```

## Architecture

```
vpd/
├── config.py              # Hyperparameters
├── component_linear.py    # ComponentLinear wrapper (from Goodfire)
├── ci_transformer.py      # Causal Importance function
├── losses.py              # 4-term VPD loss
├── ppgd.py                # Persistent PGD adversarial ablation
├── decompose.py           # Training loop
├── modal_entrypoint.py    # Modal runner
└── probe_model.py         # Model structure discovery
```
