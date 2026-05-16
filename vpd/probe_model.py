"""Probe SmolVLM-256M to discover nn.Linear module paths for VPD targeting."""
import modal

app = modal.App("vpd-probe-smolvlm")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate"
)


@app.function(image=image, gpu="T4", timeout=300)
def probe():
    import torch.nn as nn
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    print("=" * 80)
    print("ALL nn.Linear MODULES")
    print("=" * 80)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name:65s}  {tuple(module.weight.shape)}")

    print("\n" + "=" * 80)
    print("FIRST 4 LLM LAYERS (VPD TARGETS)")
    print("=" * 80)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            f"layers.{i}." in name for i in range(4)
        ):
            print(f"  {name:65s}  {tuple(module.weight.shape)}")

    print("\n" + "=" * 80)
    print("MODEL CLASS:", type(model).__name__)
    print("=" * 80)


@app.local_entrypoint()
def main():
    probe.remote()
