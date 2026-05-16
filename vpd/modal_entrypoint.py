"""Modal entrypoint for VPD decomposition of SmolVLM-256M.

This script runs the full VPD pipeline on Modal:
  1. Downloads SmolVLM-256M and COCO val2017 images
  2. Prepares the data pipeline (image + text batches)
  3. Runs the VPD decomposition on the first 4 LLM layers
  4. Saves the checkpoint to a Modal volume

Usage:
    cd interp-project && uv run modal run vpd/modal_entrypoint.py
"""

import modal
from pathlib import Path

app = modal.App("vpd-smolvlm")
nfs = modal.NetworkFileSystem.from_name("vpd-smolvlm-nfs", create_if_missing=True)

# Build image with vpd package mounted for import.
vpd_dir = str(Path(__file__).parent)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.46,<4.49",
        "accelerate",
        "datasets",
        "Pillow",
        "matplotlib",
    )
    .add_local_dir(vpd_dir, remote_path="/root/vpd")
)


@app.function(
    image=image,
    network_file_systems={"/vol": nfs},
    gpu="A100",
    timeout=36000,
)
def run_vpd():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = "/vol/hf_cache"
    
    import random
    import sys
    from collections.abc import Iterator

    sys.path.insert(0, "/root")

    import datasets
    import torch
    from PIL import Image
    from torch import Tensor
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from vpd.config import VPDConfig
    from vpd.decompose import decompose

    cfg = VPDConfig()
    device = torch.device("cuda")

    # --- Load model ---
    print("[modal] Loading SmolVLM-256M-Instruct...")
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"[modal] Model loaded: {type(model).__name__}")

    # --- Load COCO dataset ---
    print(f"[modal] Loading COCO val2017 ({cfg.coco_n_images} images)...")
    ds = datasets.load_dataset(
        "detection-datasets/coco",
        split="val",
        streaming=False,
    )
    ds = ds.shuffle(seed=cfg.seed)
    ds = ds.select(range(min(cfg.coco_n_images, len(ds))))
    print(f"[modal] Dataset ready: {len(ds)} images")

    # --- Data iterator ---
    def make_train_iter() -> Iterator[dict[str, Tensor]]:
        """Infinite iterator yielding processed batches of image-text pairs."""
        indices = list(range(len(ds)))
        first_batch = True
        while True:
            random.shuffle(indices)
            batch_images: list[Image.Image] = []
            batch_texts: list[str] = []

            for idx in indices:
                row = ds[idx]
                img = row["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                batch_images.append(img)
                batch_texts.append(cfg.prompt)

                if len(batch_images) == cfg.batch_size:
                    messages_batch = [
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": text},
                                ],
                            }
                        ]
                        for text in batch_texts
                    ]
                    prompts = [
                        processor.apply_chat_template(msgs, add_generation_prompt=True)
                        for msgs in messages_batch
                    ]
                    inputs = processor(
                        text=prompts,
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                    )
                    if first_batch:
                        print(f"[modal] Batch 0 sequence length: {inputs['input_ids'].shape[1]}")
                        first_batch = False
                    yield {k: v for k, v in inputs.items()}
                    batch_images = []
                    batch_texts = []

    train_iter = make_train_iter()

    # --- Run decomposition ---
    print("[modal] Starting VPD decomposition...")
    save_path = "/vol/vpd_checkpoint.pt"
    wrappers = decompose(model, cfg, train_iter, device, save_path=save_path)
    print("[modal] Done. Checkpoint saved to network file system.")


@app.local_entrypoint()
def main():
    run_vpd.remote()
@app.function(
    image=image,
    network_file_systems={"/vol": nfs},
    gpu="A100",
)
def visualize_atoms(checkpoint_path: str, image_path: str | None = None, top_k: int = 10):
    """Remote visualization worker."""
    import os
    import shutil
    import torch
    from PIL import Image
    from vpd.visualize import VPDVisualizer
    
    print(f"Starting remote visualization with {checkpoint_path} | top_k={top_k}...")
    viz = VPDVisualizer(checkpoint_path)
    
    with torch.no_grad():
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        else:
            print("No image provided or found. Using a default research sample.")
            img = Image.new('RGB', (512, 512), color=(73, 109, 137))
            
        inputs, ci = viz.get_importance(img, "<image>Describe the visual features of this image.")
        viz.save_heatmaps(img, inputs, ci, top_k=top_k)
    
    print("Remote visualization complete. Heatmaps saved to /root/ in the container.")
    # Move them to /vol so they can be downloaded
    import glob
    import shutil
    for f in glob.glob("heatmap_atom_*.png"):
        dest = os.path.join("/vol", os.path.basename(f))
        shutil.move(f, dest)
        print(f"Moved {f} to {dest} for download.")

@app.local_entrypoint()
def viz(checkpoint_path: str, image_path: str | None = None, top_k: int = 10):
    """Unified command: Upload -> Analyze -> Auto-Download."""
    import subprocess
    import os
    
    remote_image_path = None
    if image_path:
        # 1. Auto-upload image to NFS
        filename = os.path.basename(image_path)
        print(f"Uploading {image_path} to Modal NFS...")
        # Optional: cleanup old version
        subprocess.run(["uv", "run", "modal", "nfs", "rm", "vpd-smolvlm-nfs", filename], stderr=subprocess.DEVNULL)
        subprocess.run(["uv", "run", "modal", "nfs", "put", "vpd-smolvlm-nfs", image_path, filename], check=True)
        remote_image_path = os.path.join("/vol", filename)

    # 2. Run remote visualization
    print(f"🚀 Running remote visualization on A100...")
    visualize_atoms.remote(checkpoint_path, remote_image_path, top_k)

    # 3. Auto-download results
    print(f"📥 Downloading results to your Mac...")
    from vpd.download_results import download_heatmaps
    download_heatmaps()
    print("\n✨ Done! Your heatmaps are ready locally.")
