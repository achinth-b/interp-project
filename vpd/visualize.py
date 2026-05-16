import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from vpd.config import VPDConfig
from vpd.component_linear import install_components, ComponentLinear
from vpd.ci_transformer import CITransformer

class VPDVisualizer:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg: VPDConfig = ckpt["config"]
        
        # Load model
        print(f"Loading base model {self.cfg.model_id}...")
        dtype = torch.float32 if self.device.type != "cuda" else torch.bfloat16
        
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        # Install wrappers
        c_per_module = self.cfg.build_c_per_module()
        self.wrappers = install_components(self.model, c_per_module)
        
        # Load component weights
        for name, w_data in ckpt["wrappers"].items():
            if name in self.wrappers:
                self.wrappers[name].V.data.copy_(w_data["V"].to(self.device))
                if "U" in w_data:
                    self.wrappers[name].U.data.copy_(w_data["U"].to(self.device))
        
        # Initialize and load CI Transformer
        d_in_per_module = {n: w.V.shape[0] for n, w in self.wrappers.items()}
        self.ci_transformer = CITransformer(d_in_per_module, c_per_module, self.cfg).to(self.device).to(dtype)
        if "ci_transformer" in ckpt:
            self.ci_transformer.load_state_dict(ckpt["ci_transformer"])
        self.ci_transformer.eval()
        
        print("VPD Visualizer ready.")

    def get_importance(self, image: Image.Image, prompt: str):
        """Get importance scores for all components across all layers."""
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # 1. Forward pass in target mode to collect activations
        with torch.no_grad():
            for w in self.wrappers.values(): w.mode = "target"
            _ = self.model(**inputs)
            
            # 2. Collect activations
            acts = {n: w.last_input for n, w in self.wrappers.items()}
            
            # 3. Pass through CI Transformer to get masks
            ci_lower, ci_upper, _ = self.ci_transformer(acts)
            
            # We'll use ci_lower for visualization as it's the [0, 1] mask
            all_masks = []
            for name, mask in ci_lower.items():
                all_masks.append((name, mask))
                    
        return inputs, all_masks

    def save_heatmaps(self, image: Image.Image, inputs: dict, all_masks: list, top_k: int = 5):
        """Save heatmaps for the globally top-k active atoms."""
        # Find globally top-k
        flat_scores = []
        for name, mask in all_masks:
            # mask shape [1, seq_len, n_components]
            avg_scores = mask[0].mean(dim=0) # [n_components]
            for i in range(len(avg_scores)):
                flat_scores.append({
                    "name": name,
                    "index": i,
                    "score": avg_scores[i].item(),
                    "mask": mask[0, :, i] # [seq_len]
                })
        
        flat_scores.sort(key=lambda x: x["score"], reverse=True)
        top_atoms = flat_scores[:top_k]
        
        print(f"Top {top_k} active components globally:")
        for i, a in enumerate(top_atoms):
            print(f"  {i+1}. {a['name']} | Atom #{a['index']} (score: {a['score']:.4f})")
        
        for i, atom in enumerate(top_atoms):
            importance = atom["mask"]
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            
            # Simple 1D-to-2D mapping for visualization
            side = int(math.sqrt(len(importance)))
            if side * side == len(importance):
                heatmap = importance.view(side, side).cpu().float().numpy()
            else:
                next_side = math.ceil(math.sqrt(len(importance)))
                padded = torch.zeros(next_side * next_side, device=importance.device)
                padded[:len(importance)] = importance
                heatmap = padded.view(next_side, next_side).cpu().float().numpy()
                
            plt.imshow(heatmap, cmap='jet', alpha=0.5, extent=(0, image.size[0], image.size[1], 0), interpolation='bilinear')
            
            # Clean up title
            short_name = atom['name'].replace("model.text_model.layers.", "Layer ")
            plt.title(f"Atom #{atom['index']} | {short_name}")
            plt.axis('off')
            
            # Make filename safe
            layer_id = atom['name'].split('.')[-2]
            save_path = f"heatmap_atom_{atom['index']}_L{layer_id}.png"
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"  -> Saved {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="<image>Describe the visual features of this image.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    
    viz = VPDVisualizer(args.checkpoint)
    
    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
    else:
        print("No image provided or found. Using a default research sample.")
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        
    inputs, all_masks = viz.get_importance(img, args.prompt)
    viz.save_heatmaps(img, inputs, all_masks, top_k=args.top_k)

if __name__ == "__main__":
    main()
