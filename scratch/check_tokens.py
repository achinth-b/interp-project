from transformers import AutoProcessor
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
img = Image.new('RGB', (512, 512))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[img], return_tensors="pt")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
# Find image tokens
# SmolVLM uses <image> placeholder which gets expanded.
# We can find the token ID for image patches.
# Actually, the processor.tokenizer has specific tokens.
print(f"Image tokens count: {(inputs['input_ids'] >= 49152).sum().item()}") # 49152 is around where special tokens start
