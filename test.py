# from huggingface_hub import InferenceClient



# prompt = "Coloring page for girls, with moderated elements: Sunny countryside farm with a small farmhouse," \
#             "red barn, and a little girl (Lila) in a dress smiling at animals in the distance."


# # output is a PIL.Image object
# image = client.text_to_image(
#     prompt,
# 	height=1024,
#     width=1024,
#     guidance_scale=5.0,
#     num_inference_steps=50,
# 	seed=82706,
#     # model="HiDream-ai/HiDream-I1-Dev",
# 	model="renderartist/coloringbookhidream",
# )

# image.save("output_2.png")


# https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/diffusion_models/hidream_i1_full_fp8.safetensors?download=true

from huggingface_hub import snapshot_download

# Download base model
snapshot_download(repo_id="HiDream-ai/HiDream-I1-Full", local_dir="models/hidream", local_dir_use_symlinks=False)

# Download LoRA
snapshot_download(repo_id="renderartist/coloringbookhidream", local_dir="models/lora", local_dir_use_symlinks=False)

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from peft import PeftModel
import torch

# Load base model
pipe = DiffusionPipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Apply the LoRA weights manually — assumed to be in PEFT format
# NOTE: If LoRA isn't PEFT-compatible, you'll need a manual merge
pipe.unet.load_attn_procs("models/lora")  # This works if LoRA is compatible with diffusers

# Prompt
prompt = "a goat, c0l0ringb00k"
image = pipe(prompt).images[0]
image.save("coloringbook-goat.png")
image.show()
