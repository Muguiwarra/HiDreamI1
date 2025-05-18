import torch
from transformers import AutoTokenizer, LlamaForCausalLM
# from diffusers import HiDreamImagePipeline
import diffusers

tokenizer_4 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

pipe = diffusers.HiDreamImagePipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Full",
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

prompt = "Coloring page for girls, with moderated elements: Sunny countryside farm with a small farmhouse," \
            "red barn, and a little girl (Lila) in a dress smiling at animals in the distance."

image = pipe(
	prompt,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("output.png")
