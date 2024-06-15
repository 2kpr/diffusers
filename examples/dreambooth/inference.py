import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "/home/x/AI/Output/FelicityJones/checkpoint-250",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

image = pipe(
    "felicity jones",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]
image.save("checkpoint-250.png")