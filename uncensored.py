import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda:0"
torch_dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload() #no need to do cpu offload for >80G VRAM carts like H200, B200, etc. and do a `pipe.to(device)` instead

prompt = """
An 18-year-old girl, light blonde, blue eyes, Scandinavian, beautiful face, full body, curvy body, large breasts,
terracotta tanned skin, tan lines,standing with legs apart, urban street at night, warm city lights, soft shadows, 
shallow depth of field, blurred background,high detail, natural color gradation.
"""

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt=prompt,
    #image=[cat_image] #multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
