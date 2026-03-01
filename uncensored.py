import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "black-forest-labs/FLUX.2-dev" 
device = "cuda:0"
torch_dtype = torch.bfloat16

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = """
An 18-year-old girl, light blonde, blue eyes, Scandinavian, beautiful face, full body, curvy body, large breasts,
terracotta tanned skin, tan lines,standing with legs apart, urban street at night, warm city lights, soft shadows, 
shallow depth of field, blurred background,high detail, natural color gradation.
"""
#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=[cat_image] #optional multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
