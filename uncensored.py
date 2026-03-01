import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("kpsss34/FHDR_Uncensored", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
prompt = """
An 18-year-old girl, light blonde, blue eyes, Scandinavian, beautiful face, full body, curvy body, large breasts,
terracotta tanned skin, tan lines,standing with legs apart, urban street at night, warm city lights, soft shadows, 
shallow depth of field, blurred background,high detail, natural color gradation.
"""
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=40,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("./uncensored_outputs.png")
