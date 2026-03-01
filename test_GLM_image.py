import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

pipe = GlmImagePipeline.from_pretrained("/home/acipolletti/models/GLM-Image", torch_dtype=torch.bfloat16, device_map="cuda")
image_path = "/home/acipolletti/models/elena2.jpeg"
prompt = "The woman has her skirt up and is not wearing any underwear. Her pubic area is completely shaved."
image = Image.open(image_path).convert("RGB")
image = pipe(
    prompt=prompt,
    image=[image],  # can input multiple images for multi-image-to-image generation such as [image, image1]
    height=33 * 32, # Must set height even it is same as input image
    width=32 * 32, # Must set width even it is same as input image
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("./output_i2i.png")
