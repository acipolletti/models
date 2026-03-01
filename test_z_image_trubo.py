from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
import torch

#prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
prompt = "young Swedish woman, blonde, blue eyes, tall, bikini, model body, with large breasts."
height = 1024
width = 1024
seed = 42

#hf_path = "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/z_image_turbo-Q3_K_M.gguf"
local_path = "./z_image_turbo-Q8_0.gguf"

transformer = ZImageTransformer2DModel.from_single_file(
    local_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)

pipeline = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    dtype=torch.bfloat16,
).to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Custom attention backend for better efficiency if supported:
#pipeline.transformer.set_attention_backend("_sage_qk_int8_pv_fp16_triton") # Enable Sage Attention
#pipeline.transformer.set_attention_backend("flash") # Enable Flash-Attention-2
#pipeline.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
#pipeline.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
#pipeline.enable_model_cpu_offload()

images = pipeline(
    prompt=prompt,
    num_inference_steps=9, # This actually results in 8 DiT forwards
    guidance_scale=0.0, # Guidance should be 0 for the Turbo models
    height=height,
    width=width,
    generator=torch.Generator("cuda").manual_seed(seed)
).images[0]

images.save("zimage.png")
