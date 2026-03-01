"""FastAPI server for FireRed-Image-Edit-1.0."""

import io
import asyncio
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.0"

pipe = None
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print(f"Loading model: {MODEL_ID}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    print("Model loaded and ready.")
    yield
    del pipe
    torch.cuda.empty_cache()


app = FastAPI(title="FireRed-Image-Edit API", lifespan=lifespan)


def run_inference(image: Image.Image, prompt: str, seed: int, cfg_scale: float, steps: int) -> Image.Image:
    with torch.inference_mode():
        result = pipe(
            image=[image],
            prompt=prompt,
            negative_prompt=" ",
            generator=torch.Generator(device="cuda").manual_seed(seed),
            true_cfg_scale=cfg_scale,
            num_inference_steps=steps,
            num_images_per_prompt=1,
        )
    return result.images[0]


@app.post("/edit")
async def edit_image(
    image: UploadFile = File(..., description="Input image"),
    prompt: str = Form(..., description="Editing instruction"),
    seed: int = Form(43, description="Random seed"),
    cfg_scale: float = Form(4.0, description="True CFG scale"),
    steps: int = Form(40, description="Inference steps"),
):
    # Read and validate input image
    try:
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run inference with GPU lock to prevent concurrent access
    async with gpu_lock:
        result = await asyncio.to_thread(run_inference, img, prompt, seed, cfg_scale, steps)

    # Return as PNG
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}
