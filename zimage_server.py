"""
ZImage Turbo Server
Mantiene il modello caricato in memoria e risponde a richieste HTTP.

Uso:
    python zimage_server.py

Endpoint:
    POST /generate - Genera un'immagine
    GET /health - Verifica stato server
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
import torch
import io
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

# Configurazione
MODEL_PATH = "./z_image_turbo-Q8_0.gguf"
OUTPUT_DIR = Path("./generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Variabile globale per il pipeline (caricato una sola volta)
pipeline = None


def load_model():
    """Carica il modello una sola volta all'avvio del server."""
    global pipeline

    print("⏳ Caricamento del modello ZImage Turbo...")

    transformer = ZImageTransformer2DModel.from_single_file(
        MODEL_PATH,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )

    pipeline = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        dtype=torch.bfloat16,
    ).to("cuda")

    print("✅ Modello caricato con successo!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestisce il ciclo di vita dell'applicazione."""
    # Startup: carica il modello
    load_model()
    yield
    # Shutdown: cleanup (opzionale)
    print("🛑 Server in chiusura...")


app = FastAPI(
    title="ZImage Turbo Server",
    description="Server per generazione immagini con ZImage Turbo",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    """Schema per la richiesta di generazione."""
    prompt: str = Field(..., description="Descrizione dell'immagine da generare")
    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1024, ge=256, le=2048)
    seed: int = Field(default=-1, description="Seed per riproducibilità (-1 = random)")
    num_inference_steps: int = Field(default=9, ge=1, le=50)
    save_to_disk: bool = Field(default=False, description="Salva l'immagine su disco")


class GenerateResponse(BaseModel):
    """Schema per la risposta di generazione."""
    success: bool
    message: str
    filename: str | None = None
    seed_used: int


@app.get("/health")
async def health_check():
    """Verifica che il server sia attivo e il modello caricato."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """
    Genera un'immagine dal prompt.

    Restituisce i metadati. Usa /generate/image per ottenere l'immagine direttamente.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato")

    # Gestione seed
    actual_seed = request.seed if request.seed >= 0 else torch.randint(0, 2**32, (1,)).item()

    try:
        image = pipeline(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=0.0,  # Turbo models don't need guidance
            height=request.height,
            width=request.width,
            generator=torch.Generator("cuda").manual_seed(actual_seed)
        ).images[0]

        filename = None
        if request.save_to_disk:
            filename = f"{uuid.uuid4().hex}.png"
            image.save(OUTPUT_DIR / filename)

        return GenerateResponse(
            success=True,
            message="Immagine generata con successo",
            filename=filename,
            seed_used=actual_seed
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image")
async def generate_and_return_image(request: GenerateRequest):
    """
    Genera un'immagine e la restituisce direttamente come PNG.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato")

    actual_seed = request.seed if request.seed >= 0 else torch.randint(0, 2**32, (1,)).item()

    try:
        image = pipeline(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=0.0,
            height=request.height,
            width=request.width,
            generator=torch.Generator("cuda").manual_seed(actual_seed)
        ).images[0]

        # Converti in bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"X-Seed-Used": str(actual_seed)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{filename}")
async def get_saved_image(filename: str):
    """Recupera un'immagine salvata su disco."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Immagine non trovata")
    return FileResponse(filepath, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
