"""
Z-Image-Turbo ottimizzato con Torch-TensorRT per inferenza accelerata su GPU NVIDIA.

Requisiti:
    pip install torch-tensorrt diffusers transformers accelerate

Note:
    - La prima esecuzione sarà più lenta (compilazione TensorRT)
    - Le esecuzioni successive saranno significativamente più veloci
    - Richiede GPU NVIDIA con supporto TensorRT
"""

import torch
import torch_tensorrt
from diffusers import ZImagePipeline, ZImageTransformer2DModel
import time

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. "
    "Impeccable makeup, red floral forehead pattern. "
    "Elaborate high bun, golden phoenix headdress, red flowers, beads. "
    "Holds round folding fan with lady, trees, bird. "
    "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
    "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
    "blurred colorful distant lights."
)
HEIGHT = 1024
WIDTH = 1024
SEED = 42
NUM_INFERENCE_STEPS = 9
OUTPUT_FILE = "zimage_tensorrt.png"

# ============================================================================
# TORCH-TENSORRT COMPILATION SETTINGS
# ============================================================================
TENSORRT_CONFIG = {
    "enabled_precisions": {torch.float16, torch.float32},
    "truncate_long_and_double": True,
    "min_block_size": 1,
    "debug": False,
}


def setup_tensorrt_optimization():
    """Configura le ottimizzazioni TensorRT globali."""
    # Abilita cudnn benchmark per ottimizzazioni automatiche
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_pipeline():
    """
    Carica la pipeline Z-Image-Turbo.
    Usa il modello non-quantizzato per compatibilità TensorRT.
    """
    print("Caricamento pipeline Z-Image-Turbo...")

    pipeline = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.float16,
    ).to("cuda")

    return pipeline


def compile_with_tensorrt(pipeline):
    """
    Compila il transformer con Torch-TensorRT per inferenza ottimizzata.

    La compilazione usa torch.compile() con backend 'tensorrt' che:
    - Analizza il grafo computazionale
    - Ottimizza le operazioni per la GPU NVIDIA specifica
    - Fonde operazioni dove possibile
    - Applica ottimizzazioni di memoria
    """
    print("Compilazione transformer con Torch-TensorRT...")
    print("(Prima compilazione può richiedere alcuni minuti)")

    # Metodo 1: torch.compile con backend tensorrt
    # Più semplice e automatico
    try:
        pipeline.transformer = torch.compile(
            pipeline.transformer,
            backend="tensorrt",
            options=TENSORRT_CONFIG,
        )
        print("✓ Compilazione TensorRT completata")
        return pipeline, "tensorrt"
    except Exception as e:
        print(f"⚠ Backend TensorRT non disponibile: {e}")
        print("Fallback a torch.compile con inductor...")

    # Metodo 2: Fallback a inductor (sempre disponibile)
    try:
        pipeline.transformer = torch.compile(
            pipeline.transformer,
            backend="inductor",
            mode="max-autotune",
        )
        print("✓ Compilazione inductor completata")
        return pipeline, "inductor"
    except Exception as e:
        print(f"⚠ Compilazione fallita: {e}")
        print("Esecuzione senza compilazione...")
        return pipeline, "none"


def warmup_pipeline(pipeline):
    """
    Esegue un warmup per completare la compilazione JIT.
    """
    print("Warmup pipeline (compilazione lazy)...")

    with torch.no_grad():
        _ = pipeline(
            prompt="warmup",
            num_inference_steps=2,
            guidance_scale=0.0,
            height=256,
            width=256,
            generator=torch.Generator("cuda").manual_seed(0),
            output_type="latent",
        )

    # Sincronizza e libera cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("✓ Warmup completato")


def generate_image(pipeline, prompt, height, width, seed, steps):
    """
    Genera un'immagine con la pipeline ottimizzata.
    """
    generator = torch.Generator("cuda").manual_seed(seed)

    start_time = time.perf_counter()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        result = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,  # 0 per modelli Turbo
            height=height,
            width=width,
            generator=generator,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return result.images[0], elapsed


def benchmark(pipeline, prompt, height, width, seed, steps, num_runs=3):
    """
    Esegue un benchmark per misurare le prestazioni.
    """
    print(f"\nBenchmark ({num_runs} esecuzioni)...")
    times = []

    for i in range(num_runs):
        _, elapsed = generate_image(pipeline, prompt, height, width, seed + i, steps)
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    print(f"  Media: {avg_time:.2f}s")

    return avg_time


def main():
    print("=" * 60)
    print("Z-Image-Turbo con ottimizzazione Torch-TensorRT")
    print("=" * 60)

    # Setup ottimizzazioni
    setup_tensorrt_optimization()

    # Verifica GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile. Richiesta GPU NVIDIA.")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")

    # Carica pipeline
    pipeline = load_pipeline()

    # Compila con TensorRT
    pipeline, backend_used = compile_with_tensorrt(pipeline)

    # Warmup
    warmup_pipeline(pipeline)

    # Genera immagine
    print(f"\nGenerazione immagine {WIDTH}x{HEIGHT}...")
    image, elapsed = generate_image(
        pipeline, PROMPT, HEIGHT, WIDTH, SEED, NUM_INFERENCE_STEPS
    )

    # Salva
    image.save(OUTPUT_FILE)
    print(f"✓ Immagine salvata: {OUTPUT_FILE}")
    print(f"  Tempo: {elapsed:.2f}s")
    print(f"  Backend: {backend_used}")

    # Memory info
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  VRAM: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

    # Benchmark opzionale
    # benchmark(pipeline, PROMPT, HEIGHT, WIDTH, SEED, NUM_INFERENCE_STEPS)

    print("\n✓ Completato!")


if __name__ == "__main__":
    main()
