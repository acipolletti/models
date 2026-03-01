"""
Client per ZImage Turbo Server

Esempi di utilizzo del server.
"""

import requests
import argparse


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8000",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    output_file: str = "output.png"
):
    """Genera un'immagine e salvala localmente."""

    response = requests.post(
        f"{server_url}/generate/image",
        json={
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "num_inference_steps": 9
        }
    )

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        seed_used = response.headers.get("X-Seed-Used", "unknown")
        print(f"✅ Immagine salvata: {output_file} (seed: {seed_used})")
    else:
        print(f"❌ Errore: {response.status_code} - {response.text}")


def check_health(server_url: str = "http://localhost:8000"):
    """Verifica lo stato del server."""
    try:
        response = requests.get(f"{server_url}/health")
        print(response.json())
    except requests.exceptions.ConnectionError:
        print("❌ Server non raggiungibile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZImage Client")
    parser.add_argument("prompt", nargs="?", help="Prompt per generare l'immagine")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--output", "-o", default="output.png")
    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)
    parser.add_argument("--seed", "-s", type=int, default=-1)
    parser.add_argument("--health", action="store_true", help="Controlla stato server")

    args = parser.parse_args()

    if args.health:
        check_health(args.server)
    elif args.prompt:
        generate_image(
            prompt=args.prompt,
            server_url=args.server,
            width=args.width,
            height=args.height,
            seed=args.seed,
            output_file=args.output
        )
    else:
        parser.print_help()
