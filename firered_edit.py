"""FireRed-Image-Edit-1.0 inference script."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed-Image-Edit inference")
    parser.add_argument("--input_image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Editing instruction")
    parser.add_argument("--output_image", type=Path, default=Path("output_firered.png"), help="Output path")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument(
        "--model_path",
        type=str,
        default="FireRedTeam/FireRed-Image-Edit-1.0",
        help="HuggingFace model ID or local path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    print("Pipeline loaded.")

    image = Image.open(args.input_image).convert("RGB")
    print(f"Input image: {args.input_image} ({image.size[0]}x{image.size[1]})")
    print(f"Prompt: {args.prompt}")

    with torch.inference_mode():
        result = pipe(
            image=[image],
            prompt=args.prompt,
            negative_prompt=" ",
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            true_cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            num_images_per_prompt=1,
        )

    result.images[0].save(args.output_image)
    print(f"Output saved to: {args.output_image.resolve()}")


if __name__ == "__main__":
    main()
