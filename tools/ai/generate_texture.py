"""Generates a seamless ground texture with Stable Diffusion XL, for the fine detail of a game's terrain tiles.

Usage, with the Python of the PyTorch environment described in the README:
    python -I generate_texture.py <output.png> [--prompt <text>] [--seed <n>] [--size 1024]

Every convolution of the network wraps around at the picture's edges instead of padding them, so the picture
continues seamlessly from its right edge to its left and from its bottom to its top, and can be repeated in
both directions without a visible seam.
"""
import argparse
import os
import sys

BASE = os.path.join(os.environ["LOCALAPPDATA"], "Nax Game Restoration")
os.environ.setdefault("HF_HOME", os.path.join(BASE, "hf-cache"))

import torch  # noqa: E402

PROMPT = ("seamless tileable texture, martian ground seen from above, red-brown dusty soil with small rocks, "
          "pebbles, grit and fine cracks, even lighting, highly detailed, photograph")
NEGATIVE = "shadows from one side, perspective, horizon, sky, text, watermark, blurry, large objects"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=1992)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot reach the graphics card.")
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for model in (pipe.unet, pipe.vae):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = "circular"
    generator = torch.Generator("cuda").manual_seed(args.seed)
    image = pipe(prompt=args.prompt, negative_prompt=NEGATIVE, width=args.size, height=args.size,
                 num_inference_steps=args.steps, guidance_scale=6.0, generator=generator).images[0]
    temporary = args.output + ".part"
    image.save(temporary, format="PNG")
    os.replace(temporary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
