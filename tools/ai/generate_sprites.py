"""Redraws the sprites a game has saved on the graphics card, at eight times their size.

Usage, with the Python of the PyTorch environment described in the README:
    python -I generate_sprites.py [<game>] [--texture <png>] [--strength 0.35] [--prompt <text>] [--again]

Ground tiles (flat diamonds without bright markings, see terrain.py) get the fine detail of a seamless ground
texture, made with generate_texture.py and kept as hd\\ground.png, so the terrain continues from tile to tile
without seams. Every other sprite from %LOCALAPPDATA%\\Nax Game Restoration\\<game>\\hd\\original is enlarged by
RealESRGAN_x4plus, and with --strength above 0 it then goes through two more steps:
1. Stable Diffusion XL redraws that picture at about 1024 pixels with the ControlNet Tile model, which keeps the
   layout of every part. A low strength means the model only adds detail and does not invent new objects.
2. The result is reduced to eight times the original size, and its broad colours and shading are replaced by
   those of the enlarged picture, so the sprite keeps the game's colours and fits its neighbours, and only the fine detail
   comes from the diffusion model.
The pictures go to hd\\upscaled with the transparency of the original. Sprites that already have a picture
there are skipped unless --again is given. A running game picks the new pictures up within a few seconds.

The models are loaded from Hugging Face on first use and kept in %LOCALAPPDATA%\\Nax Game Restoration\\hf-cache.
"""
import argparse
import glob
import os
import sys

BASE = os.path.join(os.environ["LOCALAPPDATA"], "Nax Game Restoration")
os.environ.setdefault("HF_HOME", os.path.join(BASE, "hf-cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rrdb  # noqa: E402
import terrain  # noqa: E402
import upscale_sprites  # noqa: E402

BORDER = 4
SCALE = 8
PROMPT = ("isometric science fiction colony game art, remastered 1990s computer game sprite, finely detailed, "
          "sharp, clean shapes, high quality")
NEGATIVE = "blurry, noisy, jpeg artifacts, text, watermark, frame, border, signature, deformed"


class GpuRrdb:
    """RealESRGAN_x4plus on the graphics card, with the weights from the restricted loader."""

    def __init__(self, path, device):
        weights, self.blocks = rrdb.load(path)
        self.w = {k: torch.from_numpy(v).to(device, torch.float16) for k, v in weights.items()}
        self.device = device

    def conv(self, name, x):
        return F.conv2d(x, self.w[name + ".weight"], self.w[name + ".bias"], padding=1)

    @torch.no_grad()
    def __call__(self, image):
        x = torch.from_numpy(image).to(self.device, torch.float16).permute(2, 0, 1)[None] / 255
        fea = self.conv("conv_first", x)
        trunk = fea
        for i in range(self.blocks):
            block_in = trunk
            for r in (1, 2, 3):
                parts = [trunk]
                for c in range(1, 5):
                    parts.append(F.leaky_relu(self.conv(f"body.{i}.rdb{r}.conv{c}", torch.cat(parts, 1)), 0.2))
                trunk = self.conv(f"body.{i}.rdb{r}.conv5", torch.cat(parts, 1)) * 0.2 + trunk
            trunk = trunk * 0.2 + block_in
        fea = fea + self.conv("conv_body", trunk)
        for name in ("conv_up1", "conv_up2"):
            fea = F.leaky_relu(self.conv(name, F.interpolate(fea, scale_factor=2, mode="nearest")), 0.2)
        out = self.conv("conv_last", F.leaky_relu(self.conv("conv_hr", fea), 0.2))
        return (out[0].clamp(0, 1).float().permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def load_diffusion(device):
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

    controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-tile-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def broad(picture, radius):
    return np.asarray(Image.fromarray(picture).filter(ImageFilter.GaussianBlur(radius)), np.float32)


def redraw(pipe, esrgan, rgba, args, generator):
    visible = rgba[:, :, 3] > 127
    rgb = upscale_sprites.fill_transparent(rgba[:, :, :3], visible)
    padded = np.clip(np.pad(rgb, ((BORDER, BORDER), (BORDER, BORDER), (0, 0)), mode="edge") + 0.5, 0, 255).astype(np.uint8)
    h, w = padded.shape[:2]
    clean = esrgan(padded)  # four times as large
    target = (w * SCALE, h * SCALE)
    reference = np.asarray(Image.fromarray(clean).resize(target, Image.BICUBIC))
    if args.strength > 0:
        k = 1024 / max(clean.shape[:2])
        size = (max(64, int(round(clean.shape[1] * k / 8)) * 8), max(64, int(round(clean.shape[0] * k / 8)) * 8))
        canvas = Image.fromarray(clean).resize(size, Image.LANCZOS)
        result = pipe(prompt=args.prompt, negative_prompt=NEGATIVE, image=canvas, control_image=canvas,
                      strength=args.strength, num_inference_steps=args.steps, guidance_scale=args.guidance,
                      controlnet_conditioning_scale=args.control, generator=generator).images[0]
        detailed = np.asarray(result.resize(target, Image.LANCZOS), np.float32)
        # Fine detail from the diffusion model on top of the broad colours and shading of the clean picture.
        radius = SCALE * args.colour_blur
        merged = detailed - broad(detailed.astype(np.uint8), radius) + broad(reference, radius)
        reference = np.clip(merged + 0.5, 0, 255).astype(np.uint8)
    cut = reference[BORDER * SCALE:-BORDER * SCALE, BORDER * SCALE:-BORDER * SCALE]
    alpha = terrain.smooth_alpha(rgba[:, :, 3], SCALE)
    return np.concatenate([cut, alpha[:, :, None]], axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", default="utopia")
    parser.add_argument("--strength", type=float, default=0.0,
                        help="how much the diffusion model may change sprites other than ground tiles, 0 to 1")
    parser.add_argument("--paint-texture", action="store_true",
                        help="paint the ground texture with generate_texture.py first when it does not exist yet")
    parser.add_argument("--texture", help="the seamless ground texture, by default ground.png in the game's hd folder")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--control", type=float, default=0.8, help="how closely the redraw follows the clean picture's layout")
    parser.add_argument("--guidance", type=float, default=5.0, help="how closely the redraw follows the prompt")
    parser.add_argument("--colour-blur", type=float, default=1.0,
                        help="how broad the colours taken from the clean picture are, in original pixels")
    parser.add_argument("--out", help="write the pictures to this folder instead of the upscaled folder, for trying settings")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=1992)
    parser.add_argument("--again", action="store_true", help="also redo sprites that already have an upscaled picture")
    parser.add_argument("--only", help="only sprites whose name starts with this text")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot reach the graphics card.")
    root = os.path.join(BASE, args.game, "hd")
    originals = sorted(p for prefix in (args.only or "").split(",")
                       for p in glob.glob(os.path.join(glob.escape(root), "original", prefix + "*.png")))
    if not originals:
        sys.exit(f"No saved sprites in {os.path.join(root, 'original')}. Play the game for a while first.")
    out_dir = args.out or os.path.join(root, "upscaled")
    os.makedirs(out_dir, exist_ok=True)
    texture_path = args.texture or os.path.join(root, "ground.png")
    # Pictures left half-written by a run that was stopped: this tool's sprites and its ground texture.
    upscale_sprites.remove_stale_parts(glob.glob(os.path.join(glob.escape(out_dir), "*.part")) + [texture_path + ".part"])
    device = "cuda"
    esrgan = GpuRrdb(os.path.join(BASE, "models", "RealESRGAN_x4plus.pth"), device)
    pipe = load_diffusion(device) if args.strength > 0 else None
    if args.paint_texture and not os.path.exists(texture_path):
        try:
            import generate_texture
            print("Painting the ground texture", flush=True)
            generate_texture.make_texture(texture_path)
        except Exception as error:  # noqa: BLE001 - without a texture the ground tiles are simply enlarged
            print(f"The ground texture could not be painted: {error}", flush=True)
    ground = terrain.GroundTexture(np.asarray(Image.open(texture_path).convert("RGB"))) if os.path.exists(texture_path) else None
    done = 0
    for path in originals:
        target = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(target) and not args.again:
            continue
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        if ground is not None and terrain.is_ground(rgba):
            result = terrain.ground_tile(rgba, ground)
        else:
            generator = torch.Generator(device).manual_seed(args.seed)
            result = redraw(pipe, esrgan, rgba, args, generator)
        temporary = target + ".part"
        Image.fromarray(result, "RGBA").save(temporary, format="PNG")
        upscale_sprites.replace_file(temporary, target)
        done += 1
        print(f"{done} sprites redrawn", flush=True)
    print(f"Redrew {done} of {len(originals)} sprites.")


if __name__ == "__main__":
    main()
