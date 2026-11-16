"""Upscales the sprites a game has saved, so the AI filter can draw them in high resolution.

Usage: python -I upscale_sprites.py [<game>] [--model <file>] [--again]

While a game runs, the runtime saves every sprite it draws once as a PNG in
%LOCALAPPDATA%\\Nax Game Restoration\\<game>\\hd\\original (the game defaults to utopia). This tool runs the
AI network on each of them and writes a picture four times as large, with the same name, to hd\\upscaled.
Sprites that already have an upscaled picture are skipped unless --again is given, so pictures edited by hand
are kept. A running game picks the new pictures up within a few seconds.

The model is an ESRGAN checkpoint (.pth) such as RealESRGAN_x4plus.pth, run by rrdb.py, or a weight file of the
small realesr-animevideov3 network (.bin). Without --model the tool uses RealESRGAN_x4plus.pth from
%LOCALAPPDATA%\\Nax Game Restoration\\models when it is there, and otherwise the small network from the models
folder of this repository.

The network has never seen what lies around a sprite, and a transparent pixel would read to it as black, which
gives dark fringes. So before upscaling, every transparent pixel takes the colour of its nearest visible
neighbours, and the sprite gets a border that repeats its edge. The transparency itself is enlarged as it is.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_upscale as reference  # noqa: E402
import rrdb  # noqa: E402

BORDER = 4


def fill_transparent(rgb, visible):
    """Gives every transparent pixel the average colour of its visible neighbours, growing outward."""
    rgb = rgb.astype(np.float32)
    known = visible.copy()
    if not known.any():
        return rgb
    height, width = known.shape
    while not known.all():
        total = np.zeros_like(rgb)
        count = np.zeros((height, width), np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted_known = np.zeros_like(known)
            shifted_rgb = np.zeros_like(rgb)
            ys = slice(max(dy, 0), height + min(dy, 0))
            yd = slice(max(-dy, 0), height + min(-dy, 0))
            xs = slice(max(dx, 0), width + min(dx, 0))
            xd = slice(max(-dx, 0), width + min(-dx, 0))
            shifted_known[yd, xd] = known[ys, xs]
            shifted_rgb[yd, xd] = rgb[ys, xs]
            total += shifted_rgb * shifted_known[:, :, None]
            count += shifted_known
        grow = ~known & (count > 0)
        rgb[grow] = total[grow] / count[grow][:, None]
        known = known | grow
    return rgb


def load_model(path):
    """Returns a function that upscales an RGB picture four times."""
    if path.lower().endswith(".pth"):
        net = rrdb.load(path)
        return lambda image: rrdb.upscale_image(net, image)
    layers, scale = reference.load_weights(path)
    if scale != 4:
        raise ValueError("the sprite pictures must be four times as large as the originals")
    return lambda image: reference.upscale_image(layers, scale, image)


def upscale_sprite(model, rgba, scale=4):
    visible = rgba[:, :, 3] > 127
    rgb = fill_transparent(rgba[:, :, :3], visible)
    padded = np.pad(rgb, ((BORDER, BORDER), (BORDER, BORDER), (0, 0)), mode="edge")
    big = model(np.clip(padded + 0.5, 0, 255).astype(np.uint8))
    big = big[BORDER * scale:-BORDER * scale, BORDER * scale:-BORDER * scale]
    alpha = np.repeat(np.repeat(rgba[:, :, 3], scale, axis=0), scale, axis=1)
    return np.concatenate([big, alpha[:, :, None]], axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", default="utopia")
    parser.add_argument("--model", help="an ESRGAN checkpoint (.pth) or a realesr-animevideov3 weight file (.bin)")
    parser.add_argument("--again", action="store_true", help="also redo sprites that already have an upscaled picture")
    args = parser.parse_args()
    base = os.path.join(os.environ["LOCALAPPDATA"], "Nax Game Restoration")
    root = os.path.join(base, args.game, "hd")
    model_path = args.model or os.path.join(base, "models", "RealESRGAN_x4plus.pth")
    if not args.model and not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "realesr-animevideov3.bin")
    originals = sorted(glob.glob(os.path.join(root, "original", "*.png")))
    if not originals:
        sys.exit(f"No saved sprites in {os.path.join(root, 'original')}. Play the game for a while first.")
    os.makedirs(os.path.join(root, "upscaled"), exist_ok=True)
    model = load_model(model_path)
    print(f"Model: {os.path.basename(model_path)}")
    done = 0
    for path in originals:
        target = os.path.join(root, "upscaled", os.path.basename(path))
        if os.path.exists(target) and not args.again:
            continue
        result = upscale_sprite(model, reference.read_png(path, alpha=True))
        # Written under another name first, so the running game never reads a half-written picture.
        temporary = target + ".part"
        reference.write_png(temporary, result)
        os.replace(temporary, target)
        done += 1
        print(f"\r{done} sprites upscaled", end="", flush=True)
    print("\r", end="")
    print(f"Upscaled {done} of {len(originals)} sprites.")


if __name__ == "__main__":
    main()
