"""Upscales the sprites a game has saved, so the AI filter can draw them in high resolution.

Usage: python -I upscale_sprites.py [<game>] [--weights <file>] [--again]

While a game runs, the runtime saves every sprite it draws once as a PNG in
%LOCALAPPDATA%\\Nax Game Restoration\\<game>\\hd\\original (the game defaults to utopia). This tool runs the
AI network on each of them and writes a picture four times as large, with the same name, to hd\\upscaled.
Sprites that already have an upscaled picture are skipped unless --again is given, so pictures edited by hand
are kept. A running game picks the new pictures up within a few seconds.

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


def upscale_sprite(layers, scale, rgba):
    visible = rgba[:, :, 3] > 127
    rgb = fill_transparent(rgba[:, :, :3], visible)
    padded = np.pad(rgb, ((BORDER, BORDER), (BORDER, BORDER), (0, 0)), mode="edge")
    big = reference.upscale_image(layers, scale, np.clip(padded + 0.5, 0, 255).astype(np.uint8))
    big = big[BORDER * scale:-BORDER * scale, BORDER * scale:-BORDER * scale]
    alpha = np.repeat(np.repeat(rgba[:, :, 3], scale, axis=0), scale, axis=1)
    return np.concatenate([big, alpha[:, :, None]], axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", default="utopia")
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models",
                                                          "realesr-animevideov3.bin"))
    parser.add_argument("--again", action="store_true", help="also redo sprites that already have an upscaled picture")
    args = parser.parse_args()
    root = os.path.join(os.environ["LOCALAPPDATA"], "Nax Game Restoration", args.game, "hd")
    originals = sorted(glob.glob(os.path.join(root, "original", "*.png")))
    if not originals:
        sys.exit(f"No saved sprites in {os.path.join(root, 'original')}. Play the game for a while first.")
    os.makedirs(os.path.join(root, "upscaled"), exist_ok=True)
    layers, scale = reference.load_weights(args.weights)
    done = 0
    for path in originals:
        target = os.path.join(root, "upscaled", os.path.basename(path))
        if os.path.exists(target) and not args.again:
            continue
        result = upscale_sprite(layers, scale, reference.read_png(path, alpha=True))
        # Written under another name first, so the running game never reads a half-written picture.
        temporary = target + ".part"
        reference.write_png(temporary, result)
        os.replace(temporary, target)
        done += 1
    print(f"Upscaled {done} of {len(originals)} sprites.")


if __name__ == "__main__":
    main()
