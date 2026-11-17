"""Redraws a game's saved sprites and pictures as new high-resolution artwork with an image editing model.

Usage, with the Python of the PyTorch environment described in the README:
    python -I remaster.py [<game>] [--only <prefix>,...] [--out <folder>] [--again] [--seed <n>] [--tries <n>] [--offline]

An upscaler can only sharpen what a 32-pixel sprite already shows, and on such small pictures it mostly
invents noise. This tool instead asks Qwen-Image-Edit-2511, an image editing model, to paint each sprite again
as detailed artwork while keeping its shape, layout and colours. The model sees the sprite enlarged with hard
pixels on a plain background and returns a new picture of the same layout; the part where the sprite was is
cut out, reduced to eight times the original size, and given the original's outline as smooth transparency.

Flat tiles with painted markings that the game draws side by side, such as the four tiles of a landing pad, are
redrawn together as one picture and cut apart again, so the pad keeps its markings running across the tiles
instead of every tile getting edges of its own. The runtime records which sprites lie side by side in
hd\\layout.txt.

Sprites drawn while the screen was black (every visible pixel black) are skipped, because nothing can be
redrawn from a silhouette.

The model runs in 4-bit form from the GGUF files of unsloth/Qwen-Image-Edit-2511-GGUF, with the text encoder of
Qwen/Qwen-Image-Edit-2511 kept in main memory and moved to the graphics card part by part, so it fits a
graphics card with 16 GB. The Lightning adapter of lightx2v/Qwen-Image-Edit-2511-Lightning cuts the work to 8
steps. All three are under the Apache License 2.0 and are loaded from Hugging Face on first use.
"""
import argparse
import glob
import os
import sys

BASE = os.path.join(os.environ["LOCALAPPDATA"], "Nax Game Restoration")
os.environ.setdefault("HF_HOME", os.path.join(BASE, "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain  # noqa: E402
import upscale_sprites  # noqa: E402

REPO = "Qwen/Qwen-Image-Edit-2511"
GGUF = ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf")
LORA = ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
SCALE = 8
CANVAS = 1024
BACKGROUND = (255, 255, 255)

BUILDING_PROMPT = (
    "Remaster this low-resolution pixel-art isometric game sprite from a 1990s science fiction city building game "
    "as a detailed high-resolution isometric illustration. Keep exactly the same object, shape, outline, "
    "perspective, proportions and colours, and keep every window, light and marking where it is. Replace the "
    "blocky pixels with clean, sharp, finely detailed surfaces and smooth edges. Plain white background.")
GROUND_PROMPT = (
    "Remaster this low-resolution pixel-art ground as a detailed high-resolution texture. It is one continuous, "
    "completely flat field of bare Martian soil seen from above at an isometric angle: red-brown dusty ground "
    "with small dark rocks, pebbles and fine cracks. Keep the same colours and the dark rocks where they are. "
    "No tile borders, no grid, no buildings, no windows, no lights, nothing standing up. Plain white "
    "background.")
FLAT_PROMPT = (
    "Remaster this low-resolution pixel-art isometric floor tile as a detailed high-resolution texture. It is a "
    "completely flat dark grey metal platform seen from above at an isometric angle, with painted yellow lines "
    "and small red lamps. It is one piece of a larger landing pad, so its edges continue into the next tiles: "
    "do not draw any frame, border or outline along the edges of the diamond. Keep the same colours, and draw "
    "yellow lines and red lamps only exactly where they are in the original, nowhere else. Paint the yellow "
    "lines as clean, straight, smooth stripes and the red lamps as small round lights, not as pixel steps. No "
    "buildings, no "
    "windows, no extra lights, nothing standing up. Plain white background.")
GROUP_PROMPT = (
    "Remaster this low-resolution pixel-art isometric landing pad as a detailed high-resolution illustration. It "
    "is one completely flat dark grey metal platform seen from above at an isometric angle, with painted yellow "
    "lines and small red lamps. Keep its outline and colours, and keep every yellow line and red lamp exactly "
    "where it is. Paint the yellow lines as clean, straight, smooth stripes and the red lamps as small round "
    "lights, not as pixel steps. Do not add any line, frame or lamp that is not in the original. No buildings, "
    "no windows, nothing standing up. Plain white background.")
PICTURE_PROMPT = (
    "Remaster this low-resolution pixel-art screen of a 1990s science fiction strategy game as a detailed "
    "high-resolution picture. Keep the layout, every shape, line, icon and button exactly where it is, and keep "
    "every colour exactly as it is: the buttons stay grey-blue and the panels stay grey. Do not add any text, "
    "letters or new objects. Turn the grainy dithered grey areas into a finely textured dark stone surface with "
    "the same brightness, render the ornaments and icons crisply, and remove the pixelation and dithering noise.")
# How far colours are taken from the original, in pixels of the result: broader shading and every colour come
# from the original, finer detail from the model. Pictures need it most, because the model tends to recolour
# buttons and panels; sprites keep their colours well and only get a light correction.
PICTURE_COLOUR_RADIUS = 6.0
SPRITE_COLOUR_RADIUS = 10.0


def load_pipeline():
    from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.hooks import apply_group_offloading
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2_5_VLForConditionalGeneration

    transformer = QwenImageTransformer2DModel.from_single_file(
        hf_hub_download(*GGUF), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, config=REPO, subfolder="transformer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(hf_hub_download(*LORA))
    pipe.vae.to("cuda")
    # Neither the transformer (13 GB in 4-bit form) nor the text encoder (16 GB) fits on a 16 GB card next to
    # the working memory, and on Windows an overfull card silently spills into main memory and slows to a
    # crawl. So both stay in main memory and come to the card a few layers at a time, while the previous
    # layers compute.
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    pipe.transformer.enable_group_offload(onload_device=cuda, offload_device=cpu, offload_type="block_level",
                                          num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True)
    apply_group_offloading(pipe.text_encoder, onload_device=cuda, offload_device=cpu,
                           offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True)
    type(pipe)._execution_device = property(lambda self: torch.device("cuda"))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def is_silhouette(rgba):
    visible = rgba[..., 3] > 0
    return not visible.any() or rgba[..., :3][visible].max() < 8


def sprite_canvas(rgba):
    """The sprite enlarged with hard pixels and centred on a plain square canvas, and where it lies there."""
    h, w = rgba.shape[:2]
    factor = max(1, int(CANVAS * 0.75) // max(w, h))
    big = Image.fromarray(rgba, "RGBA").resize((w * factor, h * factor), Image.NEAREST)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), BACKGROUND + (255,))
    x, y = (CANVAS - big.width) // 2, (CANVAS - big.height) // 2
    canvas.alpha_composite(big, (x, y))
    return canvas.convert("RGB"), (x, y, x + big.width, y + big.height)


def prompt_for(rgba):
    """Flat tiles need their own description, or the model builds little towns on them."""
    if rgba.shape[0] > 16:
        return BUILDING_PROMPT
    return GROUND_PROMPT if terrain.is_ground(rgba) else FLAT_PROMPT


def keep_colours(result, original_rgb, radius):
    """The result with the broad colours of the original: the original is enlarged smoothly to the result's
    size, and the result's own broad colours (blurred at `radius`) are replaced by the original's."""
    size = (result.shape[1], result.shape[0])
    base = Image.fromarray(original_rgb, "RGB").resize(size, Image.BILINEAR)
    blur = ImageFilter.GaussianBlur(radius)
    own = np.asarray(Image.fromarray(result, "RGB").filter(blur), np.float32)
    want = np.asarray(base.filter(blur), np.float32)
    return np.clip(result.astype(np.float32) - own + want, 0, 255).astype(np.uint8)


def ground_canvas(rgba):
    """A field of copies of a ground tile, laid out as the game lays out its isometric map, and where the copy
    in the middle lies. Painted as one field, the ground continues across the tile edges, where a single tile
    would get a border of its own."""
    h, w = rgba.shape[:2]
    factor = 12
    big = Image.fromarray(rgba, "RGBA").resize((w * factor, h * factor), Image.NEAREST)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), BACKGROUND + (255,))
    x0, y0 = (CANVAS - big.width) // 2, (CANVAS - big.height) // 2
    step_x, step_y = w // 2 * factor, w // 4 * factor  # neighbouring tiles lie half a tile across, a quarter down
    for i in range(-4, 5):
        for j in range(-4, 5):
            canvas.alpha_composite(big, (x0 + (i + j) * step_x, y0 + (i - j) * step_y))
    return canvas.convert("RGB"), (x0, y0, x0 + big.width, y0 + big.height)


def difference(colour, rgba):
    """How far a redrawn sprite strays from the original: the redrawn picture reduced to the original size by
    averaging, compared with the original over its visible pixels (mean distance per colour channel)."""
    h, w = rgba.shape[:2]
    small = np.asarray(Image.fromarray(colour, "RGB").resize((w, h), Image.BOX), np.float32)
    visible = rgba[..., 3] > 0
    return float(np.abs(small - rgba[..., :3].astype(np.float32))[visible].mean()) if visible.any() else 0.0


def read_layout(path):
    """The pairs of sprites the game drew side by side, as the runtime records them in hd\\layout.txt: the two
    sprites' names and where the second one's bottom row starts, seen from the first one's."""
    pairs = []
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4:
                    try:
                        pairs.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
                    except ValueError:
                        pass  # a line cut short by a crash
    except OSError:
        pass
    return pairs


def is_tile(rgba):
    """Whether a sprite is a whole isometric floor tile: it covers the diamond that fills its rectangle and
    hardly anything outside it. Units, cursors and icons of a similar size have other shapes."""
    h, w = rgba.shape[:2]
    y = np.arange(h)[:, None] + 0.5
    x = np.arange(w)[None, :] + 0.5
    diamond = np.abs(x - w / 2) / (w / 2) + np.abs(y - h / 2) / (h / 2) <= 1.0
    visible = rgba[..., 3] > 0
    union = (diamond | visible).sum()
    return union > 0 and (diamond & visible).sum() / union >= 0.85


def tile_groups(sprites, pairs):
    """Flat marked tiles that the game draws side by side, such as the four tiles of a landing pad, grouped
    together with where each one's bottom row starts. Redrawn alone, each tile would get edges of its own, so a
    pad would look like several small pads. `sprites` maps names to pictures. Returns one dictionary from name
    to position per group of two or more tiles."""
    flat = {n for n, rgba in sprites.items()
            if not is_silhouette(rgba) and prompt_for(rgba) is FLAT_PROMPT and is_tile(rgba)}
    links = {}
    for a, b, dx, dy in pairs:
        if a not in flat or b not in flat or sprites[a].shape != sprites[b].shape:
            continue
        h, w = sprites[a].shape[:2]
        # Only neighbours on the isometric grid: half a tile across and half a tile's height up or down.
        if abs(dx) == w // 2 and abs(dy) == (h + 1) // 2:
            links.setdefault(a, []).append((b, dx, dy))
            links.setdefault(b, []).append((a, -dx, -dy))
    groups, placed = [], set()
    for start in sorted(links):
        if start in placed:
            continue
        group, queue = {start: (0, 0)}, [start]
        while queue:
            name = queue.pop()
            x, y = group[name]
            for other, dx, dy in links[name]:
                # A tile is placed where it was first met. A second, different position would come from a tile
                # used in two places; such a link is left out.
                if other not in group:
                    group[other] = (x + dx, y + dy)
                    queue.append(other)
        placed.update(group)
        groups.append(group)
    return groups


def group_canvas(sprites, group):
    """The tiles of a group enlarged with hard pixels and laid out as the game draws them, centred on a plain
    square canvas, and where each tile lies there."""
    tops = {n: (x, y - sprites[n].shape[0] + 1) for n, (x, y) in group.items()}
    left = min(x for x, _ in tops.values())
    top = min(y for _, y in tops.values())
    right = max(x + sprites[n].shape[1] for n, (x, _) in tops.items())
    bottom = max(y + sprites[n].shape[0] for n, (_, y) in tops.items())
    factor = max(1, int(CANVAS * 0.75) // max(right - left, bottom - top))
    x0 = (CANVAS - (right - left) * factor) // 2
    y0 = (CANVAS - (bottom - top) * factor) // 2
    canvas = Image.new("RGBA", (CANVAS, CANVAS), BACKGROUND + (255,))
    boxes = {}
    for n in sorted(group, key=lambda n: group[n][1]):  # the game draws from the back to the front
        h, w = sprites[n].shape[:2]
        big = Image.fromarray(sprites[n], "RGBA").resize((w * factor, h * factor), Image.NEAREST)
        x, y = x0 + (tops[n][0] - left) * factor, y0 + (tops[n][1] - top) * factor
        canvas.alpha_composite(big, (x, y))
        boxes[n] = (x, y, x + big.width, y + big.height)
    return canvas.convert("RGB"), boxes


def cut(out, box, rgba):
    """The part of the model's picture where a sprite lies, at eight times the sprite's size. The model may
    return a slightly different size than it was given; the sprite's place scales with it."""
    h, w = rgba.shape[:2]
    sx, sy = out.width / CANVAS, out.height / CANVAS
    crop = out.crop((round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy)))
    return np.asarray(crop.resize((w * SCALE, h * SCALE), Image.LANCZOS).convert("RGB"))


def finish(colour, rgba):
    """A redrawn sprite with the broad colours and the outline of the original."""
    # Outside the sprite the original has no colour, so the nearest visible colour is used there.
    filled = upscale_sprites.fill_transparent(rgba[..., :3], rgba[..., 3] > 0)
    colour = keep_colours(colour, np.clip(filled + 0.5, 0, 255).astype(np.uint8), SPRITE_COLOUR_RADIUS)
    alpha = terrain.smooth_alpha(rgba[..., 3], SCALE)
    return np.dstack([colour, alpha])


def remaster_sprite(pipe, rgba, seed, tries=1):
    """Redraws a sprite `tries` times with different seeds and keeps the result closest to the original, since
    the model now and then moves a marking or adds something that is not there."""
    ground = rgba.shape[0] <= 16 and terrain.is_ground(rgba)
    canvas, box = ground_canvas(rgba) if ground else sprite_canvas(rgba)
    best, best_difference = None, None
    for attempt in range(tries):
        colour = cut(redraw(pipe, canvas, prompt_for(rgba), seed + attempt), box, rgba)
        d = difference(colour, rgba)
        if best is None or d < best_difference:
            best, best_difference = colour, d
    return finish(best, rgba)


def remaster_group(pipe, sprites, group, seed, tries=1):
    """Redraws a group of tiles as one picture, `tries` times, keeps the version closest to the originals over
    all its tiles, and cuts each tile out of it. Returns a dictionary from name to redrawn tile."""
    canvas, boxes = group_canvas(sprites, group)
    best, best_difference = None, None
    for attempt in range(tries):
        out = redraw(pipe, canvas, GROUP_PROMPT, seed + attempt)
        colours = {n: cut(out, boxes[n], sprites[n]) for n in group}
        d = sum(difference(colours[n], sprites[n]) for n in group)
        if best is None or d < best_difference:
            best, best_difference = colours, d
    return {n: finish(best[n], sprites[n]) for n in group}


def remaster_picture(pipe, rgba, seed):
    h, w = rgba.shape[:2]
    big = Image.fromarray(rgba[..., :3], "RGB").resize((w * 4, h * 4), Image.NEAREST)
    out = redraw(pipe, big, PICTURE_PROMPT, seed)
    colour = np.asarray(out.resize((w * SCALE, h * SCALE), Image.LANCZOS).convert("RGB"))
    colour = keep_colours(colour, np.ascontiguousarray(rgba[..., :3]), PICTURE_COLOUR_RADIUS)
    return np.dstack([colour, np.full((h * SCALE, w * SCALE), 255, np.uint8)])


def redraw(pipe, image, prompt, seed):
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe(image=[image], prompt=prompt, num_inference_steps=8, true_cfg_scale=1.0, generator=generator).images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", default="utopia")
    parser.add_argument("--only", help="comma-separated beginnings of the sprite names to redraw")
    parser.add_argument("--out", help="the folder for the new pictures, by default the game's hd\\upscaled folder")
    parser.add_argument("--again", action="store_true", help="also redraw sprites that already have a picture")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tries", type=int, default=4,
                        help="how many versions of each sprite are painted; the one closest to the original is kept")
    parser.add_argument("--offline", action="store_true", help="only use models that are already downloaded")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"  # read when the model libraries are first imported, in load_pipeline
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot reach the graphics card.")
    root = os.path.join(BASE, args.game, "hd")
    name_of = lambda path: os.path.splitext(os.path.basename(path))[0]  # noqa: E731
    chosen = {name_of(p) for prefix in (args.only or "").split(",")
              for p in glob.glob(os.path.join(glob.escape(root), "original", prefix + "*.png"))}
    # Every saved sprite is read, because a chosen tile is redrawn together with the tiles next to it.
    sprites = {name_of(p): np.asarray(Image.open(p).convert("RGBA"))
               for p in glob.glob(os.path.join(glob.escape(root), "original", "*.png"))}
    out_dir = args.out or os.path.join(root, "upscaled")
    os.makedirs(out_dir, exist_ok=True)
    upscale_sprites.remove_stale_parts(glob.glob(os.path.join(glob.escape(out_dir), "*.part")))
    target_of = lambda name: os.path.join(out_dir, name + ".png")  # noqa: E731

    def wanted(name):
        return name in chosen and not is_silhouette(sprites[name]) and (args.again or not os.path.exists(target_of(name)))

    groups = tile_groups(sprites, read_layout(os.path.join(root, "layout.txt")))
    grouped = {n for g in groups for n in g}
    groups = [g for g in groups if any(wanted(n) for n in g)]
    singles = [n for n in sorted(chosen) if wanted(n) and n not in grouped]
    total = len(singles) + sum(wanted(n) for g in groups for n in g)
    if not total:
        print("Nothing to redraw.")
        return
    pipe = load_pipeline()
    done = 0

    def save(name, result):
        nonlocal done
        temporary = target_of(name) + ".part"
        Image.fromarray(result, "RGBA").save(temporary, format="PNG")
        upscale_sprites.replace_file(temporary, target_of(name))
        done += 1
        print(f"{done} of {total}: {name}.png", flush=True)

    for name in singles:
        rgba = sprites[name]
        whole = rgba.shape[1] >= 160 and bool((rgba[..., 3] == 255).all())
        save(name, remaster_picture(pipe, rgba, args.seed) if whole else remaster_sprite(pipe, rgba, args.seed, args.tries))
    for group in groups:
        # Tiles of the group that already have a picture keep it unless --again is given, as single sprites do.
        todo = [n for n in sorted(group) if wanted(n)]
        results = remaster_group(pipe, sprites, group, args.seed, args.tries)
        for name in todo:
            save(name, results[name])


if __name__ == "__main__":
    main()
