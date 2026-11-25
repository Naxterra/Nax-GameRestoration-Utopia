"""Redraws a game's saved sprites and pictures as new high-resolution artwork with an image editing model.

Usage, with the Python of the PyTorch environment described in the README:
    python -I remaster.py [<game>] [--only <prefix>,...] [--out <folder>] [--again] [--seed <n>] [--tries <n>] [--offline]

An upscaler can only sharpen what a 32-pixel sprite already shows, and on such small pictures it mostly
invents noise. This tool instead asks Qwen-Image-Edit-2511, an image editing model, to paint each sprite again
as detailed artwork while keeping its shape, layout and colours. The model sees the sprite enlarged with hard
pixels on a plain background and returns a new picture of the same layout; the part where the sprite was is
cut out, reduced to eight times the original size, and given the original's outline as smooth transparency.

Flat tiles with painted markings that the game draws side by side, such as the four tiles of a landing pad, and
the pieces of a building that stands on several tiles, such as the command center, are redrawn together as one
picture and cut apart again, so the pad keeps its markings running across the tiles and the building stays
one building, instead of every piece getting edges of its own. The runtime records which sprites lie side by side in
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
import shutil
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
    "Remaster this low-resolution 1990s pixel-art isometric sci-fi building sprite into modern 24-bit true-color "
    "high-resolution isometric game art. Keep the exact structure, silhouette, outline, perspective, proportions and "
    "functional features (windows, domes, vents, lights and markings) in the same places, while upgrading the harsh "
    "retro 16-color palette into rich natural modern materials with realistic directional lighting, soft ambient "
    "occlusion, subtle metallic reflections, weathered industrial plating and warm glowing windows. Clean sharp "
    "surfaces and smooth edges. Plain white background.")
ICON_PROMPT = (
    "Remaster this low-resolution pixel-art button icon from a 1990s science fiction strategy game as a crisp, "
    "modern 24-bit true-color engraved brushed-metal user-interface icon. Keep the exact symbol, shape, outline "
    "and proportions, replacing blocky pixel stair-steps with smooth sharp lines, subtle metallic bevel lighting "
    "and soft recessed shadows. Do not add extra text or objects. Plain white background.")
GROUND_PROMPT = (
    "Remaster this low-resolution pixel-art ground into clean modern 24-bit true-color painterly isometric sci-fi "
    "game terrain. It is one continuous, completely flat expanse of dusty terracotta and rust-ochre Martian desert "
    "regolith seen from above at an isometric angle, with rich natural mineral color gradients, gentle wind-carved "
    "ripples and soft natural directional sunlight shading. Avoid noisy black speckles or harsh scattered dots. "
    "No tile borders, no grid, no buildings, no windows, no lights, nothing standing up. Plain white background.")
LANDING_PAD_PROMPT = (
    "Remaster this low-resolution pixel-art isometric floor tile into modern 24-bit true-color high-resolution "
    "game art. It is a completely flat weathered dark brushed-steel landing pad seen from above at an isometric "
    "angle, with subtle metallic sheen, realistic panel seams, crisp painted safety-yellow lines and small glowing "
    "red beacon lamps. It is one piece of a larger landing pad, so its edges continue into the next tiles: do not "
    "draw any frame, border or outline along the edges of the diamond. Draw yellow lines and red lamps only exactly "
    "where they are in the original, nowhere else. No buildings, no windows, nothing standing up. Plain white "
    "background.")
FLAT_PROMPT = (
    "Remaster this low-resolution pixel-art isometric floor tile into modern 24-bit true-color high-resolution "
    "isometric sci-fi surface art. Upgrade the harsh retro palette into realistic weathered materials with smooth "
    "24-bit tonal gradients and subtle directional lighting while keeping the overall colour family, markings, "
    "lines and patterns exactly where they are. Do not draw any frame or outline along the edges of the diamond, "
    "and do not invent extra lights, buildings or objects. Plain white background.")
LANDING_PAD_GROUP_PROMPT = (
    "Remaster this low-resolution pixel-art isometric landing pad into modern 24-bit true-color high-resolution "
    "game art. It is one continuous flat weathered dark brushed-steel landing platform seen from above at an "
    "isometric angle, with realistic metal plating, subtle specular highlights, crisp painted safety-yellow stripes "
    "and small glowing red beacon lights. Keep its exact outline and keep every yellow line and red lamp exactly "
    "where it is without adding any extra line, frame or lamp. No buildings, no windows, nothing standing up. "
    "Plain white background.")
GROUP_PROMPT = (
    "Remaster this low-resolution pixel-art multi-tile isometric floor structure into one continuous modern 24-bit "
    "true-color high-resolution sci-fi surface illustration. Upgrade the harsh retro palette into realistic "
    "weathered materials with rich tonal gradients and subtle directional lighting while keeping its exact outline, "
    "colour family, lines and markings where they are. No extra frames, lights or buildings. Plain white background.")
PICTURE_PROMPT = (
    "Remaster this low-resolution 1990s pixel-art science fiction strategy game screen into crisp modern 24-bit "
    "true-color high-resolution game art. Keep the exact layout, every panel, frame, border, icon, arrow, emblem "
    "and portrait in the exact same position. Upgrade the harsh retro palette into rich modern materials (brushed "
    "gunmetal steel, carved dark stone, glowing metallic trim, realistic atmospheric lighting and smooth tonal "
    "gradients) while keeping each element recognizable in its original colour family. Replace all blocky pixel "
    "stair-steps on buttons, icons, arrows, frames and ornaments with smooth, sharp, finely detailed artwork. "
    "Do not add any new text, letters or objects.")
# How far broad colours are anchored toward the original, in pixels of the 8x result. Both radii are wider than
# one original pixel (8 px), and gentle luma/chroma blend strengths let Qwen upgrade harsh 16-color VGA tones into
# rich 24-bit (16.7 million colour) materials and lighting while keeping each building and biome recognizable.
PICTURE_COLOUR_RADIUS = 18.0
SPRITE_COLOUR_RADIUS = 12.0
PICTURE_LUMA_STRENGTH = 0.30
PICTURE_CHROMA_STRENGTH = 0.45
SPRITE_LUMA_STRENGTH = 0.25
SPRITE_CHROMA_STRENGTH = 0.35
PICTURE_SIZE = (1536, 960)  # exact 16:10 aspect ratio (320x200 * 4.8) and multiples of 64 for clean diffusion


def _ground_biome(rgba):
    """Classifies a ground tile's dominant biome from its visible pixels so non-Mars planets keep their look."""
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    if len(rgb) == 0:
        return "mars"
    r, g, b = rgb.mean(axis=0)
    lum = (r + g + b) / (3.0 * 255.0)
    if lum > 0.55 and b >= r - 10:
        return "ice"
    if b > r + 15 and b > g + 10:
        return "ice" if g >= r else "crystal"
    if r > g + 18 and b > g + 15:
        return "crystal"
    if g > r + 10 and g >= b:
        return "green"
    if r > g + 20 and r > b + 20:
        return "mars"
    return "neutral"


def ground_prompt_for(rgba):
    biome = _ground_biome(rgba)
    if biome == "mars":
        return GROUND_PROMPT
    surfaces = {
        "green": "lush natural alien grassland and mossy soil with rich olive, emerald and earth tonal gradients and soft sunlit shading",
        "ice": "frozen arctic snowfield and glacial ice tundra with realistic white snowdrifts, pale cyan-blue ice sheen, subtle wind-carved texture and soft cool-blue shadows (upgrading flat purple-blue retro pixels into natural realistic snow and ice)",
        "crystal": "alien crystalline mineral plains with smooth faceted violet-slate stone, subtle iridescent sheen and soft atmospheric shadows",
        "neutral": "natural alien planetary terrain with rich weathered soil and rock tonal gradients and soft directional sunlight shading",
    }
    return (
        f"Remaster this low-resolution pixel-art ground into clean modern 24-bit true-color painterly isometric "
        f"sci-fi game terrain. It is one continuous, completely flat expanse of {surfaces[biome]} seen from above at "
        f"an isometric angle. Avoid noisy black speckles or harsh scattered dots. No tile borders, no grid, no "
        f"buildings, no windows, no lights, nothing standing up. Plain white background.")


def is_landing_pad(rgba):
    """Whether a flat marked tile looks like Utopia's dark-grey metal landing pad with yellow painted markings."""
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    if len(rgb) == 0:
        return False
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    grey = (np.abs(r - g) < 28) & (np.abs(g - b) < 28) & (r < 145)
    yellow = (r > 180) & (g > 150) & (b < 110)
    return bool(grey.mean() >= 0.35 and yellow.any())


def is_low_structure(rgba):
    """Whether a short (h <= 16) tile is actually a 3D bunker or laser turret dome (grey metallic dome with a
    red ring, no yellow landing-pad stripes) that should be remastered with BUILDING_PROMPT rather than flat."""
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    if len(rgb) == 0:
        return False
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    grey = (np.abs(r - g) < 28) & (np.abs(g - b) < 28) & (r < 180)
    red = (r > 155) & (g < 65) & (b < 65)
    yellow = (r > 180) & (g > 150) & (b < 110)
    return bool(grey.mean() >= 0.45 and red.mean() >= 0.05 and not yellow.any())


def is_chasm_edge(rgba):
    """Whether a short (h <= 16) isometric tile is a cliff edge overhanging a pitch-black planetary abyss
    (such as frozen glacial shelves with hanging icicles over pure black void)."""
    h, w = rgba.shape[:2]
    if h > 16 or w <= 20 or w >= 160:
        return False
    vis = rgba[..., 3] > 127
    if not vis.any():
        return False
    rgb = rgba[..., :3][vis].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    black = (r < 10) & (g < 10) & (b < 10)
    if black.mean() < 0.08 or black.mean() > 0.92:
        return False
    # Exclude dark-steel landing pads (which have yellow stripes) and grey turrets (which have red rings).
    if ((r > 180) & (g > 150) & (b < 110)).any() or ((r > 155) & (g < 65) & (b < 65)).mean() >= 0.05:
        return False
    return is_tile(rgba)


def chasm_prompt_for(rgba):
    """Picks a biome-appropriate prompt for isometric cliff/icicle chasm edge tiles overlooking a pitch-black abyss."""
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    non_black = rgb[rgb.max(axis=1) >= 16]
    if len(non_black) > 0:
        r, g, b = non_black.mean(axis=0)
        is_ice = (b > r + 10 and b > g + 5) or (non_black.mean() > 140 and b >= r - 5)
    else:
        is_ice = False
    if is_ice:
        edge = (
            "a jagged frozen glacial snow shelf and pale cyan-blue ice cliff edge with sharp crystalline icicles "
            "hanging over a pitch-black abyss (upgrading flat lavender-blue retro pixels into realistic white snow "
            "and translucent pale-blue ice)"
        )
    else:
        edge = (
            "a rugged alien planetary rock cliff edge overhanging a pitch-black chasm abyss, with realistic weathered "
            "stone strata and sharp natural shadow detail"
        )
    return (
        f"Remaster this low-resolution 1990s pixel-art isometric cliff-edge tile into modern 24-bit true-color "
        f"high-resolution isometric game art. It is {edge}. Keep the exact jagged cliff boundary and keep the "
        f"black abyss region strictly solid pure black (#000000) with zero glow, haze, or objects inside the void. "
        f"No tile borders, no grid, no buildings, no lights. Plain white background."
    )


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


def is_natural_formation(rgba):
    """Whether a tall (h > 16) sprite is a natural mountain, glacier, snow ridge, rock spire, or alien flora
    rather than a man-made colony building, preventing Qwen from painting sci-fi domes and windows onto cliffs."""
    h, w = rgba.shape[:2]
    if h <= 16 or w <= 20 or w >= 160:
        return False
    vis = rgba[..., 3] > 127
    if not vis.any():
        return False
    rgb = rgba[..., :3][vis].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Man-made architectural cues: glowing yellow windows/lights, bright red roofs/beacons, manicured green lawns/roofs.
    if ((r > 175) & (g > 155) & (b < 85)).sum() >= 2:
        return False
    if ((r > 155) & (g < 65) & (b < 65)).sum() >= 3:
        return False
    if ((g > 100) & (g > r + 25) & (g > b + 25)).mean() >= 0.04:
        return False
    # Require a genuine cool blue-violet channel bias so neutral grey/silver metallic buildings are never treated as ice.
    ice_blue = (b > r + 15) & (b > g + 10) & (b > 100)
    ice_pale = (b > 130) & (r > 110) & (b > r + 6) & (b > g + 4)
    ice_px = ice_blue | ice_pale
    pink_flora = (r > 150) & (b > 140) & (g < r - 30)
    dark_rock = (np.maximum(np.maximum(r, g), b) < 115) & (np.abs(r - g) < 28)
    if ice_blue.mean() >= 0.20 and ice_px.mean() >= 0.35 and (ice_px | pink_flora | dark_rock).mean() >= 0.85:
        return True
    mars_px = (r > g + 15) & (g >= b) & (r >= 70) & (r <= 188) & (g <= 110) & (b <= 35)
    mars_shadow = (r < 75) & (g < 60) & (b < 50)
    if mars_px.mean() >= 0.45 and (mars_px | mars_shadow | dark_rock).mean() >= 0.85:
        return True
    return False


def mountain_prompt_for(rgba):
    """Picks a biome-appropriate natural geological prompt for tall mountain, glacier, rock, or flora sprites."""
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    pink_flora = ((r > 150) & (b > 140) & (g < r - 30)).mean()
    ice_px = (((b > r + 15) & (b > g + 10) & (b > 100)) | ((b > 130) & (r > 110) & (b >= r - 5))).mean()
    if pink_flora >= 0.05:
        subject = (
            "a frozen alien ice-and-rock mound covered with delicate pink-violet crystalline alien flora and frost, "
            "upgrading flat retro pixels into realistic snow, dark stone and glowing natural crystals"
        )
    elif ice_px >= 0.35:
        subject = (
            "a rugged natural frozen glacier cliff and snow-covered mountain ridge with jagged dark arctic rock "
            "draped in thick white snowdrifts and pale cyan-blue glacial ice, upgrading flat lavender-blue retro "
            "pixels into realistic natural snow, ice and stone"
        )
    else:
        subject = (
            "a rugged natural Martian sandstone mountain cliff and eroded terracotta rock formation with warm "
            "rust-ochre mineral strata, weathered canyon crevices and realistic directional sunlight shadows"
        )
    return (
        f"Remaster this low-resolution 1990s pixel-art isometric natural terrain formation into modern 24-bit "
        f"true-color high-resolution isometric game art. It is {subject}. Keep the exact silhouette, outline, "
        f"perspective and proportions. Strictly natural geology only: no buildings, no windows, no domes, no "
        f"metal structures, no artificial lights. Plain white background."
    )


def is_pink_flora_ground(rgba):
    """Whether a short (h <= 16) isometric tile is an ice/rock terrain patch covered in pink-violet alien flora."""
    h, w = rgba.shape[:2]
    if h > 16 or w <= 20 or w >= 160:
        return False
    vis = rgba[..., 3] > 127
    if not vis.any():
        return False
    rgb = rgba[..., :3][vis].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    pink = (r > 150) & (b > 140) & (g < r - 30)
    return bool(pink.mean() >= 0.05 and is_tile(rgba))


def is_toxic_lake(rgba):
    """Whether a short (h <= 16) isometric tile is an emerald-green alien toxic acid lake or shoreline.
    Requires vivid near-zero-red/blue acid green so natural olive/forest grassland biomes stay grassland."""
    h, w = rgba.shape[:2]
    if h > 16 or w <= 20 or w >= 160:
        return False
    vis = rgba[..., 3] > 127
    if not vis.any():
        return False
    rgb = rgba[..., :3][vis].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if ((r > 180) & (g > 150) & (b < 100)).any() or ((r > 155) & (g < 65) & (b < 65)).mean() >= 0.04:
        return False
    acid_green = (g > 55) & (g > r + 30) & (g > b + 30) & (r < 35) & (b < 35)
    return bool(acid_green.mean() >= 0.25 and is_tile(rgba))


def lake_prompt_for(rgba):
    """Picks a natural organic alien acid-lake liquid or snow-shoreline prompt so lakes never turn into sci-fi circuit panels."""
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    acid_green = (g > 55) & (g > r + 30) & (g > b + 30) & (r < 35) & (b < 35)
    if acid_green.mean() >= 0.85:
        subject = (
            "a seamless flat surface of glowing emerald-green alien toxic acid lake water with subtle natural liquid "
            "ripples, translucent emerald eddies and deep mossy-green chemical currents"
        )
    else:
        subject = (
            "the natural organic shoreline of a glowing emerald-green alien toxic acid pool meeting thick pale icy "
            "blue-white snowbanks and frosted dark rocky shore (upgrading flat lavender-blue retro pixels into "
            "realistic white snow and flat green retro pixels into natural shimmering emerald liquid)"
        )
    return (
        f"Remaster this low-resolution 1990s pixel-art isometric natural terrain tile into modern 24-bit true-color "
        f"high-resolution isometric game art. It is {subject}. Keep the exact shoreline contour, perspective and "
        f"proportions. Strictly natural alien liquid, snow and rock only: zero metal plates, zero sci-fi floor "
        f"panels, zero circuit lines, zero buildings, no tile borders, no grid. Plain white background."
    )


COMPASS_MARKER_PROMPT = (
    "Remaster this low-resolution 1990s pixel-art sci-fi user-interface compass marker into crisp modern 24-bit "
    "true-color game UI art. Keep the exact flat dark brushed-gunmetal isometric tile base and the exact bold "
    "crimson anodized-metal directional arrow/numeral emblem standing cleanly in the center with smooth sharp "
    "bevelled edges and subtle metallic reflections. Strictly a clean minimal UI emblem on a flat tile: zero "
    "buildings, zero skyscrapers, zero windows, zero lights, zero bolts, zero extra objects. Plain white background.")


def is_compass_marker(rgba):
    """Whether a 32xN sprite is Utopia's UI compass/sector marker (flat lavender/grey diamond base + bold pure-red
    numeral/emblem with black outline, strictly <= 9 colours and zero yellow/green/brown building detail)."""
    h, w = rgba.shape[:2]
    if w != 32 or h <= 16 or h > 30:
        return False
    vis = rgba[..., 3] > 127
    if not vis.any():
        return False
    rgb = rgba[..., :3][vis]
    uniq = len(np.unique(rgb.reshape(-1, 3), axis=0))
    if uniq > 9:
        return False
    rf, gf, bf = rgb[:, 0].astype(np.float32), rgb[:, 1].astype(np.float32), rgb[:, 2].astype(np.float32)
    if ((rf > 175) & (gf > 155) & (bf < 85)).any():
        return False
    red = (rf > 95) & (gf < 20) & (bf < 20)
    lav_or_grey = (np.abs(rf - gf) < 25) & (bf >= gf - 5) & (bf > 85)
    black = (rf < 15) & (gf < 15) & (bf < 15)
    return bool(red.mean() >= 0.08 and lav_or_grey.mean() >= 0.45 and (red | lav_or_grey | black).mean() >= 0.98)


def render_compass_marker(orig_rgba, scale=SCALE):
    """Renders Utopia's compass/sector indicator cleanly at 8x as a brushed dark-slate isometric diamond tile
    with a crisp bevelled crimson anodized-metal emblem and zero skyscraper hallucination or white halo."""
    h, w = orig_rgba.shape[:2]
    H, W = h * scale, w * scale
    vis = orig_rgba[..., 3] > 0
    rgb = orig_rgba[..., :3].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_low = (vis & (r > 95) & (g < 20) & (b < 20)).astype(np.float32)
    blk_low = (vis & (r < 15) & (g < 15) & (b < 15)).astype(np.float32)

    # Smooth upscaled masks for the red emblem and its crisp black outline.
    def up_mask(m, blur_r=1.6):
        arr = np.asarray(
            Image.fromarray((m * 255.0).astype(np.uint8), "L")
            .resize((W, H), Image.BICUBIC)
            .filter(ImageFilter.GaussianBlur(blur_r)),
            np.float32,
        ) / 255.0
        return np.clip((arr - 0.35) / 0.30, 0.0, 1.0)

    red_hi = up_mask(red_low, 1.4)
    blk_hi = up_mask(blk_low, 1.2)
    # Brushed gunmetal-slate diamond base matching the HD compass pad (subtle diagonal sheen + bevel gradient).
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sheen = 0.5 + 0.5 * np.sin((xx * 0.045 + yy * 0.08)) * 0.12 - (yy / float(H)) * 0.18
    base = np.stack([
        np.clip(78.0 + 38.0 * sheen, 45.0, 130.0),
        np.clip(82.0 + 38.0 * sheen, 48.0, 135.0),
        np.clip(102.0 + 44.0 * sheen, 62.0, 158.0),
    ], axis=-1)

    # Crisp crimson anodized-metal emblem with top-left directional highlight.
    red_grad = np.clip(1.0 - (yy / float(H)) * 0.45 + (1.0 - xx / float(W)) * 0.20, 0.45, 1.20)
    emblem = np.stack([
        np.clip(215.0 * red_grad, 110.0, 245.0),
        np.clip(32.0 * red_grad, 14.0, 62.0),
        np.clip(36.0 * red_grad, 16.0, 68.0),
    ], axis=-1)

    outline = np.array([14.0, 16.0, 22.0], np.float32)
    out = base * (1.0 - blk_hi[..., None]) + outline * blk_hi[..., None]
    out = out * (1.0 - red_hi[..., None]) + emblem * red_hi[..., None]
    alpha = terrain.smooth_alpha(orig_rgba[..., 3], scale)
    return np.dstack([np.clip(out + 0.5, 0, 255).astype(np.uint8), alpha])


def prompt_for(rgba):
    """Picks a prompt matched to the sprite's role so icons, ground biomes, mountains, chasms, lakes, pads, and buildings stay faithful."""
    h, w = rgba.shape[:2]
    if w <= 20 and h <= 20:
        return ICON_PROMPT
    if h > 16:
        if is_compass_marker(rgba):
            return COMPASS_MARKER_PROMPT
        return mountain_prompt_for(rgba) if is_natural_formation(rgba) else BUILDING_PROMPT
    if is_low_structure(rgba):
        return BUILDING_PROMPT
    if is_chasm_edge(rgba):
        return chasm_prompt_for(rgba)
    if is_pink_flora_ground(rgba):
        return mountain_prompt_for(rgba)
    if is_toxic_lake(rgba):
        return lake_prompt_for(rgba)
    if terrain.is_ground(rgba):
        return ground_prompt_for(rgba)
    if is_landing_pad(rgba):
        return LANDING_PAD_PROMPT
    return FLAT_PROMPT


def _rgb_to_ycbcr(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def _ycbcr_to_rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def keep_colours(result, original_rgb, radius, luma_strength=0.6, chroma_strength=1.0):
    """Gently anchors the model's broad colours toward the original while keeping rich 24-bit shading and detail.

    Broad luminance (Y) is blended by `luma_strength` and broad chrominance (Cb, Cr) by `chroma_strength` at
    `radius`, letting modern 24-bit lighting, metallic reflections and natural material tones shine through
    without snapping back to a flat 16-colour VGA palette.
    """
    size = (result.shape[1], result.shape[0])
    base = Image.fromarray(original_rgb, "RGB").resize(size, Image.BILINEAR)
    blur = ImageFilter.GaussianBlur(radius)
    own_rgb = np.asarray(Image.fromarray(result, "RGB").filter(blur), np.float32)
    want_rgb = np.asarray(base.filter(blur), np.float32)
    res_ycc = _rgb_to_ycbcr(result.astype(np.float32))
    own_ycc = _rgb_to_ycbcr(own_rgb)
    want_ycc = _rgb_to_ycbcr(want_rgb)
    out_ycc = res_ycc.copy()
    out_ycc[..., 0] = res_ycc[..., 0] + luma_strength * (want_ycc[..., 0] - own_ycc[..., 0])
    out_ycc[..., 1:] = res_ycc[..., 1:] + chroma_strength * (want_ycc[..., 1:] - own_ycc[..., 1:])
    return np.clip(_ycbcr_to_rgb(out_ycc) + 0.5, 0, 255).astype(np.uint8)


def ground_canvas(rgba):
    """A field of copies of a ground tile, laid out as the game lays out its isometric map, and where the copy
    in the middle lies. Softening the enlarged field prevents 1-pixel shadow dots from turning into harsh black
    squares, while painting as one field avoids seams across tile edges."""
    h, w = rgba.shape[:2]
    factor = 12
    big = Image.fromarray(rgba, "RGBA").resize((w * factor, h * factor), Image.NEAREST)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), BACKGROUND + (255,))
    x0, y0 = (CANVAS - big.width) // 2, (CANVAS - big.height) // 2
    step_x, step_y = w // 2 * factor, w // 4 * factor  # neighbouring tiles lie half a tile across, a quarter down
    for i in range(-4, 5):
        for j in range(-4, 5):
            canvas.alpha_composite(big, (x0 + (i + j) * step_x, y0 + (i - j) * step_y))
    soft = canvas.convert("RGB").filter(ImageFilter.GaussianBlur(1.0))
    return soft, (x0, y0, x0 + big.width, y0 + big.height)


def difference(colour, rgba):
    """How far a redrawn sprite strays from the original: the redrawn picture reduced to the original size by
    averaging, compared with the original over its visible pixels (mean distance per colour channel)."""
    h, w = rgba.shape[:2]
    small = np.asarray(Image.fromarray(colour, "RGB").resize((w, h), Image.BOX), np.float32)
    visible = rgba[..., 3] > 0
    return float(np.abs(small - rgba[..., :3].astype(np.float32))[visible].mean()) if visible.any() else 0.0


def take_from_twins(sprites, names, target_of, sources):
    """Gives each of `names` that has no picture yet the picture of a sprite with exactly the same pixels from
    `sources`, instead of redrawing it. Such twins differ only in a colour of their palette bank that they do
    not use, for example sprites saved by earlier versions while a lamp of that bank blinked. Returns the
    names that got a picture."""
    by_pixels = {}
    for n in sorted(sources):
        if os.path.exists(target_of(n)):
            by_pixels.setdefault((sprites[n].shape, sprites[n].tobytes()), n)
    copied = []
    for name in names:
        twin = by_pixels.get((sprites[name].shape, sprites[name].tobytes()))
        if twin is None or twin == name or os.path.exists(target_of(name)):
            continue
        temporary = target_of(name) + ".part"
        shutil.copyfile(target_of(twin), temporary)
        upscale_sprites.replace_file(temporary, target_of(name))
        copied.append(name)
    return copied


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


MAX_GROUP = 9  # larger groups, a whole row of buildings for example, would leave too few pixels per tile


def stands_on_whole_tile(rgba):
    """Whether a tall sprite covers the lower half of the diamond of the tile it stands on, as a building piece
    does. A unit or a tree standing on a tile leaves most of that ground free."""
    h, w = rgba.shape[:2]
    y = np.arange(h)[:, None] + 0.5
    x = np.arange(w)[None, :] + 0.5
    middle = h - w / 4  # the middle row of the tile's diamond, which ends at the sprite's bottom row
    lower = (np.abs(x - w / 2) / (w / 2) + np.abs(y - middle) / (w / 4) <= 1.0) & (y >= middle)
    return lower.any() and (rgba[..., 3] > 0)[lower].mean() >= 0.85


def kind_of(rgba):
    """'flat' for a floor tile with painted markings, 'building' for a tall man-made sprite that fills its tile,
    None for anything else (whole screens, ground, natural mountains, icons, silhouettes, and units)."""
    if is_silhouette(rgba):
        return None
    h, w = rgba.shape[:2]
    if w >= 160 or (w <= 20 and h <= 20):
        return None
    if h > 16:
        return "building" if stands_on_whole_tile(rgba) and not is_natural_formation(rgba) and not is_compass_marker(rgba) else None
    if is_low_structure(rgba) or is_chasm_edge(rgba) or is_pink_flora_ground(rgba) or is_toxic_lake(rgba):
        return None
    return "flat" if not terrain.is_ground(rgba) and is_tile(rgba) else None


def tile_groups(sprites, pairs, skipped=None):
    """Sprites that the game draws side by side on its tile grid and that belong together: flat marked tiles
    such as the four tiles of a landing pad, and the pieces of a building that stands on several tiles, such as
    the command center. Each group comes with where each sprite's bottom row starts. Redrawn alone, every piece
    would get edges of its own, so a pad or a building would look like several small ones. `sprites` maps names
    to pictures. Returns one dictionary from name to position per group of two to MAX_GROUP sprites. Sprites
    that lie side by side but form no such group are added to the list `skipped`, one dictionary per set."""
    kinds = {n: kind_of(rgba) for n, rgba in sprites.items()}
    links = {}
    for a, b, dx, dy in pairs:
        if a not in kinds or b not in kinds or not kinds[a] or kinds[a] != kinds[b]:
            continue
        # Tiles of one grid have the same width; the pieces of one building can differ in height.
        if sprites[a].shape[1] != sprites[b].shape[1] or (kinds[a] == "flat" and sprites[a].shape != sprites[b].shape):
            continue
        w = sprites[a].shape[1]
        # Only neighbours on the isometric grid: half a tile across and a quarter of its width up or down.
        if abs(dx) == w // 2 and abs(dy) == w // 4:
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
        # A building standing on several tiles covers a full square of them. Pieces that do not fill one are
        # separate buildings standing side by side and are redrawn alone.
        if len(group) > MAX_GROUP or (kinds[start] == "building" and not fills_square(group, sprites[start].shape[1])):
            if skipped is not None:
                skipped.append(group)
            continue
        groups.append(group)
    return groups


def fills_square(group, width):
    """Whether the positions of a group fill a square of n by n tiles of the isometric grid, for n = 2 or 3."""
    cells = set()
    for x, y in group.values():
        # One step along one axis of the grid is (half a tile, a quarter tile), along the other (-half, quarter).
        a, b = x / (width / 2), y / (width / 4)
        cells.add((round((a + b) / 2), round((b - a) / 2)))
    if len(cells) != len(group):
        return False
    us, vs = {u for u, _ in cells}, {v for _, v in cells}
    n = len(us)
    return n in (2, 3) and len(vs) == n and len(cells) == n * n


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
    resized = crop.resize((w * SCALE, h * SCALE), Image.LANCZOS).convert("RGB")
    sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=95, threshold=2))
    return np.asarray(sharpened)


def bleed_edges(colour, alpha, steps=40):
    """Extends opaque interior RGB outward across transparent pixels (covering >3 sigma of SPRITE_COLOUR_RADIUS)
    so the plain white canvas background never darkens sprite perimeters in keep_colours or bleeds into GPU alpha edges."""
    core = alpha >= 200
    if not core.any() or core.all():
        return colour
    rgb = colour.astype(np.float32).copy()
    known = core.copy()
    h, w = known.shape
    for _ in range(steps):
        total = np.zeros_like(rgb)
        count = np.zeros((h, w), np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ys = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            k = known[ys, xs]
            total[yd, xd] += rgb[ys, xs] * k[..., None]
            count[yd, xd] += k
        grow = ~known & (count > 0)
        if not grow.any():
            break
        rgb[grow] = total[grow] / count[grow][:, None]
        known |= grow
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def finish(colour, rgba):
    """A redrawn sprite with fringe-free anti-aliased edges, gentle true-color tint anchoring, and smooth alpha."""
    alpha = terrain.smooth_alpha(rgba[..., 3], SCALE)
    colour = bleed_edges(colour, alpha)
    # Outside the sprite the original has no colour, so the nearest visible colour is used there.
    filled = upscale_sprites.fill_transparent(rgba[..., :3], rgba[..., 3] > 0)
    colour = keep_colours(colour, np.clip(filled + 0.5, 0, 255).astype(np.uint8), SPRITE_COLOUR_RADIUS,
                          luma_strength=SPRITE_LUMA_STRENGTH, chroma_strength=SPRITE_CHROMA_STRENGTH)
    # For chasm cliff-edge tiles, keep original pitch-black void pixels (and their edge-extended exterior) strictly
    # solid black so they join seamlessly with adjacent pure-black abyss tiles without diagonal perimeter seams.
    if is_chasm_edge(rgba):
        void = filled.max(axis=2) < 10
        void_hi = np.asarray(Image.fromarray((void.astype(np.uint8) * 255), "L").resize(
            (rgba.shape[1] * SCALE, rgba.shape[0] * SCALE), Image.BILINEAR), np.float32) / 255.0
        void_hi = np.clip((void_hi - 0.35) / 0.30, 0.0, 1.0)
        colour = np.clip(colour.astype(np.float32) * (1.0 - void_hi[..., None]) + 0.5, 0, 255).astype(np.uint8)
    return np.dstack([colour, alpha])


def stabilize_animation_frame(base_orig, base_up, frame_orig, frame_up, scale=SCALE, dilate_orig=1, feather_px=4.0):
    """Locks static pixels of an animated building or structure frame to `base_up` while keeping the moving
    animation region from `frame_up` with a softly feathered seam, preventing per-frame visual morphing.
    Handles bottom-aligned frames whose top heights differ by a few pixels (such as rotating vanes or masts)."""
    hb, w = base_orig.shape[:2]
    hf = frame_orig.shape[0]
    if (
        frame_orig.shape[1] != w
        or base_up.shape[:2] != (hb * scale, w * scale)
        or frame_up.shape[:2] != (hf * scale, w * scale)
    ):
        return frame_up
    hmax = max(hb, hf)
    # Bottom-align both originals and upscales so the building footprint aligns identically.
    def pad_top(arr, target_h):
        if arr.shape[0] == target_h:
            return arr
        out = np.zeros((target_h, arr.shape[1], arr.shape[2]), arr.dtype)
        out[target_h - arr.shape[0]:] = arr
        return out
    bo = pad_top(base_orig, hmax)
    fo = pad_top(frame_orig, hmax)
    bu = pad_top(base_up, hmax * scale)
    if hf > hb:
        # Rows above base_up's top have no base pixels, so inherit frame_up's RGB instead of black padding.
        bu[:(hf - hb) * scale] = frame_up[:(hf - hb) * scale]
    diff_orig = np.abs(bo.astype(np.int16) - fo.astype(np.int16)).max(axis=2) > 4
    if not diff_orig.any():
        return bu[hmax * scale - hf * scale:].copy()
    mask_orig = diff_orig.copy()
    for _ in range(max(0, int(dilate_orig))):
        grown = mask_orig.copy()
        grown[1:, :] |= mask_orig[:-1, :]
        grown[:-1, :] |= mask_orig[1:, :]
        grown[:, 1:] |= mask_orig[:, :-1]
        grown[:, :-1] |= mask_orig[:, 1:]
        mask_orig = grown
    # Crop mask and base back to frame's own height (bottom-aligned).
    mask_f = mask_orig[hmax - hf:]
    bu_f = bu[hmax * scale - hf * scale:]
    h_hi, w_hi = frame_up.shape[:2]
    mask_img = Image.fromarray((mask_f.astype(np.uint8) * 255), "L").resize((w_hi, h_hi), Image.NEAREST)
    if feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(feather_px))
    blend = np.asarray(mask_img, np.float32)[..., None] / 255.0
    rgb = bu_f[..., :3].astype(np.float32) * (1.0 - blend) + frame_up[..., :3].astype(np.float32) * blend
    return np.dstack([np.clip(rgb + 0.5, 0, 255).astype(np.uint8), frame_up[..., 3]])


def is_toxic_lake_center(rgba):
    """Whether a toxic lake tile is pure open acid liquid (tiling seamlessly across a lake interior)."""
    if not is_toxic_lake(rgba):
        return False
    rgb = rgba[..., :3][rgba[..., 3] > 127].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    acid_green = (g > 55) & (g > r + 30) & (g > b + 30) & (r < 35) & (b < 35)
    return bool(acid_green.mean() >= 0.85)


def is_toxic_lake_shore(rgba):
    """Whether a toxic lake tile is a shoreline transition between acid liquid and surrounding terrain."""
    return is_toxic_lake(rgba) and not is_toxic_lake_center(rgba)


def composite_toxic_lake_shore(orig_rgba, water_tile_hd, ground_tile_hd, scale=SCALE):
    """Composites a toxic-lake shoreline tile seamlessly from the HD acid-water tile and HD biome ground tile.

    Extends the original DOS water/land mask across transparent diamond borders before upscaling so every
    water-to-water edge matches `water_tile_hd` and every land-to-land edge matches `ground_tile_hd` with
    zero artificial tile walls, while adding a natural wet-bank rim and subtle emerald caustic shimmer along
    the exact shoreline contour.
    """
    h, w = orig_rgba.shape[:2]
    H, W = h * scale, w * scale
    if water_tile_hd.shape[:2] != (H, W) or ground_tile_hd.shape[:2] != (H, W):
        return None
    vis = orig_rgba[..., 3] > 0
    filled = upscale_sprites.fill_transparent(orig_rgba[..., :3], vis)
    r, g, b = filled[..., 0], filled[..., 1], filled[..., 2]
    acid_low = ((g > 55) & (g > r + 30) & (g > b + 30) & (r < 35) & (b < 35)).astype(np.float32)
    dark_shore_low = ((~(acid_low > 0.5)) & (np.maximum(np.maximum(r, g), b) < 110)).astype(np.float32)

    acid_hi = np.asarray(
        Image.fromarray((acid_low * 255.0).astype(np.uint8), "L")
        .resize((W, H), Image.BICUBIC)
        .filter(ImageFilter.GaussianBlur(3.2)),
        np.float32,
    ) / 255.0
    dark_hi = np.asarray(
        Image.fromarray((dark_shore_low * 255.0).astype(np.uint8), "L")
        .resize((W, H), Image.BILINEAR)
        .filter(ImageFilter.GaussianBlur(2.5)),
        np.float32,
    ) / 255.0

    water_rgb = water_tile_hd[..., :3].astype(np.float32).copy()
    ground_rgb = ground_tile_hd[..., :3].astype(np.float32).copy()
    # Deterministic organic waviness derived from the seamlessly tiling water/ground luminance textures.
    tex_noise = ((water_rgb[..., 1] - water_rgb[..., 1].mean()) / 255.0) * 0.22 + (
        (ground_rgb[..., 0] - ground_rgb[..., 0].mean()) / 255.0
    ) * 0.18
    dist = acid_hi + tex_noise * (4.0 * acid_hi * (1.0 - acid_hi))

    water_w = np.clip((dist - 0.42) / 0.16, 0.0, 1.0)
    water_w = water_w * water_w * (3.0 - 2.0 * water_w)
    bank_band = np.exp(-((dist - 0.43) / 0.09) ** 2)
    caustic_band = np.exp(-((dist - 0.57) / 0.08) ** 2)

    wet_tint = np.array([0.42, 0.56, 0.52], np.float32)
    bank_strength = np.clip(bank_band * 0.68 + dark_hi * 0.35, 0.0, 0.78)
    ground_rgb = ground_rgb * (1.0 - bank_strength[..., None]) + (ground_rgb * wet_tint) * bank_strength[..., None]

    glow_col = np.array([115.0, 235.0, 85.0], np.float32)
    water_rgb = water_rgb * (1.0 - 0.22 * caustic_band[..., None]) + glow_col * (0.22 * caustic_band[..., None])

    blended = ground_rgb * (1.0 - water_w[..., None]) + water_rgb * water_w[..., None]
    alpha = terrain.smooth_alpha(orig_rgba[..., 3], scale)
    return np.dstack([np.clip(blended + 0.5, 0, 255).astype(np.uint8), alpha])


def _find_lake_references(orig_rgba, sprites, target_of):
    """Finds already-upscaled HD acid-lake center and matching-biome ground tiles for seamless shore compositing."""
    h, w = orig_rgba.shape[:2]
    vis = orig_rgba[..., 3] > 127
    rgb = orig_rgba[..., :3][vis].astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    non_acid = ~((g > 55) & (g > r + 30) & (g > b + 30) & (r < 35) & (b < 35))
    if non_acid.any():
        shore_dummy = np.zeros((1, int(non_acid.sum()), 4), np.uint8)
        shore_dummy[0, :, :3] = np.clip(rgb[non_acid], 0, 255).astype(np.uint8)
        shore_dummy[0, :, 3] = 255
        want_biome = _ground_biome(shore_dummy)
    else:
        want_biome = "ice"

    water_img = None
    ground_img = None
    want_size = (w * SCALE, h * SCALE)
    for name in sorted(sprites):
        cand = sprites[name]
        if cand.shape[:2] != (h, w) or not os.path.exists(target_of(name)):
            continue
        if water_img is None and is_toxic_lake_center(cand):
            with Image.open(target_of(name)) as im:
                if im.size == want_size:
                    water_img = np.asarray(im.convert("RGBA"))
        elif ground_img is None and not is_toxic_lake(cand) and not is_chasm_edge(cand) and not is_pink_flora_ground(cand):
            if terrain.is_ground(cand) and _ground_biome(cand) == want_biome:
                with Image.open(target_of(name)) as im:
                    if im.size == want_size:
                        ground_img = np.asarray(im.convert("RGBA"))
        if water_img is not None and ground_img is not None:
            break
    if water_img is None or ground_img is None:
        return None, None
    return water_img, ground_img


def remaster_sprite(pipe, rgba, seed, tries=1):
    """Redraws a sprite `tries` times with different seeds and keeps the result closest to the original, since
    the model now and then moves a marking or adds something that is not there."""
    ground = (
        rgba.shape[0] <= 16
        and not is_low_structure(rgba)
        and not is_chasm_edge(rgba)
        and not is_pink_flora_ground(rgba)
        and (is_toxic_lake_center(rgba) or (not is_toxic_lake(rgba) and terrain.is_ground(rgba)))
    )
    canvas, box = ground_canvas(rgba) if ground else sprite_canvas(rgba)
    best, best_difference = None, None
    for attempt in range(tries):
        colour = cut(redraw(pipe, canvas, prompt_for(rgba), seed + attempt), box, rgba)
        d = difference(colour, rgba)
        if best is None or d < best_difference:
            best, best_difference = colour, d
    return finish(best, rgba)


def group_prompt_for(sprites, group):
    first = next(iter(group))
    if kind_of(sprites[first]) == "building":
        return BUILDING_PROMPT
    if any(is_landing_pad(sprites[n]) for n in group):
        return LANDING_PAD_GROUP_PROMPT
    return GROUP_PROMPT


def remaster_group(pipe, sprites, group, seed, tries=1):
    """Redraws a group of tiles as one picture, `tries` times, keeps the version closest to the originals over
    all its tiles, and cuts each tile out of it. Returns a dictionary from name to redrawn tile."""
    canvas, boxes = group_canvas(sprites, group)
    prompt = group_prompt_for(sprites, group)
    best, best_difference = None, None
    for attempt in range(tries):
        out = redraw(pipe, canvas, prompt, seed + attempt)
        colours = {n: cut(out, boxes[n], sprites[n]) for n in group}
        d = sum(difference(colours[n], sprites[n]) for n in group)
        if best is None or d < best_difference:
            best, best_difference = colours, d
    return {n: finish(best[n], sprites[n]) for n in group}


def remaster_picture(pipe, rgba, seed, tries=2):
    h, w = rgba.shape[:2]
    target = PICTURE_SIZE if (w, h) == (320, 200) else (max(64, (w * 4) // 64 * 64), max(64, (h * 4) // 64 * 64))
    big = Image.fromarray(rgba[..., :3], "RGB").resize(target, Image.BICUBIC)
    best, best_difference = None, None
    for attempt in range(max(1, min(tries, 2))):
        out = redraw(pipe, big, PICTURE_PROMPT, seed + attempt)
        resized = out.resize((w * SCALE, h * SCALE), Image.LANCZOS).convert("RGB")
        colour = np.asarray(resized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=95, threshold=2)))
        d = difference(colour, rgba)
        if best is None or d < best_difference:
            best, best_difference = colour, d
    colour = keep_colours(best, np.ascontiguousarray(rgba[..., :3]), PICTURE_COLOUR_RADIUS,
                          luma_strength=PICTURE_LUMA_STRENGTH, chroma_strength=PICTURE_CHROMA_STRENGTH)
    return np.dstack([colour, np.full((h * SCALE, w * SCALE), 255, np.uint8)])


def redraw(pipe, image, prompt, seed):
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe(image=[image], prompt=prompt, num_inference_steps=8, true_cfg_scale=1.0, generator=generator).images[0]


def canonical_shape_and_palette(rgba):
    """Returns a palette-independent shape signature and 6-bit RGB table for sprites up to 256x256."""
    h, w = rgba.shape[:2]
    if w > 256 or h > 256:
        return None, []
    flat = rgba.reshape(-1, 4)
    seen = {}
    pal = []
    out = bytearray()
    for r, g, b, a in flat:
        if a == 0:
            out.append(0)
        else:
            key = (int(r), int(g), int(b))
            idx = seen.get(key)
            if idx is None:
                if len(seen) >= 64:
                    return None, []
                idx = len(seen) + 1
                seen[key] = idx
                pal.append((int(r) >> 2, int(g) >> 2, int(b) >> 2))
            out.append(idx)
    return (w, h, bytes(out)), pal


def is_dimmed_or_near_duplicate(ref_pal, cur_pal):
    """True when cur_pal is a 1..2 DAC rounding drift or uniform multi-channel ambient dim of ref_pal.

    Exact-pixel twins (max_diff == 0) return False so they continue flowing to take_from_twins(), and
    single-channel or near-black palettes return False to stay aligned with hintless runtime matching.
    """
    if len(ref_pal) != len(cur_pal) or not ref_pal:
        return False
    has_visible = any(max(r) >= 6 for r in ref_pal)
    max_diff = max(abs(c[ch] - r[ch]) for r, c in zip(ref_pal, cur_pal) for ch in range(3))
    if has_visible and 1 <= max_diff <= 2:
        return True
    if any(c[ch] > r[ch] + 2 for r, c in zip(ref_pal, cur_pal) for ch in range(3)):
        return False

    def try_fit(allow_emissive):
        sum_ref, sum_cur = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        dimmed, unchanged, decreased = 0, 0, 0
        for r, c in zip(ref_pal, cur_pal):
            rmax = max(r)
            if rmax < 6:
                continue
            if any(c[ch] < r[ch] for ch in range(3)):
                decreased += 1
            if allow_emissive and rmax >= 16 and all(abs(c[ch] - r[ch]) <= 2 for ch in range(3)):
                unchanged += 1
                continue
            dimmed += 1
            for ch in range(3):
                sum_ref[ch] += r[ch]
                sum_cur[ch] += c[ch]
        if dimmed == 0:
            return False
        if not allow_emissive and len(ref_pal) >= 4 and decreased < 2:
            return False
        if allow_emissive and (dimmed < 2 or dimmed <= unchanged):
            return False
        scales = [sum_cur[ch] / sum_ref[ch] if sum_ref[ch] >= 8.0 else -1.0 for ch in range(3)]
        active = [s for s in scales if s >= 0.0]
        if len(active) < 2:
            return False
        avg = sum(active) / len(active)
        scales = [s if s >= 0.0 else avg for s in scales]
        if min(scales) < 0.25 or max(scales) > 0.98 or max(scales) / min(scales) > 1.60:
            return False
        if sum(1 for s in active if s < 0.96) < 2:
            return False
        for r, c in zip(ref_pal, cur_pal):
            rmax = max(r)
            if rmax < 6:
                continue
            if allow_emissive and rmax >= 16 and all(abs(c[ch] - r[ch]) <= 2 for ch in range(3)):
                continue
            if any(abs(c[ch] - r[ch] * scales[ch]) > 3.5 for ch in range(3)):
                return False
        return True

    if try_fit(False):
        return True
    return len(ref_pal) >= 4 and try_fit(True)


def filter_ambient_duplicates(sprites, candidates, upscaled_names):
    """Drops candidates that are dimmed nighttime or +-2 DAC rounding copies of an upscaled or kept brighter sprite."""
    by_shape = {}
    for name, rgba in sprites.items():
        if is_silhouette(rgba):
            continue
        sig, pal = canonical_shape_and_palette(rgba)
        if sig is not None and pal:
            bright = sum(r + g + b for r, g, b in pal)
            by_shape.setdefault(sig, []).append((name, name in upscaled_names, bright, pal))
    skip = set()
    for items in by_shape.values():
        if len(items) < 2:
            continue
        kept_pals = [pal for _, is_up, _, pal in items if is_up]
        pending = sorted(((name, bright, pal) for name, is_up, bright, pal in items if not is_up and name in candidates),
                         key=lambda x: (-x[1], x[0]))
        for name, _, pal in pending:
            if any(is_dimmed_or_near_duplicate(ref_pal, pal) for ref_pal in kept_pals):
                skip.add(name)
            else:
                kept_pals.append(pal)
    return [n for n in candidates if n not in skip]


def priority_key_single(name, rgba):
    """Remaster tall buildings first, then natural mountain formations, then pure ground/lake centers before shorelines, then screens, and icons last."""
    h, w = rgba.shape[:2]
    if w <= 64 and h > 16:
        return (1 if is_natural_formation(rgba) else 0, -h, name)
    if w >= 32 and w <= 64 and h <= 16:
        return (2, 1 if is_toxic_lake_shore(rgba) else 0, name)
    if w >= 160:
        return (3, 0, name)
    return (4, 0, name)


def priority_key_group(sprites, group):
    """Remaster multi-tile buildings before landing pads and other flat tile groups."""
    first = next(iter(group))
    return (0 if kind_of(sprites[first]) == "building" else 1, -len(group), sorted(group)[0])


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
    if args.tries < 1:
        sys.exit("--tries must be at least 1.")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"  # read when the model libraries are first imported, in load_pipeline
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot reach the graphics card.")
    root = os.path.join(BASE, args.game, "hd")
    name_of = lambda path: os.path.splitext(os.path.basename(path))[0]  # noqa: E731
    chosen = {name_of(p) for prefix in (args.only or "").split(",")
              for p in glob.glob(os.path.join(glob.escape(root), "original", prefix + "*.png"))}
    # Every saved sprite is read, because a chosen tile is redrawn together with the tiles next to it.
    sprites = {}
    for p in glob.glob(os.path.join(glob.escape(root), "original", "*.png")):
        try:
            with Image.open(p) as im:
                sprites[name_of(p)] = np.asarray(im.convert("RGBA"))
        except (OSError, ValueError) as err:
            print(f"Skipping unreadable sprite {os.path.basename(p)}: {err}", flush=True)
    chosen &= set(sprites)
    out_dir = args.out or os.path.join(root, "upscaled")
    os.makedirs(out_dir, exist_ok=True)
    upscale_sprites.remove_stale_parts(glob.glob(os.path.join(glob.escape(out_dir), "*.part")))
    target_of = lambda name: os.path.join(out_dir, name + ".png")  # noqa: E731

    def wanted(name):
        return name in chosen and not is_silhouette(sprites[name]) and (args.again or not os.path.exists(target_of(name)))

    if not args.again:
        for name in take_from_twins(sprites, sorted(n for n in chosen if not is_silhouette(sprites[n])), target_of, sprites):
            print(f"{name}.png has the same pixels as a sprite that is redrawn already and takes its picture", flush=True)
    upscaled_names = {n for n in sprites if os.path.exists(target_of(n))}
    if not args.again:
        active = set(filter_ambient_duplicates(sprites, {n for n in chosen if wanted(n)}, upscaled_names))
    else:
        active = {n for n in chosen if wanted(n)}

    def wanted_active(name):
        return name in active

    skipped = []
    groups = tile_groups(sprites, read_layout(os.path.join(root, "layout.txt")), skipped)
    for group in skipped:
        if any(wanted_active(n) for n in group):
            print(f"These {len(group)} sprites lie side by side but are not one building or pad, so each is redrawn "
                  f"alone: {', '.join(sorted(group))}", flush=True)
    grouped = {n for g in groups for n in g}
    groups = sorted((g for g in groups if any(wanted_active(n) for n in g)), key=lambda g: priority_key_group(sprites, g))
    singles = [n for n in sorted(chosen) if wanted_active(n) and n not in grouped]
    # Of several new sprites with the same pixels, one is redrawn and the others take its picture afterwards.
    twins_later, seen_pixels = [], set()
    if not args.again:
        for n in singles:
            pixels = (sprites[n].shape, sprites[n].tobytes())
            (twins_later if pixels in seen_pixels else []).append(n)
            seen_pixels.add(pixels)
        singles = [n for n in singles if n not in twins_later]
    singles.sort(key=lambda n: priority_key_single(n, sprites[n]))
    total = len(singles) + sum(wanted_active(n) for g in groups for n in g)
    if not total:
        print("Nothing to redraw.")
        return
    pipe = None
    done = 0

    def ensure_pipe():
        nonlocal pipe
        if pipe is None:
            pipe = load_pipeline()
        return pipe

    def save(name, result):
        nonlocal done
        temporary = target_of(name) + ".part"
        Image.fromarray(result, "RGBA").save(temporary, format="PNG")
        upscale_sprites.replace_file(temporary, target_of(name))
        done += 1
        print(f"{done} of {total}: {name}.png", flush=True)

    # Multi-tile buildings and landing pads are painted first so the player's colony upgrades immediately.
    for group in groups:
        # Tiles of the group that already have a picture keep it unless --again is given, as single sprites do.
        todo = [n for n in sorted(group) if wanted_active(n)]
        results = remaster_group(ensure_pipe(), sprites, group, args.seed, args.tries)
        for name in todo:
            save(name, results[name])
    for name in singles:
        rgba = sprites[name]
        whole = rgba.shape[1] >= 160 and bool((rgba[..., 3] == 255).all())
        if not whole and is_compass_marker(rgba):
            save(name, render_compass_marker(rgba))
            continue
        if not whole and is_toxic_lake_shore(rgba):
            water_hd, ground_hd = _find_lake_references(rgba, sprites, target_of)
            if water_hd is not None and ground_hd is not None:
                composited = composite_toxic_lake_shore(rgba, water_hd, ground_hd)
                if composited is not None:
                    save(name, composited)
                    continue
        save(name, remaster_picture(ensure_pipe(), rgba, args.seed, args.tries) if whole else remaster_sprite(ensure_pipe(), rgba, args.seed, args.tries))
    take_from_twins(sprites, twins_later, target_of, singles)


if __name__ == "__main__":
    main()
