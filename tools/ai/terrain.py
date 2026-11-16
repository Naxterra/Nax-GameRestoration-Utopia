"""High-resolution ground tiles from a seamless texture, so that the terrain gets real fine detail and the tiles
continue into each other without seams.

An isometric ground tile shows a square piece of ground as a diamond, 32 pixels wide and 16 high, and its
neighbours sit 16 pixels across and 8 down (or up). Each pixel of a tile is mapped back onto the ground square,
and the texture is read there. Because the texture repeats seamlessly and neighbouring tiles are one whole
square apart on the ground, the detail of one tile continues exactly in the next.

The game's own tile still decides the look: its colours, blurred by about half a pixel, are the base, the
texture only adds fine light and dark variation and a little colour variation on top, and the tile's dark rocks
and cracks are kept as sharp dark shapes.
"""
import numpy as np

SCALE = 8


def gaussian_blur(image, sigma):
    """Separable Gaussian blur of an H x W or H x W x C float array, repeating the edge pixels."""
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = image.astype(np.float32)
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in range(out.ndim)], mode="edge")
        acc = np.zeros_like(out)
        for i, k in enumerate(kernel):
            acc += k * np.take(padded, np.arange(i, i + out.shape[axis]), axis=axis)
        out = acc
    return out


def enlarge(image, factor):
    """Bilinear enlargement with sample positions between the source pixels."""
    h, w = image.shape[:2]
    ys = np.clip((np.arange(h * factor) + 0.5) / factor - 0.5, 0, h - 1)
    xs = np.clip((np.arange(w * factor) + 0.5) / factor - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    if image.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    img = image.astype(np.float32)
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bottom = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


class GroundTexture:
    def __init__(self, rgb):
        """rgb: a seamless square texture, H x W x 3 with values 0..255."""
        texture = rgb.astype(np.float32) / 255
        self.size = texture.shape[0]
        lum = texture.mean(axis=2)
        detail = lum - gaussian_blur(lum, self.size / 40)
        self.detail = detail / (detail.std() + 1e-6)
        self.colour = texture / (texture.mean(axis=(0, 1)) + 1e-6)

    def sample(self, layer, u, v):
        n = self.size
        x, y = (u % 1.0) * n, (v % 1.0) * n
        x0, y0 = np.floor(x).astype(int) % n, np.floor(y).astype(int) % n
        x1, y1 = (x0 + 1) % n, (y0 + 1) % n
        fx, fy = x - np.floor(x), y - np.floor(y)
        if layer.ndim == 3:
            fx, fy = fx[..., None], fy[..., None]
        return (layer[y0, x0] * (1 - fx) * (1 - fy) + layer[y0, x1] * fx * (1 - fy) +
                layer[y1, x0] * (1 - fx) * fy + layer[y1, x1] * fx * fy)


def shift(a, dy, dx):
    """The array moved by dy rows and dx columns, with zeros where nothing moved in."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def is_ground(rgba):
    """A flat ground tile: 32 pixels wide, at most 16 high, filling about the diamond, without bright markings."""
    h, w = rgba.shape[:2]
    if w != 32 or h > 16 or h < 12:
        return False
    visible = rgba[:, :, 3] > 127
    if visible.sum() < 0.45 * w * h:
        return False
    lum = rgba[:, :, :3].astype(np.float32).mean(axis=2) / 255
    return bool(lum[visible].max() <= 0.5)


def ground_tile(rgba, texture, amount=0.8, rocks=0.8):
    """Returns the tile eight times as large (RGBA, 0..255) with the texture's detail on the tile's colours."""
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3].astype(np.float32) / 255
    visible = rgba[:, :, 3] > 127
    # Transparent corners take the colours of their visible neighbours, so blurring does not darken the edges.
    filled = rgb.copy()
    known = visible.copy()
    while not known.all() and known.any():
        total = np.zeros_like(filled)
        count = np.zeros(known.shape, np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = shift(known, dy, dx)
            total += shift(filled, dy, dx) * shifted[..., None]
            count += shifted
        grown = ~known & (count > 0)
        filled[grown] = total[grown] / count[grown][:, None]
        known |= grown
    base = gaussian_blur(enlarge(filled, SCALE), SCALE * 0.6)
    dark = ((rgb.mean(axis=2) < 0.12) & visible).astype(np.float32)
    rock = np.clip((enlarge(dark, SCALE) - 0.3) / 0.35, 0, 1)
    rock = rock * rock * (3 - 2 * rock)
    ys, xs = np.mgrid[0:h * SCALE, 0:w * SCALE].astype(np.float32)
    x, y = (xs + 0.5) / SCALE, (ys + 0.5) / SCALE
    u, v = x / 32 + y / 16, -x / 32 + y / 16
    detail = texture.sample(texture.detail, u, v)
    colour = texture.sample(texture.colour, u, v)
    out = base * (1 + amount * 0.25 * detail)[..., None] * (1 + amount * 0.5 * (colour - 1))
    out = out * (1 - rocks * rock * (0.85 + 0.15 * np.clip(detail, -1, 1)))[..., None]
    alpha = np.repeat(np.repeat(rgba[:, :, 3], SCALE, axis=0), SCALE, axis=1)
    return np.concatenate([(np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), alpha[:, :, None]], axis=2)
