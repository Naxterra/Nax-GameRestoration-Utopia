"""Runs the AI upscaler on the CPU, as a reference for checking the runtime's GPU version.

Usage: python -I reference_upscale.py <weights.bin> <input.png> <output.png> [<gpu.png>]

The input is a 320x200 picture as written by the runtime's screenshots. With a fourth argument the result is
compared against the GPU output of the same picture, and the peak signal-to-noise ratio is printed.
"""
import struct
import sys
import zlib

import numpy as np


def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    pos, chunks, width, height, colour = 8, b"", 0, 0, 0
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, _depth, colour = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            chunks += body
        pos += 12 + length
    channels = {2: 3, 6: 4}[colour]
    raw = zlib.decompress(chunks)
    stride = width * channels
    rows = []
    for y in range(height):
        line = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        if line[0] != 0:
            raise ValueError("only unfiltered PNG rows are supported")
        rows.append(np.frombuffer(line[1:], dtype=np.uint8))
    image = np.stack(rows).reshape(height, width, channels)[:, :, :3]
    return image


def write_png(path, image):
    height, width, _ = image.shape
    raw = b"".join(b"\0" + image[y].astype(np.uint8).tobytes() for y in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def load_weights(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"NAXSRV1\0":
        raise ValueError("not an AI filter weight file")
    count, _features, upscale = struct.unpack_from("<3I", data, 8)
    pos, layers = 20, []
    for i in range(count):
        cin, cout = struct.unpack_from("<2I", data, pos)
        pos += 8
        n = cout * cin * 9
        w = np.frombuffer(data, np.float32, n, pos).reshape(cout, cin, 3, 3)
        pos += n * 4
        b = np.frombuffer(data, np.float32, cout, pos)
        pos += cout * 4
        a = None
        if i < count - 1:
            a = np.frombuffer(data, np.float32, cout, pos)
            pos += cout * 4
        layers.append((w, b, a))
    return layers, upscale


def conv3x3(x, w, b):
    cin, h, wd = x.shape
    padded = np.zeros((cin, h + 2, wd + 2), np.float32)
    padded[:, 1:-1, 1:-1] = x
    out = np.zeros((w.shape[0], h, wd), np.float32)
    for dy in range(3):
        for dx in range(3):
            patch = padded[:, dy:dy + h, dx:dx + wd].reshape(cin, -1)
            out += (w[:, :, dy, dx] @ patch).reshape(-1, h, wd)
    return out + b[:, None, None]


def upscale_image(layers, upscale, image):
    x = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    out = x
    for w, b, a in layers:
        out = conv3x3(out, w, b)
        if a is not None:
            out = np.where(out >= 0, out, out * a[:, None, None])
    c, h, wd = out.shape
    out = out.reshape(3, upscale, upscale, h, wd).transpose(0, 3, 1, 4, 2).reshape(3, h * upscale, wd * upscale)
    out += np.repeat(np.repeat(x, upscale, axis=1), upscale, axis=2)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__.splitlines()[2])
    layers, upscale = load_weights(sys.argv[1])
    result = upscale_image(layers, upscale, read_png(sys.argv[2]))
    write_png(sys.argv[3], result)
    if len(sys.argv) == 5:
        gpu = read_png(sys.argv[4]).astype(np.float64)
        mse = np.mean((gpu - result.astype(np.float64)) ** 2)
        largest = np.max(np.abs(gpu - result.astype(np.float64)))
        psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
        print(f"PSNR {psnr:.1f} dB, largest difference {largest:.0f} of 255")


if __name__ == "__main__":
    main()
