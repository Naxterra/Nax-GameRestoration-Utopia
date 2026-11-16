"""The ESRGAN network (RRDBNet) on the CPU, for upscaling sprites with large models such as 4x-PixelPerfectV4
or RealESRGAN_x4plus.

The network has 23 blocks of three densely connected units with five convolutions each, then two steps that
double the size, so it upscales four times. Checkpoints in the original ESRGAN layout ("model.0", "model.1.sub")
and in the Real-ESRGAN layout ("conv_first", "body") are both accepted. They are read with the restricted
loader of convert_realesrgan.py, so no PyTorch is needed and a crafted file cannot run code.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_realesrgan  # noqa: E402

OLD_NAMES = [
    (re.compile(r"^model\.0\.(weight|bias)$"), r"conv_first.\1"),
    (re.compile(r"^model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(weight|bias)$"), lambda m: f"body.{m[1]}.rdb{m[2]}.conv{m[3]}.{m[4]}"),
    (re.compile(r"^model\.3\.(weight|bias)$"), r"conv_up1.\1"),
    (re.compile(r"^model\.6\.(weight|bias)$"), r"conv_up2.\1"),
    (re.compile(r"^model\.8\.(weight|bias)$"), r"conv_hr.\1"),
    (re.compile(r"^model\.10\.(weight|bias)$"), r"conv_last.\1"),
]


def is_rrdb(state):
    return "conv_first.weight" in state or "model.1.sub.0.RDB1.conv1.0.weight" in state


def load(path):
    """Returns the network's weights as float32 arrays under the Real-ESRGAN names."""
    state = convert_realesrgan.load_state(path)
    if not is_rrdb(state):
        raise ValueError(f"{path} is not an ESRGAN (RRDBNet) checkpoint")
    blocks = 1 + max(int(m[1]) for k in state if (m := re.match(r"^model\.1\.sub\.(\d+)\.", k)) or (m := re.match(r"^body\.(\d+)\.", k)))
    weights = {}
    for key in state:
        name = key
        if key.startswith("model."):
            if key.startswith(f"model.1.sub.{blocks - 1}.") and ".RDB" not in key:
                name = "conv_body." + key.rsplit(".", 1)[1]
            else:
                for pattern, replacement in OLD_NAMES:
                    if pattern.match(key):
                        name = pattern.sub(replacement, key)
                        break
                else:
                    raise ValueError(f"unexpected layer {key}")
        tensor = state[key]
        weights[name] = np.asarray(tensor["values"], np.float32).reshape(tensor["shape"])
    if "model.1.sub.0.RDB1.conv1.0.weight" in state:
        blocks -= 1  # in the old layout the last entry of model.1.sub is the convolution after the blocks
    if weights["conv_first.weight"].shape[1] != 3 or weights["conv_last.weight"].shape[0] != 3:
        raise ValueError("only networks with three colour channels in and out are supported")
    return weights, blocks


def conv3x3(x, w, b):
    channels, h, wd = x.shape
    padded = np.zeros((channels, h + 2, wd + 2), np.float32)
    padded[:, 1:-1, 1:-1] = x
    out = np.empty((w.shape[0], h * wd), np.float32)
    out[:] = b[:, None]
    for dy in range(3):
        for dx in range(3):
            out += w[:, :, dy, dx] @ padded[:, dy:dy + h, dx:dx + wd].reshape(channels, -1)
    return out.reshape(-1, h, wd)


def lrelu(x):
    return np.where(x >= 0, x, x * 0.2)


def upscale_image(net, image):
    """Upscales an RGB picture (height x width x 3, 0..255) four times."""
    weights, blocks = net
    conv = lambda name, x: conv3x3(x, weights[name + ".weight"], weights[name + ".bias"])  # noqa: E731
    x = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    fea = conv("conv_first", x)
    trunk = fea
    for i in range(blocks):
        block_in = trunk
        for r in (1, 2, 3):
            parts = [trunk]
            for c in range(1, 5):
                parts.append(lrelu(conv(f"body.{i}.rdb{r}.conv{c}", np.concatenate(parts))))
            trunk = conv(f"body.{i}.rdb{r}.conv5", np.concatenate(parts)) * 0.2 + trunk
        trunk = trunk * 0.2 + block_in
    fea = fea + conv("conv_body", trunk)
    for name in ("conv_up1", "conv_up2"):
        fea = lrelu(conv(name, fea.repeat(2, axis=1).repeat(2, axis=2)))
    out = conv("conv_last", lrelu(conv("conv_hr", fea)))
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)
