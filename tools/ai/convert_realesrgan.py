"""Converts a Real-ESRGAN SRVGGNetCompact checkpoint (.pth) into the weight file the runtime's AI filter reads.

Usage: python -I convert_realesrgan.py <model.pth> <output.bin>

The checkpoint is read without PyTorch. A restricted unpickler accepts only the classes a weights file needs
(an ordered dictionary, float tensors and their storages), so a crafted file cannot run code.

Output layout, all little-endian: the 8 bytes "NAXSRV1\\0", then the number of convolutions, the feature count
and the upscale factor as 32-bit integers. Then, for each convolution in order: its input and output channel
counts as 32-bit integers, the weights as floats in [out][in][3][3] order, the biases, and, for every
convolution except the last, the PReLU slopes of its output channels.
"""
import array
import collections
import pickle
import struct
import sys
import zipfile


class FloatStorage:
    pass


class HalfStorage:
    pass


def rebuild_tensor(storage, offset, size, stride, *_rest):
    data, dtype = storage
    count = 1
    for s in size:
        count *= s
    expected = []
    acc = 1
    for s in reversed(size):
        expected.insert(0, acc)
        acc *= s
    if list(stride) != expected:
        raise ValueError("only contiguous tensors are supported")
    return {"shape": tuple(size), "values": data[offset:offset + count]}


class WeightsUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
        ("torch", "FloatStorage"): FloatStorage,
        ("torch", "HalfStorage"): HalfStorage,
    }

    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing {module}.{name} in a weights file") from None

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError("unexpected persistent id")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        if storage_type is FloatStorage:
            values = array.array("f")
            values.frombytes(raw)
        elif storage_type is HalfStorage:
            values = array.array("f", (struct.unpack_from("<e", raw, i)[0] for i in range(0, len(raw), 2)))
        else:
            raise pickle.UnpicklingError("unsupported storage type")
        if sys.byteorder != "little":
            values.byteswap()
        return values, storage_type


def load_state(path):
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        with archive.open(pkl) as f:
            state = WeightsUnpickler(f, archive, prefix).load()
    for key in ("params_ema", "params"):
        if isinstance(state, dict) and key in state:
            return state[key]
    return state


def convert(src, dst):
    state = load_state(src)
    convs = []
    index = 0
    while f"body.{index}.weight" in state:
        weight = state[f"body.{index}.weight"]
        if len(weight["shape"]) != 4 or weight["shape"][2:] != (3, 3):
            raise ValueError(f"body.{index} is not a 3x3 convolution")
        bias = state[f"body.{index}.bias"]
        slopes = state.get(f"body.{index + 1}.weight")
        if slopes is not None and len(slopes["shape"]) != 1:
            slopes = None
        convs.append((weight, bias, slopes))
        index += 2 if slopes is not None else 1
    if len(convs) < 2:
        raise ValueError("no SRVGGNetCompact body found")
    features = convs[0][0]["shape"][0]
    last_out = convs[-1][0]["shape"][0]
    upscale = int(round((last_out / 3) ** 0.5))
    if upscale * upscale * 3 != last_out or convs[0][0]["shape"][1] != 3:
        raise ValueError("unexpected input or output channels")
    for i, (_w, _b, slopes) in enumerate(convs):
        if (slopes is None) != (i == len(convs) - 1):
            raise ValueError("every convolution but the last must be followed by PReLU")
    with open(dst, "wb") as out:
        out.write(b"NAXSRV1\0")
        out.write(struct.pack("<3I", len(convs), features, upscale))
        for weight, bias, slopes in convs:
            out.write(struct.pack("<2I", weight["shape"][1], weight["shape"][0]))
            for part in (weight, bias) + ((slopes,) if slopes is not None else ()):
                values = array.array("f", part["values"])
                if sys.byteorder != "little":
                    values.byteswap()
                out.write(values.tobytes())
    return len(convs), features, upscale


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.splitlines()[2])
    convs, features, upscale = convert(sys.argv[1], sys.argv[2])
    print(f"{convs} convolutions with {features} features, upscale x{upscale}")


if __name__ == "__main__":
    main()
