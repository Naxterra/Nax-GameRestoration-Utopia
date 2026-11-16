"""Converts a Real-ESRGAN SRVGGNetCompact checkpoint (.pth) into the weight file the runtime's AI filter reads.

Usage: python -I convert_realesrgan.py <model.pth> <output.bin>

The checkpoint is read without PyTorch, in the zip format of current PyTorch versions or the older format. A
restricted unpickler accepts only the classes a weights file needs (an ordered dictionary, float tensors and
their storages), so a crafted file cannot run code.

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


class Storage:
    """The numbers behind one or more tensors; in the old file format they are read after the tensors."""

    def __init__(self, storage_type):
        if storage_type not in (FloatStorage, HalfStorage):
            raise pickle.UnpicklingError("unsupported storage type")
        self.storage_type = storage_type
        self.values = None

    def fill(self, raw):
        if self.storage_type is FloatStorage:
            values = array.array("f")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
        else:
            values = array.array("f", (struct.unpack_from("<e", raw, i)[0] for i in range(0, len(raw), 2)))
        self.values = values


class Tensor:
    def __init__(self, storage, offset, shape):
        self.storage, self.offset, self.shape = storage, offset, tuple(shape)

    def __getitem__(self, name):
        if name == "shape":
            return self.shape
        if name == "values":
            count = 1
            for s in self.shape:
                count *= s
            return self.storage.values[self.offset:self.offset + count]
        raise KeyError(name)


def rebuild_tensor(storage, offset, size, stride, *_rest):
    expected = []
    acc = 1
    for s in reversed(size):
        expected.insert(0, acc)
        acc *= s
    if list(stride) != expected:
        raise ValueError("only contiguous tensors are supported")
    return Tensor(storage, offset, size)


class WeightsUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
        ("torch._utils", "_rebuild_tensor"): rebuild_tensor,
        ("torch", "FloatStorage"): FloatStorage,
        ("torch", "HalfStorage"): HalfStorage,
    }

    def __init__(self, file, archive=None, prefix=None):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix
        self.storages = {}

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing {module}.{name} in a weights file") from None

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError("unexpected persistent id")
        _kind, storage_type, key, _location, _numel = pid[:5]
        if len(pid) > 5 and pid[5] is not None:
            raise pickle.UnpicklingError("storage views are not supported")
        if key not in self.storages:
            storage = Storage(storage_type)
            if self.archive is not None:
                storage.fill(self.archive.read(f"{self.prefix}/data/{key}"))
            self.storages[key] = storage
        return self.storages[key]


class PlainUnpickler(pickle.Unpickler):
    """For the header records of the old format, which hold only numbers, strings, lists and dictionaries."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name} in a weights file")


LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


def load_legacy(f):
    """Reads a checkpoint in the format torch.save used before PyTorch 1.6: four pickles, then the storages."""
    if PlainUnpickler(f).load() != LEGACY_MAGIC:
        raise ValueError("not a PyTorch checkpoint")
    PlainUnpickler(f).load()  # protocol version
    PlainUnpickler(f).load()  # system information
    unpickler = WeightsUnpickler(f)
    state = unpickler.load()
    for key in PlainUnpickler(f).load():
        storage = unpickler.storages[key]
        count = struct.unpack("<q", f.read(8))[0]
        size = 4 if storage.storage_type is FloatStorage else 2
        raw = f.read(count * size)
        if len(raw) != count * size:
            raise ValueError("the checkpoint ends early")
        storage.fill(raw)
    return state


def load_state(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
            prefix = pkl.rsplit("/", 1)[0]
            with archive.open(pkl) as f:
                state = WeightsUnpickler(f, archive, prefix).load()
    else:
        with open(path, "rb") as f:
            state = load_legacy(f)
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
