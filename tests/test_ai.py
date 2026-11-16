"""Tests for the AI filter tools: the checkpoint converter and the CPU reference of the network."""
import io
import os
import pickle
import struct
import sys
import tempfile
import types
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "ai"))

import convert_realesrgan  # noqa: E402
import reference_upscale  # noqa: E402


class FakeStorage:
    """Stands in for torch.FloatStorage while writing a checkpoint in PyTorch's file layout."""


def rebuild(*args):
    raise AssertionError("only used as a pickled reference")


def checkpoint(state, path, extra=None):
    """Writes a checkpoint the way torch.save does: a zip with data.pkl and one raw file per tensor."""
    torch_utils = types.ModuleType("torch._utils")
    torch_utils._rebuild_tensor_v2 = rebuild
    rebuild.__module__, rebuild.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    torch_mod = types.ModuleType("torch")
    torch_mod.FloatStorage = FakeStorage
    FakeStorage.__module__, FakeStorage.__qualname__ = "torch", "FloatStorage"
    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = torch_mod, torch_utils
    storages = {}

    class Tensor:
        def __init__(self, key, shape):
            self.key, self.shape = key, shape

        def __reduce__(self):
            stride, acc = [], 1
            for s in reversed(self.shape):
                stride.insert(0, acc)
                acc *= s
            return rebuild, (("storage", self.key), 0, tuple(self.shape), tuple(stride), False, None)

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "storage":
                return ("storage", FakeStorage, obj[1], "cpu", len(storages[obj[1]]))
            return None

    try:
        import collections
        tensors = collections.OrderedDict()
        for i, (name, values) in enumerate(state.items()):
            key = str(1000 + i)
            storages[key] = np.asarray(values, np.float32).ravel()
            tensors[name] = Tensor(key, np.asarray(values).shape)
        buf = io.BytesIO()
        Writer(buf, protocol=2).dump({"params": tensors} if extra is None else extra)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for key, values in storages.items():
            z.writestr(f"archive/data/{key}", values.astype("<f4").tobytes())


def tiny_state(rng):
    """A two-convolution SRVGGNetCompact with 16 features and an upscale factor of 4."""
    return {
        "body.0.weight": rng.standard_normal((16, 3, 3, 3)),
        "body.0.bias": rng.standard_normal(16),
        "body.1.weight": rng.standard_normal(16),
        "body.2.weight": rng.standard_normal((48, 16, 3, 3)),
        "body.2.bias": rng.standard_normal(48),
    }


class ConverterTests(unittest.TestCase):
    def test_checkpoint_converts_to_weights_the_reference_reads_back_unchanged(self):
        state = tiny_state(np.random.default_rng(1))
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "m.pth"), os.path.join(tmp, "m.bin")
            checkpoint(state, src)
            self.assertEqual(convert_realesrgan.convert(src, dst), (2, 16, 4))
            layers, upscale = reference_upscale.load_weights(dst)
        self.assertEqual(upscale, 4)
        np.testing.assert_allclose(layers[0][0], state["body.0.weight"], rtol=1e-6)
        np.testing.assert_allclose(layers[0][2], state["body.1.weight"], rtol=1e-6)
        np.testing.assert_allclose(layers[1][1], state["body.2.bias"], rtol=1e-6)
        self.assertIsNone(layers[1][2])

    def test_checkpoint_that_calls_other_code_is_refused(self):
        class Evil:
            def __reduce__(self):
                return os.system, ("echo should never run",)

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "evil.pth")
            with zipfile.ZipFile(src, "w") as z:
                z.writestr("archive/data.pkl", pickle.dumps({"params": Evil()}, protocol=2))
            with self.assertRaises(pickle.UnpicklingError):
                convert_realesrgan.convert(src, os.path.join(tmp, "out.bin"))
            self.assertFalse(os.path.exists(os.path.join(tmp, "out.bin")))


class ReferenceTests(unittest.TestCase):
    @staticmethod
    def network(last_bias, first_zero=True):
        w0 = np.zeros((16, 3, 3, 3), np.float32)
        layers = [(w0, np.zeros(16, np.float32), np.zeros(16, np.float32)),
                  (np.zeros((48, 16, 3, 3), np.float32), last_bias.astype(np.float32), None)]
        return layers

    def test_zero_network_returns_the_picture_enlarged_four_times(self):
        image = np.random.default_rng(2).integers(0, 256, (5, 7, 3), dtype=np.uint8)
        out = reference_upscale.upscale_image(self.network(np.zeros(48)), 4, image)
        self.assertEqual(out.shape, (20, 28, 3))
        np.testing.assert_array_equal(out, np.repeat(np.repeat(image, 4, axis=0), 4, axis=1))

    def test_output_channels_are_arranged_like_pixel_shuffle(self):
        # Channel c*16 + i*4 + j must land in row i and column j of each 4x4 block, in colour c.
        bias = np.array([(c * 16 + i * 4 + j) / 255.0 for c in range(3) for i in range(4) for j in range(4)])
        out = reference_upscale.upscale_image(self.network(bias), 4, np.zeros((1, 1, 3), np.uint8))
        for c in range(3):
            for i in range(4):
                for j in range(4):
                    self.assertEqual(out[i, j, c], c * 16 + i * 4 + j)

    def test_weight_file_with_wrong_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.bin")
            with open(path, "wb") as f:
                f.write(b"NOTAFILE" + struct.pack("<3I", 2, 16, 4))
            with self.assertRaises(ValueError):
                reference_upscale.load_weights(path)


if __name__ == "__main__":
    unittest.main()
