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
import rrdb  # noqa: E402
import terrain  # noqa: E402
import upscale_sprites  # noqa: E402


class FakeStorage:
    """Stands in for torch.FloatStorage while writing a checkpoint in PyTorch's file layout."""


def rebuild(*args):
    raise AssertionError("only used as a pickled reference")


def checkpoint(state, path, extra=None, legacy=False):
    """Writes a checkpoint the way torch.save does: a zip with data.pkl and one raw file per tensor.

    With legacy it uses the format of PyTorch before 1.6 instead, with the bare state dictionary as old ESRGAN
    checkpoints have it: four pickles, the list of storage keys, then each storage's length and numbers.
    """
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
        Writer(buf, protocol=2).dump(tensors if legacy else {"params": tensors} if extra is None else extra)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    if legacy:
        with open(path, "wb") as f:
            for record in (0x1950A86A20F9469CFC6C, 1001, {"little_endian": True}):
                f.write(pickle.dumps(record, protocol=2))
            f.write(buf.getvalue())
            f.write(pickle.dumps(list(storages), protocol=2))
            for values in storages.values():
                f.write(struct.pack("<q", len(values)) + values.astype("<f4").tobytes())
        return
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


class RrdbTests(unittest.TestCase):
    def test_old_format_esrgan_checkpoint_loads_and_upscales_four_times(self):
        rng = np.random.default_rng(3)
        nf, gc = 4, 2
        state = {"model.0.weight": rng.standard_normal((nf, 3, 3, 3)), "model.0.bias": rng.standard_normal(nf)}
        for r in (1, 2, 3):
            for c in range(1, 6):
                cin, cout = nf + (c - 1) * gc, (gc if c < 5 else nf)
                state[f"model.1.sub.0.RDB{r}.conv{c}.0.weight"] = rng.standard_normal((cout, cin, 3, 3))
                state[f"model.1.sub.0.RDB{r}.conv{c}.0.bias"] = rng.standard_normal(cout)
        for name, cin, cout in (("model.1.sub.1", nf, nf), ("model.3", nf, nf), ("model.6", nf, nf), ("model.8", nf, nf), ("model.10", nf, 3)):
            state[name + ".weight"] = rng.standard_normal((cout, cin, 3, 3))
            state[name + ".bias"] = rng.standard_normal(cout)
        state["model.10.weight"] = np.zeros((3, nf, 3, 3))
        state["model.10.bias"] = np.full(3, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.pth")
            checkpoint(state, path, legacy=True)
            net = rrdb.load(path)
        self.assertEqual(net[1], 1)
        self.assertEqual(net[0]["conv_body.weight"].shape, (nf, nf, 3, 3))
        out = rrdb.upscale_image(net, rng.integers(0, 256, (3, 5, 3)).astype(np.uint8))
        self.assertEqual(out.shape, (12, 20, 3))
        self.assertTrue((out == 128).all())


class TerrainTests(unittest.TestCase):
    def tile(self):
        rgba = np.zeros((15, 32, 4), np.uint8)
        rgba[:, :, :3] = (120, 50, 10)
        for y in range(15):
            half = min(16, int(16 * (1 - abs(y - 7) / 7.5)) + 1)
            rgba[y, 16 - half:16 + half, 3] = 255
        rgba[7, 10:12, :3] = 0
        return rgba

    def test_neighbouring_ground_tiles_continue_the_texture_without_a_seam(self):
        texture = terrain.GroundTexture(np.random.default_rng(1).integers(0, 256, (64, 64, 3)))
        big = terrain.ground_tile(self.tile(), texture, amount=1.0, rocks=0.0)
        # A neighbour sits 16 pixels across and 8 down, so its pixel at (x, y) shows the ground at (x + 16, y + 8)
        # of this tile. With an even base colour both must show the same texture there.
        k = terrain.SCALE
        ys, xs = np.mgrid[4 * k:6 * k, 18 * k:22 * k]
        x, y = (xs + 0.5) / k, (ys + 0.5) / k
        here = texture.sample(texture.detail, x / 32 + y / 16, -x / 32 + y / 16)
        there = texture.sample(texture.detail, (x - 16) / 32 + (y - 8) / 16, -(x - 16) / 32 + (y - 8) / 16)
        self.assertTrue(np.allclose(here, there, atol=1e-4))
        self.assertEqual(big.shape, (15 * k, 32 * k, 4))

    def test_flat_tile_counts_as_ground_and_tile_with_bright_marking_does_not(self):
        tile = self.tile()
        self.assertTrue(terrain.is_ground(tile))
        marked = tile.copy()
        marked[7, 15:17, :3] = (255, 255, 0)
        self.assertFalse(terrain.is_ground(marked))
        tall = np.zeros((40, 32, 4), np.uint8)
        tall[:, :, 3] = 255
        self.assertFalse(terrain.is_ground(tall))

    def test_ground_tile_keeps_size_transparency_and_dark_rocks(self):
        texture = terrain.GroundTexture(np.full((32, 32, 3), 128, np.uint8))
        big = terrain.ground_tile(self.tile(), texture)
        k = terrain.SCALE
        self.assertEqual(big.shape, (15 * k, 32 * k, 4))
        self.assertEqual(big[0, 0, 3], 0)
        self.assertEqual(big[7 * k + 4, 16 * k, 3], 255)
        self.assertLess(big[7 * k + 4, 10 * k + 8, :3].mean(), 40)  # the rock stays dark
        self.assertGreater(big[7 * k + 4, 20 * k, 0], 90)  # the soil keeps its colour


class SpriteTests(unittest.TestCase):
    def test_picture_replaced_once_the_game_stops_reading_the_old_one(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            target, temporary = os.path.join(tmp, "a.png"), os.path.join(tmp, "a.png.part")
            with open(target, "wb") as f:
                f.write(b"old")
            with open(temporary, "wb") as f:
                f.write(b"new")
            reader = open(target, "rb")  # held open like the game's picture decoder
            threading.Timer(0.6, reader.close).start()
            upscale_sprites.replace_file(temporary, target)
            with open(target, "rb") as f:
                self.assertEqual(f.read(), b"new")
            self.assertFalse(os.path.exists(temporary))

    def test_picture_that_cannot_be_replaced_leaves_no_temporary_file(self):
        original_sleep = upscale_sprites.time.sleep
        upscale_sprites.time.sleep = lambda _seconds: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target, temporary = os.path.join(tmp, "a.png"), os.path.join(tmp, "a.png.part")
                with open(target, "wb") as f:
                    f.write(b"old")
                with open(temporary, "wb") as f:
                    f.write(b"new")
                with open(target, "rb"):
                    with self.assertRaises(PermissionError):
                        upscale_sprites.replace_file(temporary, target)
                self.assertFalse(os.path.exists(temporary))
                with open(target, "rb") as f:
                    self.assertEqual(f.read(), b"old")
        finally:
            upscale_sprites.time.sleep = original_sleep

    def test_transparent_pixels_take_colour_of_visible_neighbours_and_visible_ones_stay(self):
        rgb = np.zeros((3, 4, 3), np.uint8)
        rgb[1, 1] = (200, 100, 50)
        rgb[1, 2] = (100, 50, 0)
        visible = np.zeros((3, 4), bool)
        visible[1, 1:3] = True
        filled = upscale_sprites.fill_transparent(rgb, visible)
        self.assertEqual(filled[1, 1].tolist(), [200, 100, 50])
        self.assertEqual(filled[1, 0].tolist(), [200, 100, 50])
        self.assertEqual(filled[1, 3].tolist(), [100, 50, 0])
        self.assertTrue((filled[0, 1:3] > 0).all())

    def test_upscaled_sprite_is_four_times_as_large_and_keeps_its_transparency(self):
        layers = [(np.zeros((48, 3, 3, 3), np.float32), np.zeros(48, np.float32), None)]
        rgba = np.zeros((5, 6, 4), np.uint8)
        rgba[2, 3] = (10, 20, 30, 255)
        big = upscale_sprites.upscale_sprite(lambda image: reference_upscale.upscale_image(layers, 4, image), rgba)
        self.assertEqual(big.shape, (20, 24, 4))
        self.assertEqual(big[8:12, 12:16, 3].min(), 255)
        self.assertEqual(big[:8, :, 3].max(), 0)
        self.assertEqual(big[9, 13, :3].tolist(), [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
