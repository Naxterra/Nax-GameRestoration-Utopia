"""Tests for tools/ai/remaster.py. They need the PyTorch environment described in the README, because the tool
imports PyTorch and Pillow, and are skipped by any other Python:
    "%LOCALAPPDATA%\\Nax Game Restoration\\ai-env\\Scripts\\python.exe" -I -m unittest discover -s tests -p test_remaster.py
No model is loaded; only the steps around the model are tested."""
import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "ai"))
HAVE_TOOLS = all(importlib.util.find_spec(m) for m in ("torch", "PIL"))
if HAVE_TOOLS:
    import remaster  # noqa: E402
    from PIL import Image  # noqa: E402


def diamond(width=32, rows=15, colour=(120, 50, 20)):
    """A flat isometric tile: a filled diamond of one colour with transparent corners."""
    rgba = np.zeros((rows, width, 4), np.uint8)
    for y in range(rows):
        half = min(y + 1, rows - y) * width // (rows + 1)
        rgba[y, width // 2 - half:width // 2 + half] = colour + (255,)
    return rgba


@unittest.skipUnless(HAVE_TOOLS, "needs the PyTorch environment")
class RemasterTests(unittest.TestCase):
    def test_ground_tile_field_leaves_no_background_around_the_middle_tile(self):
        canvas, (x0, y0, x1, y1) = remaster.ground_canvas(diamond())
        around = np.asarray(canvas)[y0 - 40:y1 + 40, x0 - 40:x1 + 40]
        self.assertFalse((around == 255).all(axis=2).any())

    def test_flat_tiles_get_tile_prompts_and_tall_sprites_the_building_prompt(self):
        self.assertIs(remaster.prompt_for(np.zeros((28, 32, 4), np.uint8)), remaster.BUILDING_PROMPT)
        soil = diamond()
        self.assertIs(remaster.prompt_for(soil), remaster.GROUND_PROMPT)
        marked = soil.copy()
        marked[7, 4:28, :3] = (255, 255, 0)  # a bright painted line
        self.assertIs(remaster.prompt_for(marked), remaster.FLAT_PROMPT)

    def test_version_closer_to_the_original_has_the_smaller_difference(self):
        original = diamond()
        faithful = np.asarray(Image.fromarray(original[..., :3], "RGB").resize((256, 120), Image.NEAREST))
        changed = faithful.copy()
        changed[40:80, 60:200] = (255, 255, 0)  # a marking that is not in the original
        self.assertEqual(remaster.difference(faithful, original), 0.0)
        self.assertGreater(remaster.difference(changed, original), 10.0)

    def test_broad_colours_come_from_the_original_and_fine_detail_from_the_model(self):
        model = np.full((80, 80, 3), (200, 40, 40), np.uint8)
        model[::4, :] = (230, 70, 70)  # fine stripes the model painted
        original = np.full((10, 10, 3), (40, 40, 200), np.uint8)
        result = remaster.keep_colours(model, original, 6.0).astype(np.float32)
        self.assertLess(abs(result[..., 2].mean() - 200), 15)  # blue like the original
        self.assertGreater(result[0::4, :].mean(), result[2::4, :].mean() + 10)  # the stripes are still there


if __name__ == "__main__":
    unittest.main()
