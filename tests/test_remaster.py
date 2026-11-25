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
        marked[7, 4:28, :3] = (255, 255, 0)  # a bright painted line on non-grey soil
        self.assertIs(remaster.prompt_for(marked), remaster.FLAT_PROMPT)
        pad = diamond(colour=(65, 65, 70))
        pad[7, 4:28, :3] = (255, 255, 0)  # dark grey platform with yellow lines
        self.assertIs(remaster.prompt_for(pad), remaster.LANDING_PAD_PROMPT)
        turret = diamond(colour=(95, 95, 105))
        turret[6:9, 6:26, :3] = (210, 20, 20)  # grey metal dome bunker with red ring, no yellow stripes
        self.assertIs(remaster.prompt_for(turret), remaster.BUILDING_PROMPT)
        self.assertIsNone(remaster.kind_of(turret))
        self.assertEqual(remaster.tile_groups({"pad": pad, "turret": turret}, [("pad", "turret", 16, 8)]), [])

    def test_ground_prompts_adapt_to_planet_biome_and_icons_get_icon_prompt(self):
        green = diamond(colour=(30, 110, 40))
        ice = diamond(colour=(185, 210, 235))
        crystal = diamond(colour=(120, 50, 130))
        self.assertIn("grassland", remaster.prompt_for(green).lower())
        self.assertIn("snowfield", remaster.prompt_for(ice).lower())
        self.assertIn("crystalline", remaster.prompt_for(crystal).lower())
        icon = np.full((16, 16, 4), (160, 160, 180, 255), np.uint8)
        self.assertIs(remaster.prompt_for(icon), remaster.ICON_PROMPT)
        self.assertIsNone(remaster.kind_of(icon))

    def test_natural_mountain_formations_get_geology_prompt_while_buildings_keep_building_prompt(self):
        glacier = np.zeros((28, 32, 4), np.uint8)
        glacier[4:, 2:30] = (155, 160, 240, 255)  # pale blue glacier snow on dark cliff rock
        glacier[18:, 6:14] = (65, 68, 75, 255)
        self.assertTrue(remaster.is_natural_formation(glacier))
        self.assertIn("glacier", remaster.prompt_for(glacier).lower())
        self.assertIsNone(remaster.kind_of(glacier))
        mesa = np.zeros((28, 32, 4), np.uint8)
        mesa[4:, 2:30] = (155, 78, 12, 255)  # Martian sandstone mesa without windows
        self.assertTrue(remaster.is_natural_formation(mesa))
        self.assertIn("sandstone", remaster.prompt_for(mesa).lower())
        # Adding glowing yellow windows turns a tall sprite back into a colony building.
        building = glacier.copy()
        building[10:12, 12:15] = (240, 230, 20, 255)
        self.assertFalse(remaster.is_natural_formation(building))
        self.assertIs(remaster.prompt_for(building), remaster.BUILDING_PROMPT)
        self.assertEqual(remaster.kind_of(building), "building")
        # Neutral grey/silver metallic structures without yellow windows must never be misclassified as ice cliffs.
        silver = np.zeros((28, 32, 4), np.uint8)
        silver[4:, 2:30] = (175, 175, 175, 255)
        silver[16:, 6:18] = (70, 70, 70, 255)
        self.assertFalse(remaster.is_natural_formation(silver))
        self.assertIs(remaster.prompt_for(silver), remaster.BUILDING_PROMPT)
        self.assertEqual(remaster.kind_of(silver), "building")

    def test_bleed_edges_replaces_white_canvas_halo_with_opaque_sprite_colours(self):
        colour = np.full((16, 16, 3), 255, np.uint8)
        colour[4:12, 4:12] = (120, 60, 20)
        alpha = np.zeros((16, 16), np.uint8)
        alpha[4:12, 4:12] = 255
        alpha[3, 4:12] = 100  # semi-transparent anti-aliased edge pixel
        bled = remaster.bleed_edges(colour, alpha)
        self.assertEqual(tuple(bled[3, 6]), (120, 60, 20))

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
        result = remaster.keep_colours(model, original, 6.0, luma_strength=1.0, chroma_strength=1.0).astype(np.float32)
        self.assertLess(abs(result[..., 2].mean() - 200), 15)  # full anchor matches original blue
        self.assertGreater(result[0::4, :].mean(), result[2::4, :].mean() + 10)  # the stripes are still there
        # Gentle true-color tinting preserves most of the model's richer shading without forcing full retro saturation.
        gentle = remaster.keep_colours(model, original, 6.0, luma_strength=0.25, chroma_strength=0.35).astype(np.float32)
        self.assertGreater(gentle[..., 0].mean(), 120)
        self.assertLess(gentle[..., 0].mean(), 190)
        self.assertGreater(gentle[0::4, :].mean(), gentle[2::4, :].mean() + 10)

    def test_pad_tiles_drawn_side_by_side_form_one_group_where_the_game_drew_them(self):
        marked = diamond()
        marked[7, 4:28, :3] = (255, 255, 0)
        sprites = {n: marked.copy() for n in "abcd"}
        sprites["ground"] = diamond()
        pairs = [("a", "b", -16, 8), ("a", "c", 16, 8), ("b", "d", 16, 8), ("a", "ground", 0, -16),
                 ("c", "d", -16, 8), ("b", "c", 32, 0)]
        groups = remaster.tile_groups(sprites, pairs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {"a": (0, 0), "b": (-16, 8), "c": (16, 8), "d": (0, 16)})

    def test_unit_on_a_pad_tile_and_tiles_off_the_grid_are_not_joined_to_the_pad(self):
        marked = diamond()
        marked[7, 4:28, :3] = (255, 255, 0)
        unit = np.zeros((15, 32, 4), np.uint8)
        unit[4:12, 10:22] = (255, 255, 0, 255)  # a small yellow vehicle, the size of a tile but not its shape
        sprites = {"a": marked, "b": marked.copy(), "c": marked.copy(), "unit": unit}
        pairs = [("a", "b", 16, 8), ("a", "unit", 16, 8), ("b", "c", 8, 4)]
        self.assertFalse(remaster.is_tile(unit))
        self.assertEqual(remaster.tile_groups(sprites, pairs), [{"a": (0, 0), "b": (16, 8)}])

    def test_four_pieces_of_a_building_on_four_tiles_form_one_group_but_a_whole_row_does_not(self):
        def piece(rows):
            p = np.zeros((rows, 32, 4), np.uint8)
            p[2:, 4:28] = (200, 120, 40, 255)  # a tall building piece
            return p
        # Like the command center: a low piece at the back, two side pieces and a tall one at the front.
        sprites = {"a": piece(23), "b": piece(28), "c": piece(28), "d": piece(30)}
        pairs = [("a", "b", -16, 8), ("a", "c", 16, 8), ("b", "d", 16, 8)]
        self.assertEqual(remaster.tile_groups(sprites, pairs), [{"a": (0, 0), "b": (-16, 8), "c": (16, 8), "d": (0, 16)}])
        # Two one-tile buildings side by side do not make a square, so neither is joined, and that is reported.
        skipped = []
        self.assertEqual(remaster.tile_groups({"house": piece(28), "shop": piece(20)}, [("house", "shop", 16, 8)], skipped), [])
        self.assertEqual(skipped, [{"house": (0, 0), "shop": (16, 8)}])
        row = {f"p{k}": piece(28) for k in range(12)}
        chain = [(f"p{k}", f"p{k + 1}", 16, 8) for k in range(11)]
        self.assertEqual(remaster.tile_groups(row, chain), [])

    def test_unit_standing_on_a_tile_is_not_taken_for_a_building_piece(self):
        unit = np.zeros((24, 32, 4), np.uint8)
        unit[4:20, 10:22] = (90, 90, 120, 255)  # a vehicle in the middle of its tile, ground free around it
        building = np.zeros((24, 32, 4), np.uint8)
        building[2:, 1:31] = (200, 120, 40, 255)
        self.assertIsNone(remaster.kind_of(unit))
        self.assertEqual(remaster.kind_of(building), "building")

    def test_tiles_of_a_group_lie_on_the_canvas_where_the_game_draws_them(self):
        sprites = {"top": diamond(), "right": diamond(colour=(200, 200, 0))}
        canvas, boxes = remaster.group_canvas(sprites, {"top": (0, 0), "right": (16, 8)})
        (x0, y0, x1, _), (x2, y2, _, _) = boxes["top"], boxes["right"]
        factor = (x1 - x0) // 32
        self.assertEqual((x2 - x0, y2 - y0), (16 * factor, 8 * factor))
        middle = np.asarray(canvas)[y2 + 7 * factor + factor // 2, x2 + 16 * factor]
        self.assertEqual(tuple(middle), (200, 200, 0))  # the right tile, drawn after the top one

    def test_sprite_with_the_same_pixels_as_a_redrawn_one_takes_its_picture_instead_of_being_redrawn(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target_of = lambda n: os.path.join(folder, n + ".png")  # noqa: E731
            sprites = {"old": diamond(), "new": diamond(), "other": diamond(colour=(0, 90, 0))}
            with open(target_of("old"), "wb") as f:
                f.write(b"redrawn picture")
            copied = remaster.take_from_twins(sprites, ["new", "other"], target_of, sprites)
            self.assertEqual(copied, ["new"])
            with open(target_of("new"), "rb") as f:
                self.assertEqual(f.read(), b"redrawn picture")
            self.assertFalse(os.path.exists(target_of("other")))

    def test_layout_lines_cut_short_are_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "layout.txt")
            with open(path, "w", encoding="ascii") as f:
                f.write("aa bb 16 8\ncc dd -16\nee ff x 8\n")
            self.assertEqual(remaster.read_layout(path), [("aa", "bb", 16, 8)])
            self.assertEqual(remaster.read_layout(os.path.join(folder, "missing.txt")), [])

    def test_ambient_duplicate_filtering_skips_dimmed_copies_and_prioritises_buildings(self):
        day = diamond(colour=(160, 164, 244))
        twin = diamond(colour=(160, 164, 244))
        night = diamond(colour=(100, 104, 176))
        near = diamond(colour=(156, 160, 240))
        mars = diamond(colour=(160, 64, 16))
        sprites = {"day": day, "twin": twin, "night": night, "near": near, "mars": mars}
        kept = remaster.filter_ambient_duplicates(sprites, ["day", "twin", "night", "near", "mars"], set())
        # Exact-pixel twins are both kept for take_from_twins(), while dimmed and +-2 drift copies are dropped.
        self.assertEqual(kept, ["day", "twin", "mars"])
        kept_after_up = remaster.filter_ambient_duplicates(sprites, ["night", "near", "mars"], {"day"})
        self.assertEqual(kept_after_up, ["mars"])
        # Non-transitive chain: A -> B (0.60, 0.80, 0.80) -> C (0.37, 0.63, 0.63) must keep C because A -> C fails ratio > 1.60.
        chain_a = diamond(colour=(240, 240, 240))
        chain_b = diamond(colour=(144, 192, 192))
        chain_c = diamond(colour=(88, 152, 152))
        kept_chain = remaster.filter_ambient_duplicates({"a": chain_a, "b": chain_b, "c": chain_c}, ["a", "b", "c"], set())
        self.assertEqual(kept_chain, ["a", "c"])
        building = np.zeros((28, 32, 4), np.uint8)
        tile = np.zeros((15, 32, 4), np.uint8)
        icon = np.zeros((16, 16, 4), np.uint8)
        order = sorted([("i", icon), ("t", tile), ("b", building)], key=lambda item: remaster.priority_key_single(*item))
        self.assertEqual([n for n, _ in order], ["b", "t", "i"])

    def test_ice_chasm_edge_gets_chasm_prompt_and_preserves_pitch_black_void(self):
        chasm = diamond(colour=(155, 160, 240))
        chasm[8:, :, :3] = (0, 0, 0)  # lower half is pitch-black abyss void
        self.assertTrue(remaster.is_chasm_edge(chasm))
        self.assertIn("icicles", remaster.prompt_for(chasm).lower())
        self.assertIsNone(remaster.kind_of(chasm))
        fake_qwen = np.full((15 * remaster.SCALE, 32 * remaster.SCALE, 3), (60, 70, 90), np.uint8)
        finished = remaster.finish(fake_qwen, chasm)
        # Both interior and outer diagonal perimeter pixels of the pitch-black abyss must stay solid #000000.
        self.assertEqual(int(finished[12 * remaster.SCALE, 16 * remaster.SCALE, :3].max()), 0)
        lower_visible = finished[10 * remaster.SCALE:, :, 3] >= 128
        self.assertTrue(lower_visible.any())
        self.assertEqual(int(finished[10 * remaster.SCALE:, :, :3][lower_visible].max()), 0)
        # Short pink-flora snow tile gets the alien flora prompt rather than plain snow.
        flora = diamond(colour=(155, 160, 240))
        flora[5:10, 10:22, :3] = (230, 90, 220)
        self.assertTrue(remaster.is_pink_flora_ground(flora))
        self.assertIn("pink-violet", remaster.prompt_for(flora).lower())
        self.assertIsNone(remaster.kind_of(flora))
        # Emerald-green alien toxic lake center and shoreline tiles get natural liquid prompts, never circuit panels.
        lake_center = diamond(colour=(0, 175, 0))
        self.assertTrue(remaster.is_toxic_lake(lake_center))
        self.assertIn("acid lake", remaster.prompt_for(lake_center).lower())
        self.assertIn("zero metal plates", remaster.prompt_for(lake_center).lower())
        self.assertIsNone(remaster.kind_of(lake_center))
        lake_shore = diamond(colour=(160, 165, 245))
        lake_shore[6:, :, :3] = (0, 170, 0)
        self.assertTrue(remaster.is_toxic_lake(lake_shore))
        self.assertTrue(remaster.is_toxic_lake_shore(lake_shore))
        self.assertIn("shoreline", remaster.prompt_for(lake_shore).lower())
        self.assertIsNone(remaster.kind_of(lake_shore))
        # Seamless lake shore compositing matches water_hd strictly on water edges and ground_hd on snow edges.
        water_hd = np.full((15 * remaster.SCALE, 32 * remaster.SCALE, 4), (10, 190, 20, 255), np.uint8)
        snow_hd = np.full((15 * remaster.SCALE, 32 * remaster.SCALE, 4), (210, 225, 250, 255), np.uint8)
        comp = remaster.composite_toxic_lake_shore(lake_shore, water_hd, snow_hd)
        self.assertIsNotNone(comp)
        self.assertEqual(comp.shape, (15 * remaster.SCALE, 32 * remaster.SCALE, 4))
        # Compass indicator marker (flat lavender diamond + bold red emblem) gets COMPASS_MARKER_PROMPT, not BUILDING_PROMPT.
        compass = np.zeros((23, 32, 4), np.uint8)
        compass[8:, :, :] = (160, 160, 200, 255)
        compass[2:18, 14:19, :] = (200, 10, 10, 255)
        self.assertTrue(remaster.is_compass_marker(compass))
        self.assertIsNone(remaster.kind_of(compass))
        self.assertIn("compass marker", remaster.prompt_for(compass).lower())
        self.assertIn("zero skyscrapers", remaster.prompt_for(compass).lower())
        rendered_compass = remaster.render_compass_marker(compass)
        self.assertEqual(rendered_compass.shape, (23 * remaster.SCALE, 32 * remaster.SCALE, 4))
        # Even a single yellow building-window pixel must disqualify a tall sprite from compass classification.
        with_window = compass.copy()
        with_window[10, 10, :3] = (230, 210, 20)
        self.assertFalse(remaster.is_compass_marker(with_window))

        # Upper snow interior stays snow-coloured (blue > 200), lower water interior stays vivid green (green > 170, red < 40).
        self.assertGreater(int(comp[2 * remaster.SCALE, 16 * remaster.SCALE, 2]), 200)
        self.assertGreater(int(comp[12 * remaster.SCALE, 16 * remaster.SCALE, 1]), 170)
        self.assertLess(int(comp[12 * remaster.SCALE, 16 * remaster.SCALE, 0]), 40)

    def test_stabilize_animation_frame_locks_static_building_pixels_and_keeps_moving_part(self):
        base_orig = np.full((30, 32, 4), (100, 100, 110, 255), np.uint8)
        frame_orig = base_orig.copy()
        frame_orig[2:5, 24:28, :3] = (255, 255, 0)  # small animated beacon at top-right
        base_up = np.full((30 * 8, 32 * 8, 4), (40, 50, 60, 255), np.uint8)
        frame_up = np.full((30 * 8, 32 * 8, 4), (200, 180, 30, 255), np.uint8)
        out = remaster.stabilize_animation_frame(base_orig, base_up, frame_orig, frame_up)
        # Far from the animated beacon, static building pixels match base_up identically.
        self.assertEqual(tuple(out[180, 40, :3]), (40, 50, 60))
        # Inside the animated beacon, frame_up's pixels are preserved.
        self.assertGreater(int(out[28, 208, 0]), 180)
        # When frame_orig is taller than base_orig, extra top rows must inherit frame_up's RGB (no black halo).
        taller_orig = np.full((32, 32, 4), (100, 100, 110, 255), np.uint8)
        taller_up = np.full((32 * 8, 32 * 8, 4), (200, 180, 30, 255), np.uint8)
        out_taller = remaster.stabilize_animation_frame(base_orig, base_up, taller_orig, taller_up, dilate_orig=0)
        self.assertEqual(tuple(out_taller[4, 100, :3]), (200, 180, 30))
        # Scale mismatch returns frame_up cleanly instead of crashing.
        wrong_scale = np.full((30 * 4, 32 * 4, 4), (90, 90, 90, 255), np.uint8)
        self.assertIs(remaster.stabilize_animation_frame(base_orig, base_up, frame_orig, wrong_scale), wrong_scale)
        # Mild uniform fade (94% brightness where mid-tones move <= 2 DAC and bright yellow moves 4 DAC) matches.
        ref_pal = [(20, 20, 24), (30, 30, 36), (38, 0, 0), (48, 36, 0), (60, 60, 0)]
        cur_pal = [(19, 19, 23), (28, 28, 34), (36, 0, 0), (45, 34, 0), (56, 56, 0)]
        self.assertTrue(remaster.is_dimmed_or_near_duplicate(ref_pal, cur_pal))
        # Single mildly blinking bright indicator with 4 unchanged colours must NOT match as ambient dim.
        blink_ref = [(16, 16, 16), (18, 18, 18), (20, 20, 20), (22, 22, 22), (63, 63, 63)]
        blink_cur = [(16, 16, 16), (18, 18, 18), (20, 20, 20), (22, 22, 22), (57, 57, 57)]
        self.assertFalse(remaster.is_dimmed_or_near_duplicate(blink_ref, blink_cur))

    def test_solar_panel_nighttime_accent_dim_is_filtered_as_duplicate_while_hue_swap_is_kept(self):
        day_pal = [(51, 0, 0), (44, 0, 0), (60, 60, 0), (0, 17, 0), (33, 0, 0), (0, 30, 0),
                   (38, 22, 12), (29, 28, 34), (31, 31, 31), (24, 24, 34), (16, 16, 24)]
        night_pal = [(31, 0, 0), (26, 0, 0), (29, 22, 0), (0, 10, 0), (20, 0, 0), (0, 17, 0),
                     (22, 12, 8), (17, 16, 24), (19, 19, 23), (14, 14, 24), (9, 9, 17)]
        self.assertTrue(remaster.is_dimmed_or_near_duplicate(day_pal, night_pal))
        swapped_pal = list(night_pal)
        swapped_pal[2] = (0, 22, 29)  # hue swap introducing blue where daytime had zero blue
        self.assertFalse(remaster.is_dimmed_or_near_duplicate(day_pal, swapped_pal))


if __name__ == "__main__":
    unittest.main()
