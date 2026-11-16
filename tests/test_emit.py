"""Tests for the C++ generator's pixel origin tracking and hooks, using hand-assembled 16-bit programs."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "recomp"))

import discover  # noqa: E402
import emit  # noqa: E402
import mz  # noqa: E402

EXIT = bytes.fromhex("b44ccd21")  # mov ah, 4Ch / int 21h


def generate(code, hooks=None):
    disc = discover.Discovery(mz.MzImage(bytearray(code), [], 0, 0, 0, 0x100))
    disc.run()
    text = emit.Emitter(disc, "t", hooks=hooks).module("t.bin", 0, False, b"")
    return {line.split("// ")[-1].strip(): line for line in text.splitlines() if "// " in line and "case 0x" in line}, text


class OriginTests(unittest.TestCase):
    def test_byte_loaded_from_memory_and_stored_keeps_its_origin(self):
        lines, _ = generate(bytes.fromhex("8a04" "268805") + EXIT)  # mov al, [si] / mov es:[di], al
        self.assertIn("c.org[0] = g0", lines["mov al, byte ptr [si]"])
        self.assertIn("m.org(s0", lines["mov al, byte ptr [si]"])
        self.assertIn("m.wr8o(s0, o0, c.a.l, c.org[0]);", lines["mov byte ptr es:[di], al"])

    def test_computed_register_loses_its_origin(self):
        lines, _ = generate(bytes.fromhex("8a04" "0401") + EXIT)  # mov al, [si] / add al, 1
        self.assertIn("c.org[0] = 0;", lines["add al, 1"])

    def test_string_copy_carries_origin_of_both_bytes(self):
        lines, _ = generate(bytes.fromhex("a5") + EXIT)  # movsw
        self.assertIn("m.wr16o(c.es, c.di, m.rd16(c.ds, c.si), m.org(c.ds, c.si), m.org(c.ds, (uint16_t)(c.si + 1)))", lines["movsw word ptr es:[di], word ptr [si]"])

    def test_hook_runs_before_its_instruction_and_is_declared(self):
        lines, text = generate(bytes.fromhex("8a04") + EXIT, hooks={(0, 2): "watch_exit"})
        self.assertIn("void watch_exit(nax::Machine& m);", text)
        self.assertIn("watch_exit(m);", lines["mov ah, 0x4c"])
        self.assertNotIn("watch_exit(m);", lines["mov al, byte ptr [si]"])


if __name__ == "__main__":
    unittest.main()
