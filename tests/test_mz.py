"""Tests for the MZ loader and EXEPACK unpacker, built from small synthetic programs so no game files are needed."""
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "recomp"))

import mz  # noqa: E402


def mz_file(body, cs, ip, relocs=(), ss=0, sp=0x100):
    """Wraps a program body in an MZ header whose size is rounded up to whole paragraphs."""
    header_size = (28 + 4 * len(relocs) + 15) // 16 * 16
    total = header_size + len(body)
    header = struct.pack("<2s13H", b"MZ", total % 512, (total + 511) // 512, len(relocs), header_size // 16,
                         0, 0xFFFF, ss, sp, 0, ip, cs, 28, 0)
    header += b"".join(struct.pack("<HH", off, seg) for seg, off in relocs)
    return header.ljust(header_size, b"\0") + body


EXPECTED = b"ABCD" + b"\xAA" * 20 + b"XYZ" + b"\xBB" * 5


def packed_body(stream_tail=None):
    """Builds an EXEPACK body that expands to EXPECTED and carries one relocation at 0000:0010."""
    stream = b"ABCD" + b"\xAA" + struct.pack("<H", 20) + b"\xB1" + b"XYZ" + struct.pack("<H", 3) + b"\xB2" \
        + b"\xBB" + struct.pack("<H", 5) + (stream_tail or b"\xB0")
    compressed = stream.ljust(32, b"\xFF")
    stub_code = b"\x90" * 8 + mz.EXEPACK_ERROR_TEXT
    relocs = struct.pack("<HH", 1, 0x0010) + struct.pack("<H", 0) * 15
    exepack_size = 18 + len(stub_code) + len(relocs)
    header = struct.pack("<8H", 0x0003, 0x0000, 0, exepack_size, 0x0100, 0x0001, 2, 1) + mz.EXEPACK_SIGNATURE
    return compressed + header + stub_code + relocs


class MzLoaderTests(unittest.TestCase):
    def load_bytes(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prog.exe")
            with open(path, "wb") as f:
                f.write(data)
            return mz.load(path)

    def test_exepack_fill_and_copy_commands_rebuild_original_image(self):
        image = self.load_bytes(mz_file(packed_body(), cs=2, ip=0x12))
        self.assertEqual(image.packed_with, "EXEPACK")
        self.assertEqual(bytes(image.data), EXPECTED)
        self.assertEqual((image.cs, image.ip, image.ss, image.sp), (0, 3, 1, 0x100))
        self.assertEqual(image.relocs, [(0, 0x0010)])

    def test_exepack_unknown_command_is_rejected_instead_of_producing_an_image(self):
        with self.assertRaises(mz.MzError):
            self.load_bytes(mz_file(packed_body(stream_tail=b"\xC4"), cs=2, ip=0x12))

    def test_unpacked_program_keeps_body_entry_point_and_relocations(self):
        body = bytes(range(48))
        image = self.load_bytes(mz_file(body, cs=1, ip=4, relocs=[(0, 6), (2, 8)]))
        self.assertEqual(image.packed_with, "")
        self.assertEqual(bytes(image.data), body)
        self.assertEqual((image.cs, image.ip), (1, 4))
        self.assertEqual(image.relocs, [(0, 6), (2, 8)])

    def test_file_without_mz_signature_is_rejected(self):
        with self.assertRaises(mz.MzError):
            self.load_bytes(b"PK\x03\x04" + b"\0" * 60)


if __name__ == "__main__":
    unittest.main()
