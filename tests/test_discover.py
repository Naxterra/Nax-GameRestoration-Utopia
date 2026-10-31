"""Tests for code discovery, using hand-assembled 16-bit programs."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "recomp"))

import discover  # noqa: E402
import mz  # noqa: E402

EXIT = bytes.fromhex("b44ccd21")  # mov ah, 4Ch / int 21h


def program(code):
    return mz.MzImage(bytearray(code), [], 0, 0, 0, 0x100)


def settings_table_program(second_store=bytes.fromhex("2ec70640000200")):
    """A program whose jump table is indexed by the word at cs:0040, which is only ever set to 0 or 2.

    The table holds a third entry that points at 0099, which no value of the variable can select.
    """
    code = bytearray(0x50)
    code[0x00:0x07] = bytes.fromhex("2ec70640000000")  # mov word ptr cs:[0040], 0
    code[0x07:0x0E] = second_store                      # mov word ptr cs:[0040], 2 (by default)
    code[0x0E:0x13] = bytes.fromhex("2e8b1e4000")      # mov bx, word ptr cs:[0040]
    code[0x13:0x18] = bytes.fromhex("2effa73000")      # jmp word ptr cs:[bx + 0030]
    code[0x20:0x24] = EXIT
    code[0x28:0x2C] = EXIT
    code[0x30:0x36] = bytes.fromhex("200028009900")    # dw 0020, 0028, 0099
    return program(code)


class DiscoveryTests(unittest.TestCase):
    def test_table_indexed_by_settings_variable_follows_only_reachable_entries(self):
        disc = discover.Discovery(settings_table_program())
        disc.run()
        self.assertIn(0x20, disc.code)
        self.assertIn(0x28, disc.code)
        self.assertNotIn(0x99, disc.code)
        self.assertEqual(disc.indirect, [])
        self.assertEqual(len(disc.tables), 1)

    def test_table_stays_unresolved_when_variable_gets_a_computed_value(self):
        # mov word ptr cs:[0040], ax makes the possible values unknown, so guessing a length would be unsafe.
        disc = discover.Discovery(settings_table_program(second_store=bytes.fromhex("2ea34000909090")))
        disc.run()
        self.assertNotIn(0x20, disc.code)
        self.assertEqual(len(disc.indirect), 1)
        self.assertEqual(disc.tables, [])

    def test_table_bounded_by_cmp_before_doubling_reads_exactly_that_many_entries(self):
        code = bytearray(0x60)
        code[0x00:0x03] = bytes.fromhex("83fb01")          # cmp bx, 1
        code[0x03:0x05] = bytes.fromhex("7710")            # ja 0015
        code[0x05:0x07] = bytes.fromhex("d1e3")            # shl bx, 1
        code[0x07:0x0C] = bytes.fromhex("2effa74000")      # jmp word ptr cs:[bx + 0040]
        code[0x15:0x19] = EXIT
        code[0x20:0x24] = EXIT
        code[0x28:0x2C] = EXIT
        code[0x40:0x46] = bytes.fromhex("200028009900")    # dw 0020, 0028, 0099
        disc = discover.Discovery(program(code))
        disc.run()
        self.assertIn(0x20, disc.code)
        self.assertIn(0x28, disc.code)
        self.assertNotIn(0x99, disc.code)
        self.assertEqual(disc.indirect, [])

    def test_dos_exit_ends_the_path_so_following_bytes_are_not_decoded(self):
        disc = discover.Discovery(program(EXIT + bytes.fromhex("ffff")))
        disc.run()
        self.assertEqual(sorted(disc.code), [0, 2])


if __name__ == "__main__":
    unittest.main()
