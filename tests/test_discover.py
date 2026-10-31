"""Tests for code discovery, using hand-assembled 16-bit programs."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "recomp"))

import discover  # noqa: E402
import mz  # noqa: E402

EXIT = bytes.fromhex("b44ccd21")  # mov ah, 4Ch / int 21h


def program(code):
    return mz.MzImage(bytearray(code), [], 0, 0, 0, 0x100)


def explore(code):
    disc = discover.Discovery(program(code))
    disc.run()
    return disc


def settings_table_program(second_store=bytes.fromhex("2ec70640000200"), initial=0):
    """A program whose jump table is indexed by the word at cs:0040, which the code sets to 0 or 2.

    The table holds a third entry that points at 002C, which only an initial value of 4 can select.
    """
    code = bytearray(0x50)
    code[0x00:0x07] = bytes.fromhex("2ec70640000000")  # mov word ptr cs:[0040], 0
    code[0x07:0x0E] = second_store                      # mov word ptr cs:[0040], 2 (by default)
    code[0x0E:0x13] = bytes.fromhex("2e8b1e4000")      # mov bx, word ptr cs:[0040]
    code[0x13:0x18] = bytes.fromhex("2effa73000")      # jmp word ptr cs:[bx + 0030]
    code[0x20:0x24] = EXIT
    code[0x28:0x2C] = EXIT
    code[0x2C:0x30] = EXIT
    code[0x30:0x36] = bytes.fromhex("200028002c00")    # dw 0020, 0028, 002C
    code[0x40:0x42] = initial.to_bytes(2, "little")
    return code


def bounded_table_program(guard):
    """A table jump at 0007 preceded by the given guard, then "shl bx, 1"; the table at 0040 points at 0020 and 0028."""
    code = bytearray(0x60)
    code[0x00:len(guard)] = guard
    code[0x05:0x07] = bytes.fromhex("d1e3")            # shl bx, 1
    code[0x07:0x0C] = bytes.fromhex("2effa74000")      # jmp word ptr cs:[bx + 0040]
    code[0x15:0x19] = EXIT
    code[0x20:0x24] = EXIT
    code[0x28:0x2C] = EXIT
    code[0x40:0x46] = bytes.fromhex("200028009900")    # dw 0020, 0028, 0099
    return code


class SettingsTableTests(unittest.TestCase):
    def test_table_indexed_by_settings_variable_follows_only_reachable_entries(self):
        disc = explore(settings_table_program())
        self.assertIn(0x20, disc.code)
        self.assertIn(0x28, disc.code)
        self.assertNotIn(0x2C, disc.code)
        self.assertEqual(disc.indirect, [])
        self.assertEqual(len(disc.tables), 1)

    def test_initial_value_of_settings_variable_selects_its_table_entry(self):
        disc = explore(settings_table_program(initial=4))
        self.assertIn(0x2C, disc.code)

    def test_table_stays_unresolved_when_variable_gets_a_computed_value(self):
        # mov word ptr cs:[0040], ax makes the possible values unknown, so guessing a length would be unsafe.
        disc = explore(settings_table_program(second_store=bytes.fromhex("2ea34000909090")))
        self.assertNotIn(0x20, disc.code)
        self.assertEqual(len(disc.indirect), 1)
        self.assertEqual(disc.tables, [])

    def test_computed_word_store_overlapping_variable_low_byte_makes_it_unknown(self):
        # mov word ptr cs:[003F], ax also writes the low byte of the variable at 0040.
        disc = explore(settings_table_program(second_store=bytes.fromhex("2ea33f00909090")))
        self.assertEqual(disc.tables, [])
        self.assertEqual(len(disc.indirect), 1)

    def test_report_never_calls_settings_tables_proven_when_no_aliasing_store_is_detected(self):
        disc = explore(settings_table_program())
        self.assertEqual(disc.alias_risks, [])
        report = disc.report("test")
        self.assertIn("no aliasing store was detected", report)
        self.assertNotIn("are proven", report)

    def test_jump_between_index_load_and_table_jump_marks_table_unproven(self):
        code = bytearray(0x50)
        code[0x00:0x07] = bytes.fromhex("2ec70640000000")  # mov word ptr cs:[0040], 0
        code[0x07:0x09] = bytes.fromhex("720f")            # jc 0018
        code[0x09:0x0E] = bytes.fromhex("2e8b1e4000")      # mov bx, word ptr cs:[0040]
        code[0x0E:0x13] = bytes.fromhex("2effa73000")      # jmp word ptr cs:[bx + 0030]
        code[0x18:0x1D] = bytes.fromhex("bb0400" "ebf1")   # mov bx, 4 / jmp 000E, which skips the load
        code[0x20:0x24] = EXIT
        code[0x30:0x34] = bytes.fromhex("20002800")
        disc = explore(code)
        self.assertTrue(disc.tables[0][4].endswith("(unproven)"))
        self.assertTrue(any("unproven" in text for _cs, _ip, text in disc.indirect))

    def test_settings_table_in_second_code_segment_reports_no_false_growth(self):
        # A relocated far jump enters segment 0001, where the settings program lies. Segment 0000 holds other bytes.
        code = bytearray(0x10) + settings_table_program()
        code[0:5] = bytes.fromhex("ea00000100")              # ljmp 0001:0000
        disc = discover.Discovery(mz.MzImage(code, [(0, 3)], 0, 0, 0, 0x100))
        disc.run()
        self.assertEqual(len(disc.tables), 1)
        self.assertEqual(disc.grown_vars, [])

    def second_segment_program(self, store):
        """Runs the given store in segment 0000, then jumps far to the settings program in segment 0001."""
        code = bytearray(0x10) + settings_table_program()
        code[0:7] = store
        code[7:12] = bytes.fromhex("ea00000100")              # ljmp 0001:0000, segment word at 000A
        disc = discover.Discovery(mz.MzImage(code, [(0, 10)], 0, 0, 0, 0x100))
        disc.run()
        return disc

    def test_store_to_same_offset_in_other_segment_does_not_add_a_value(self):
        # 0000:0040 is linear 0040, but the variable 0001:0040 is linear 0050.
        disc = self.second_segment_program(bytes.fromhex("2ec70640000700"))  # mov word ptr cs:[0040], 7
        self.assertEqual(len(disc.tables), 1)
        self.assertEqual(disc.tables[0][3], 2)

    def test_store_through_overlapping_segment_adds_its_value_to_the_variable(self):
        # 0000:0050 and 0001:0040 name the same linear word, so the value 4 selects the third table entry.
        disc = self.second_segment_program(bytes.fromhex("2ec70650000400"))  # mov word ptr cs:[0050], 4
        self.assertEqual(disc.tables[0][3], 3)
        self.assertIn(0x10 + 0x2C, disc.code)

    def test_explicit_ds_store_with_bp_base_counts_as_alias_risk_after_ds_is_loaded_from_cs(self):
        disc = explore(bytes.fromhex("0e1f" "3e894600") + EXIT)  # push cs / pop ds / mov ds:[bp], ax
        self.assertEqual(len(disc.alias_risks), 1)

    def test_register_store_through_cs_marks_settings_tables_as_assumptions(self):
        # mov word ptr cs:[bx], ax could write the variable without naming it.
        disc = explore(settings_table_program(second_store=bytes.fromhex("2e890790909090")))
        self.assertEqual(len(disc.tables), 1)
        self.assertEqual(len(disc.alias_risks), 1)
        self.assertIn("assumptions", disc.report("test"))


class BoundedTableTests(unittest.TestCase):
    def test_table_bounded_by_ja_before_doubling_reads_exactly_that_many_entries(self):
        disc = explore(bounded_table_program(bytes.fromhex("83fb01" "7710")))  # cmp bx, 1 / ja 0015
        self.assertIn(0x20, disc.code)
        self.assertIn(0x28, disc.code)
        self.assertNotIn(0x99, disc.code)
        self.assertEqual(disc.indirect, [])

    def test_jump_into_bounded_sequence_after_cmp_marks_table_unproven(self):
        code = bytearray(0x50)
        code[0x00:0x02] = bytes.fromhex("720e")            # jc 0010
        code[0x02:0x07] = bytes.fromhex("83fb01" "7729")    # cmp bx, 1 / ja 0030
        code[0x07:0x09] = bytes.fromhex("d1e3")            # shl bx, 1
        code[0x09:0x0E] = bytes.fromhex("2effa74000")      # jmp word ptr cs:[bx + 0040]
        code[0x10:0x15] = bytes.fromhex("bb0500" "ebf2")   # mov bx, 5 / jmp 0007, which skips the cmp
        code[0x20:0x24] = EXIT
        code[0x28:0x2C] = EXIT
        code[0x30:0x34] = EXIT
        code[0x40:0x44] = bytes.fromhex("20002800")
        disc = explore(code)
        self.assertTrue(disc.tables[0][4].endswith("(unproven)"))
        self.assertTrue(any("unproven" in text for _cs, _ip, text in disc.indirect))

    def test_jb_before_table_jump_is_not_taken_as_upper_bound(self):
        # The jump is reached when jb is not taken, that is with bx >= 4, so the cmp proves no upper bound.
        disc = explore(bounded_table_program(bytes.fromhex("83fb04" "7210")))  # cmp bx, 4 / jb 0015
        self.assertEqual(disc.tables, [])
        self.assertNotIn(0x20, disc.code)
        self.assertEqual(len(disc.indirect), 1)

    def test_cmp_on_other_register_does_not_bound_table(self):
        disc = explore(bounded_table_program(bytes.fromhex("3c03" "90" "7710")))  # cmp al, 3 / nop / ja 0015
        self.assertEqual(disc.tables, [])
        self.assertEqual(len(disc.indirect), 1)

    def test_table_running_past_image_end_is_recorded_outside_without_crashing(self):
        code = bytearray(0x20)
        code[0x00:0x05] = bytes.fromhex("83fb03" "770b")    # cmp bx, 3 / ja 0010
        code[0x05:0x07] = bytes.fromhex("d1e3")            # shl bx, 1
        code[0x07:0x0C] = bytes.fromhex("2effa71c00")      # jmp word ptr cs:[bx + 001C]
        code[0x10:0x14] = EXIT
        code[0x1C:0x20] = bytes.fromhex("10001000")        # two entries fit, two lie past the end
        disc = explore(code)
        self.assertIn(0x10, disc.code)
        self.assertEqual(len(disc.outside), 2)


class FlowTests(unittest.TestCase):
    def test_dos_exit_ends_the_path_so_following_bytes_are_not_decoded(self):
        disc = explore(EXIT + bytes.fromhex("ffff"))
        self.assertEqual(sorted(disc.code), [0, 2])

    def test_jump_straight_to_exit_int_explores_code_after_it(self):
        # jc 0004 reaches the int 21h without the mov ah, 4Ch before it, so AH is unknown on that path.
        disc = explore(bytes.fromhex("7202" "b44c" "cd21") + EXIT)
        self.assertIn(0x06, disc.code)

    def test_handler_installed_with_unknown_ds_is_reported_not_decoded(self):
        code = bytearray(0x30)
        code[0:12] = bytes.fromhex("ba2000" "b80825" "cd21") + EXIT[:2]  # mov dx, 0020 / mov ax, 2508h / int 21h
        code[12:14] = EXIT[2:]
        code[0x20] = 0xCF                                              # iret
        disc = explore(code)
        self.assertNotIn(0x20, disc.code)
        self.assertTrue(any("DS:DX" in text for _cs, _ip, text in disc.indirect))

    def test_handler_installed_with_ds_from_cs_is_decoded(self):
        code = bytearray(0x30)
        code[0:14] = bytes.fromhex("0e1f" "ba2000" "b80825" "cd21") + EXIT[:2]  # push cs / pop ds / ...
        code[14:16] = EXIT[2:]
        code[0x20] = 0xCF
        disc = explore(code)
        self.assertIn(0x20, disc.code)
        self.assertIn((0, 0x20), disc.int_handlers)
        self.assertEqual(disc.unproven, [])

    def test_jump_between_dx_load_and_handler_install_marks_handler_unproven(self):
        code = bytearray(0x30)
        code[0x00:0x02] = bytes.fromhex("720e")                            # jc 0010
        code[0x02:0x0C] = bytes.fromhex("0e1f" "ba2000" "b80825" "cd21")   # push cs / pop ds / mov dx, 0020 / mov ax, 2508h / int 21h
        code[0x0C:0x10] = EXIT
        code[0x10:0x15] = bytes.fromhex("ba2800" "ebf2")                   # mov dx, 0028 / jmp 0007, a second handler address
        code[0x20] = 0xCF
        disc = explore(code)
        self.assertEqual(len(disc.unproven), 1)
        self.assertTrue(any("handler" in text and "unproven" in text for _cs, _ip, text in disc.indirect))

    def test_jump_between_ah_load_and_handler_install_marks_handler_unproven(self):
        code = bytearray(0x30)
        code[0x00:0x02] = bytes.fromhex("720e")                            # jc 0010
        code[0x02:0x0B] = bytes.fromhex("b425" "0e1f" "ba2000" "cd21")     # mov ah, 25h / push cs / pop ds / mov dx, 0020 / int 21h
        code[0x0B:0x0F] = EXIT
        code[0x10:0x14] = bytes.fromhex("b430" "ebf0")                     # mov ah, 30h / jmp 0004, a different DOS service
        code[0x20] = 0xCF
        disc = explore(code)
        self.assertEqual(len(disc.unproven), 1)

    def test_memory_destination_without_access_flags_is_treated_as_written(self):
        mem = types.SimpleNamespace(type=discover.x86.X86_OP_MEM, access=0)
        reg = types.SimpleNamespace(type=discover.x86.X86_OP_REG, access=0)
        self.assertEqual(len(discover.Discovery.stored_operands(types.SimpleNamespace(mnemonic="mov", operands=[mem]))), 1)
        self.assertEqual(discover.Discovery.stored_operands(types.SimpleNamespace(mnemonic="cmp", operands=[mem])), [])
        # xchg ax, word ptr cs:[x] writes memory through its second operand.
        self.assertEqual(len(discover.Discovery.stored_operands(types.SimpleNamespace(mnemonic="xchg", operands=[reg, mem]))), 1)

    def test_far_call_through_memory_is_reported_as_run_time_target(self):
        disc = explore(bytes.fromhex("ff5f10") + EXIT)  # lcall [bx + 10]
        self.assertEqual(len(disc.indirect), 1)
        self.assertIn("far pointer in memory", disc.indirect[0][2])

    def test_instruction_covering_start_of_decoded_instruction_is_counted_as_overlap(self):
        # 0006 (ret) is decoded first. The path to 0005 then decodes "mov al, C3", which would cover 0006.
        disc = explore(bytes.fromhex("7203" "7202" "c3" "b0" "c3"))
        self.assertEqual(disc.overlaps, [(0, 5)])
        self.assertNotIn(0x05, disc.code)

    def test_report_for_empty_image_does_not_divide_by_zero(self):
        disc = explore(b"")
        self.assertIn("no image", disc.report("empty"))


if __name__ == "__main__":
    unittest.main()
