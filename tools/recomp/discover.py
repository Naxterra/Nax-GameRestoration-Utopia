"""Finds the reachable 16-bit code of an MZ program by recursive descent and reports what a translator must handle.

Usage: python -I discover.py <program> [--listing <file>]

The report covers code coverage, control transfers whose target is only known at run time, DOS and BIOS
interrupt services, hardware port access and stores through the CS segment (data kept inside code, or
code that modifies itself). Segment values are relative to the load segment, as produced by mz.load.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capstone  # noqa: E402
from capstone import x86  # noqa: E402

import mz  # noqa: E402

COND_JUMPS = {
    "jo", "jno", "jb", "jae", "je", "jne", "jbe", "ja", "js", "jns", "jp", "jnp", "jl", "jge", "jle", "jg",
    "jcxz", "loop", "loope", "loopne",
}
PORT_INSTRUCTIONS = {"in", "out", "insb", "insw", "outsb", "outsw"}


class Discovery:
    def __init__(self, image):
        self.image = image
        self.md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_16)
        self.md.detail = True
        self.reloc_words = image.reloc_linear_set()
        self.code = {}  # linear address -> (cs, ip, instruction)
        self.owner = {}  # linear byte -> linear address of the instruction that covers it
        self.functions = set()
        self.int_handlers = set()
        self.indirect = []
        self.interrupts = collections.Counter()
        self.ports = collections.Counter()
        self.cs_stores = []
        self.overlaps = []
        self.outside = []
        self.tables = []  # (cs, ip, table linear address, entry count, how the count was found)
        self.table_bytes = set()
        self.deferred = []  # table jumps waiting for the value set of the variable that indexes them
        self.resolved_vars = {}
        self.grown_vars = []
        self.work = []

    def segword(self, linear):
        return self.image.data[linear] | (self.image.data[linear + 1] << 8)

    def add(self, cs, ip, kind=None):
        ip &= 0xFFFF
        if kind == "function":
            self.functions.add((cs, ip))
        elif kind == "handler":
            self.int_handlers.add((cs, ip))
        self.work.append((cs, ip))

    def run(self):
        self.add(self.image.cs, self.image.ip, "function")
        while True:
            while self.work:
                self.trace(*self.work.pop())
            if not self.resolve_deferred():
                break
        for cs, ip, insn, _disp, var in self.deferred:
            self.indirect.append((cs, ip, f"{insn.mnemonic} {insn.op_str} (index from cs:[{var:04x}], values unknown)"))
        # Code found after a table was sized may write further values to its settings variable.
        final = self.variable_values()
        self.grown_vars = sorted(var for var, used in self.resolved_vars.items() if final.get(var) != used)

    def trace(self, cs, ip):
        recent = collections.deque(maxlen=8)
        while True:
            lin = self.image.linear(cs, ip)
            if lin in self.code:
                return
            if lin >= len(self.image.data):
                self.outside.append((cs, ip))
                return
            if lin in self.owner:
                self.overlaps.append((cs, ip))
                return
            chunk = bytes(self.image.data[lin:lin + 16])
            insn = next(self.md.disasm(chunk, ip), None)
            if insn is None:
                self.outside.append((cs, ip))
                return
            self.code[lin] = (cs, ip, insn)
            for b in range(lin, lin + insn.size):
                self.owner[b] = lin
            nxt = (ip + insn.size) & 0xFFFF
            if not self.flow(cs, ip, insn, lin, recent):
                return
            recent.append(insn)
            ip = nxt

    def flow(self, cs, ip, insn, lin, recent):
        """Records the effects of one instruction and returns False when execution cannot fall through."""
        m = insn.mnemonic
        ops = insn.operands
        self.note_cs_store(cs, ip, insn)
        if m in PORT_INSTRUCTIONS:
            port = next((o.imm for o in ops if o.type == x86.X86_OP_IMM), None)
            self.ports["0x%02x" % port if port is not None else "dx"] += 1
        if m == "int":
            num = ops[0].imm
            ah = self.last_ah(recent)
            self.interrupts["int 0x%02x" % num + (" ah=0x%02x" % ah if ah is not None and num in (0x10, 0x16, 0x21, 0x33) else "")] += 1
            if num == 0x21 and ah == 0x25:
                dx = self.last_imm(recent, x86.X86_REG_DX)
                if dx is not None:
                    self.add(cs, dx, "handler")
            if num == 0x20 or (num == 0x21 and ah == 0x4C):
                return False
            return True
        if m in ("ret", "retf", "iret", "hlt"):
            return False
        if m in COND_JUMPS or m in ("jmp", "call", "lcall", "ljmp"):
            op = ops[0]
            if op.type == x86.X86_OP_IMM and m in ("jmp", "call") and insn.size in (2, 3) or op.type == x86.X86_OP_IMM and m in COND_JUMPS:
                self.add(cs, op.imm, "function" if m == "call" else None)
            elif m in ("ljmp", "lcall") or (insn.bytes[0] in (0x9A, 0xEA)):
                off = insn.bytes[1] | (insn.bytes[2] << 8)
                seg_lin = lin + 3
                seg = self.segword(seg_lin)
                if seg_lin not in self.reloc_words:
                    self.indirect.append((cs, ip, insn.mnemonic + " " + insn.op_str + " (segment not relocated)"))
                else:
                    self.add(seg, off, "function" if insn.bytes[0] == 0x9A else None)
            elif not self.jump_table(cs, ip, insn, recent):
                self.indirect.append((cs, ip, insn.mnemonic + " " + insn.op_str))
            return m not in ("jmp", "ljmp")
        return True

    def jump_table(self, cs, ip, insn, recent):
        """Resolves jmp/call word ptr cs:[reg + table] and returns False when the pattern does not apply."""
        op = insn.operands[0]
        if insn.mnemonic not in ("jmp", "call") or op.type != x86.X86_OP_MEM or op.size != 2:
            return False
        mem = op.mem
        if mem.segment != x86.X86_REG_CS or mem.base == 0 or mem.index != 0:
            return False
        disp = mem.disp & 0xFFFF
        limit = self.table_bound(recent, mem.base)
        if limit is not None:
            self.resolve_table(cs, ip, insn, disp, [2 * i for i in range(limit)], "bounded by cmp")
            return True
        source = self.index_source(recent, mem.base)
        if source is not None:
            self.deferred.append((cs, ip, insn, disp, source))
            return True
        return False

    def resolve_table(self, cs, ip, insn, disp, indexes, how):
        base = self.image.linear(cs, disp)
        for index in indexes:
            pos = self.image.linear(cs, disp + index)
            self.table_bytes.update((pos, pos + 1))
            self.add(cs, self.segword(pos), "function" if insn.mnemonic == "call" else None)
        self.tables.append((cs, ip, base, len(indexes), how))

    @staticmethod
    def index_source(recent, index_reg):
        """Returns the CS-relative variable the table index was loaded from, when the jump directly follows that load."""
        if not recent:
            return None
        load = recent[-1]
        ops = load.operands
        if load.mnemonic == "mov" and len(ops) == 2 and ops[0].type == x86.X86_OP_REG and ops[0].reg == index_reg \
                and ops[1].type == x86.X86_OP_MEM and ops[1].mem.segment == x86.X86_REG_CS \
                and ops[1].mem.base == 0 and ops[1].mem.index == 0 and ops[1].size == 2:
            return ops[1].mem.disp & 0xFFFF
        return None

    def variable_values(self):
        """Maps each CS-relative word variable to the set of values written to it, or to None when a write is not an immediate."""
        values = {}
        for cs, ip, insn in self.code.values():
            ops = insn.operands
            if not ops or ops[0].type != x86.X86_OP_MEM or ops[0].mem.segment != x86.X86_REG_CS:
                continue
            if insn.mnemonic in ("cmp", "test", "push", "jmp", "call", "lcall", "ljmp"):
                continue
            mem = ops[0].mem
            if mem.base != 0 or mem.index != 0:
                continue
            for var in (mem.disp & 0xFFFF, (mem.disp - 1) & 0xFFFF):
                if var == (mem.disp & 0xFFFF) and insn.mnemonic == "mov" and ops[0].size == 2 and ops[1].type == x86.X86_OP_IMM:
                    current = values.get(var, set())
                    if current is not None:
                        current.add(ops[1].imm & 0xFFFF)
                        values[var] = current
                else:
                    values[var] = None
        return values

    def resolve_deferred(self):
        """Sizes tables indexed by a settings variable from the values that variable can hold, and returns True when that found new code."""
        values = self.variable_values()
        still = []
        found = False
        for cs, ip, insn, disp, var in self.deferred:
            known = values.get(var)
            if known and all(v % 2 == 0 and v < 512 for v in known):
                self.resolve_table(cs, ip, insn, disp, sorted(known), f"indexed by cs:[{var:04x}] in {sorted(known)}")
                self.resolved_vars[var] = set(known)
                found = True
            else:
                still.append((cs, ip, insn, disp, var))
        self.deferred = still
        return found

    @staticmethod
    def table_bound(recent, index_reg):
        """Derives the entry count from a preceding cmp and its branch, or returns None when there is none.

        A cmp that comes before the index is doubled compares an entry number, and a cmp after the doubling
        compares a byte offset into the table. The branch after the cmp decides whether the bound is inclusive.
        """
        doubled_after_cmp = False
        following = None
        for insn in reversed(recent):
            ops = insn.operands
            if insn.mnemonic in ("shl", "add") and len(ops) == 2 and ops[0].type == x86.X86_OP_REG and ops[0].reg == index_reg:
                doubled_after_cmp = True
            if insn.mnemonic == "cmp" and len(ops) == 2 and ops[1].type == x86.X86_OP_IMM:
                branch = following.mnemonic if following is not None else ""
                if branch not in ("ja", "jae", "jb", "jbe", "jg", "jge", "jl", "jle"):
                    return None
                imm = ops[1].imm & 0xFFFF
                inclusive = branch in ("ja", "jg", "jbe", "jle")
                values = imm + 1 if inclusive else imm
                count = values if doubled_after_cmp else (values + 1) // 2
                return count if 0 < count <= 512 else None
            following = insn
        return None

    def note_cs_store(self, cs, ip, insn):
        if not insn.operands:
            return
        dst = insn.operands[0]
        writes = insn.mnemonic not in ("cmp", "test", "push", "jmp", "call", "lcall", "ljmp")
        if writes and dst.type == x86.X86_OP_MEM and dst.mem.segment == x86.X86_REG_CS:
            target = None
            if dst.mem.base == 0 and dst.mem.index == 0:
                target = self.image.linear(cs, dst.mem.disp)
            self.cs_stores.append((cs, ip, insn.mnemonic + " " + insn.op_str, target))

    @staticmethod
    def last_ah(recent):
        for insn in reversed(recent):
            if insn.mnemonic == "mov" and len(insn.operands) == 2 and insn.operands[1].type == x86.X86_OP_IMM:
                reg = insn.operands[0].reg if insn.operands[0].type == x86.X86_OP_REG else None
                if reg == x86.X86_REG_AH:
                    return insn.operands[1].imm & 0xFF
                if reg == x86.X86_REG_AX:
                    return (insn.operands[1].imm >> 8) & 0xFF
        return None

    @staticmethod
    def last_imm(recent, register):
        for insn in reversed(recent):
            if insn.mnemonic == "mov" and len(insn.operands) == 2 and insn.operands[0].type == x86.X86_OP_REG \
                    and insn.operands[0].reg == register and insn.operands[1].type == x86.X86_OP_IMM:
                return insn.operands[1].imm & 0xFFFF
        return None

    def report(self, name):
        covered = len(self.owner)
        total = len(self.image.data)
        lines = [
            f"Program: {name} ({self.image.packed_with or 'not packed'}), {total} bytes, {len(self.image.relocs)} relocations",
            f"Entry {self.image.cs:04x}:{self.image.ip:04x}. Reachable code: {len(self.code)} instructions, {covered} bytes ({100 * covered / total:.1f} % of the image).",
            f"Functions found through direct calls: {len(self.functions)}. Interrupt handlers installed through INT 21h AH=25h: {len(self.int_handlers)}.",
            f"Control transfers with a run-time target: {len(self.indirect)}.",
        ]
        for cs, ip, text in self.indirect:
            lines.append(f"  {cs:04x}:{ip:04x}  {text}")
        bounded = sum(1 for t in self.tables if t[4] == "bounded by cmp")
        lines.append(f"Jump tables resolved: {len(self.tables)} ({bounded} bounded by a cmp, {len(self.tables) - bounded} indexed by a settings variable).")
        for var, n in collections.Counter(t[4] for t in self.tables if t[4] != "bounded by cmp").items():
            lines.append(f"  {n} tables {var}")
        lines.append(f"Settings variables whose value set changed after their tables were sized: {len(self.grown_vars)}.")
        clash = sorted(b for b in self.table_bytes if b in self.owner)
        lines.append(f"Table bytes that were also decoded as code: {len(clash)}.")
        lines.append(f"Instructions that overlap other decoded code: {len(self.overlaps)}. Targets outside the image: {len(self.outside)}.")
        lines.append("Interrupt services used:")
        for key, n in sorted(self.interrupts.items()):
            lines.append(f"  {key}: {n}")
        lines.append("Hardware ports accessed (immediate port numbers, or dx when the port is in a register):")
        for key, n in sorted(self.ports.items()):
            lines.append(f"  {key}: {n}")
        code_hits = [s for s in self.cs_stores if s[3] is not None and s[3] in self.owner]
        lines.append(f"Stores through CS: {len(self.cs_stores)}, of which {len(code_hits)} hit decoded code (self-modifying code).")
        for cs, ip, text, _target in code_hits:
            lines.append(f"  {cs:04x}:{ip:04x}  {text}")
        return "\n".join(lines)

    def listing(self):
        out = []
        for lin in sorted(self.code):
            cs, ip, insn = self.code[lin]
            mark = ">" if (cs, ip) in self.functions or (cs, ip) in self.int_handlers else " "
            out.append(f"{mark}{cs:04x}:{ip:04x}  {insn.bytes.hex():<14} {insn.mnemonic} {insn.op_str}")
        return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program")
    parser.add_argument("--listing", help="write a disassembly of all reachable code to this file")
    args = parser.parse_args()
    image = mz.load(args.program)
    disc = Discovery(image)
    disc.run()
    print(disc.report(os.path.basename(args.program)))
    if args.listing:
        with open(args.listing, "w", encoding="ascii") as f:
            f.write(disc.listing())


if __name__ == "__main__":
    main()
