"""Finds the reachable 16-bit code of an MZ program by recursive descent and reports what a translator must handle.

Usage: python -I discover.py <program> [--listing <file>]

The report covers code coverage, control transfers whose target is only known at run time, DOS and BIOS
interrupt services, hardware port access and stores through the CS segment (data kept inside code, or
code that modifies itself). Segment values are relative to the load segment, as produced by mz.load.

Register values are only tracked along one straight-line path. A call or an interrupt ends that knowledge,
because the routine it reaches may change any register.
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
# Only unsigned branches bound a table index from above on the path that falls through to the table jump.
# Signed branches still let a negative index through, which reads far outside the table.
UPPER_BOUND_BRANCHES = {"ja": True, "jae": False}  # branch -> whether the compared value itself is allowed
BARRIERS = {"call", "lcall", "int", "into"}
READ_ONLY_DESTINATIONS = {"cmp", "test", "push", "jmp", "call", "lcall", "ljmp"}
EXCHANGES = {"xchg", "xadd", "cmpxchg"}
MAX_TABLE_ENTRIES = 256

_FAMILIES = {}
for _wide, _parts in {"ax": ("al", "ah", "eax"), "bx": ("bl", "bh", "ebx"), "cx": ("cl", "ch", "ecx"),
                      "dx": ("dl", "dh", "edx"), "si": ("esi", "sil"), "di": ("edi", "dil"),
                      "bp": ("ebp", "bpl"), "sp": ("esp", "spl")}.items():
    _FAMILIES[_wide] = _wide
    for _part in _parts:
        _FAMILIES[_part] = _wide


def family(name):
    """Maps a register name to the 16-bit register it belongs to, so that a write to bl counts as a write to bx."""
    return _FAMILIES.get(name, name)


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
        self.alias_risks = []  # stores that might write a settings variable without naming its address
        # Instructions between "mov ah, 4Ch" and its "int 21h". Reaching one of them by another path
        # means AH is unknown there, so the code after that int must be explored after all.
        self.exit_chain = {}
        # Every address that some transfer reaches. A deduction drawn from a straight-line sequence only holds
        # when no other path enters that sequence after its first instruction.
        self.targets = set()
        self.deductions = []  # (cs, ip, linear addresses the deduction depends on, table index or None, description)
        self.unproven = []
        self.work = []

    def segword(self, linear):
        """Returns the word at a linear address, or None when the word does not lie inside the image."""
        if linear < 0 or linear + 1 >= len(self.image.data):
            return None
        return self.image.data[linear] | (self.image.data[linear + 1] << 8)

    def add(self, cs, ip, kind=None):
        ip &= 0xFFFF
        if kind == "function":
            self.functions.add((cs, ip))
        elif kind == "handler":
            self.int_handlers.add((cs, ip))
        self.targets.add(self.image.linear(cs, ip))
        self.work.append((cs, ip))

    @staticmethod
    def position(seq, insn):
        return next(i for i, x in enumerate(seq) if x is insn)

    def dependents(self, cs, seq, start, point_lin):
        """Returns the addresses after seq[start] up to the deduction point, which no other path may enter."""
        return {self.image.linear(cs, x.address) for x in seq[start + 1:]} | {point_lin}

    def check_deductions(self):
        """Demotes every deduction whose checked sequence is also entered by another path after its first instruction."""
        for cs, ip, deps, table, text in self.deductions:
            entered = sorted(deps & self.targets)
            if not entered:
                continue
            where = ", ".join(f"{cs:04x}:{(lin - cs * 16) & 0xFFFF:04x}" for lin in entered)
            self.unproven.append((cs, ip, text))
            self.indirect.append((cs, ip, f"{text} is unproven, because {where} inside the checked sequence is also reached by a jump"))
            if table is not None:
                t = self.tables[table]
                self.tables[table] = t[:4] + (t[4] + " (unproven)",)

    def run(self):
        self.add(self.image.cs, self.image.ip, "function")
        while True:
            while self.work:
                self.trace(*self.work.pop())
            if not self.resolve_deferred():
                break
        for cs, ip, insn, _disp, var, _deps in self.deferred:
            self.indirect.append((cs, ip, f"{insn.mnemonic} {insn.op_str} (index from cs:[{var:04x}], values unknown)"))
        # Code found after a table was sized may write further values to its settings variable.
        final = self.variable_values()
        self.grown_vars = sorted((cs, var) for (cs, var), used in self.resolved_vars.items() if self.known_values(final, var, cs) != used)
        self.alias_risks = self.find_alias_risks()
        self.check_deductions()

    def trace(self, cs, ip):
        recent = collections.deque(maxlen=16)
        while True:
            lin = self.image.linear(cs, ip)
            if lin in self.code:
                resume = self.exit_chain.pop(lin, None)
                if resume is not None:
                    self.add(*resume)
                return
            if lin >= len(self.image.data):
                self.outside.append((cs, ip))
                return
            chunk = bytes(self.image.data[lin:lin + 16])
            insn = next(self.md.disasm(chunk, ip), None)
            if insn is None:
                self.outside.append((cs, ip))
                return
            if any(b in self.owner for b in range(lin, lin + insn.size)):
                self.overlaps.append((cs, ip))
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
            return self.interrupt(cs, ip, insn, lin, recent)
        if m in ("ret", "retf", "iret", "hlt"):
            return False
        if m in COND_JUMPS or m in ("jmp", "call", "lcall", "ljmp"):
            op = ops[0]
            if m in ("ljmp", "lcall") and len(ops) == 2 and all(o.type == x86.X86_OP_IMM for o in ops):
                # Direct far transfer: the segment word is the last word of the instruction, after any prefix.
                seg_lin = lin + insn.size - 2
                if seg_lin not in self.reloc_words:
                    self.indirect.append((cs, ip, insn.mnemonic + " " + insn.op_str + " (segment not relocated)"))
                else:
                    self.add(ops[0].imm & 0xFFFF, ops[1].imm, "function" if m == "lcall" else None)
            elif op.type == x86.X86_OP_IMM:
                self.add(cs, op.imm, "function" if m == "call" else None)
            elif m in ("ljmp", "lcall"):
                self.indirect.append((cs, ip, insn.mnemonic + " " + insn.op_str + " (far pointer in memory)"))
            elif not self.jump_table(cs, ip, insn, recent):
                self.indirect.append((cs, ip, insn.mnemonic + " " + insn.op_str))
            return m not in ("jmp", "ljmp")
        return True

    def interrupt(self, cs, ip, insn, lin, recent):
        num = insn.operands[0].imm
        ah, ah_index = self.last_ah(recent)
        self.interrupts["int 0x%02x" % num + (" ah=0x%02x" % ah if ah is not None and num in (0x10, 0x16, 0x21, 0x33) else "")] += 1
        if num == 0x21 and ah == 0x25:
            self.install_handler(cs, ip, recent, ah_index)
        if num == 0x20:
            return False
        if num == 0x21 and ah == 0x4C:
            resume = (cs, (ip + insn.size) & 0xFFFF)
            chain = list(recent)[ah_index + 1:]
            for later in chain:
                self.exit_chain[self.image.linear(cs, later.address)] = resume
            self.exit_chain[lin] = resume
            return False
        return True

    def install_handler(self, cs, ip, recent, ah_index):
        """Queues the handler of INT 21h AH=25h, which lives at DS:DX, when both registers are known on this path.

        The deduction rests on the instructions that set AH, DS and DX, so a path entering after the first of
        them may arrive with a different service number or handler address.
        """
        dx = self.reg_value(recent, "dx")
        ds = self.segment_value(recent, "ds", cs)
        if dx is not None and dx[0] == "imm" and ds is not None:
            seq = list(recent)
            start = min(ah_index, self.position(seq, dx[2]), self.position(seq, ds[1]))
            point = self.image.linear(cs, ip)
            self.add(ds[0], dx[1], "handler")
            self.deductions.append((cs, ip, self.dependents(cs, seq, start, point), None,
                                    f"int 21h ah=25h handler {ds[0]:04x}:{dx[1]:04x}"))
        else:
            self.indirect.append((cs, ip, "int 21h ah=25h (handler address in DS:DX is not known)"))

    def jump_table(self, cs, ip, insn, recent):
        """Resolves jmp/call word ptr cs:[reg + table] and returns False when the pattern does not apply."""
        op = insn.operands[0]
        if insn.mnemonic not in ("jmp", "call") or op.type != x86.X86_OP_MEM or op.size != 2:
            return False
        mem = op.mem
        if mem.segment != x86.X86_REG_CS or mem.base == 0 or mem.index != 0:
            return False
        disp = mem.disp & 0xFFFF
        seq = list(recent)
        point = self.image.linear(cs, ip)
        bound = self.table_bound(recent, insn.reg_name(mem.base))
        if bound is not None:
            offsets, start = bound
            self.resolve_table(cs, ip, insn, disp, offsets, "bounded by cmp", self.dependents(cs, seq, start, point))
            return True
        source = self.index_source(recent, mem.base)
        if source is not None:
            self.deferred.append((cs, ip, insn, disp, source, self.dependents(cs, seq, len(seq) - 1, point)))
            return True
        return False

    def resolve_table(self, cs, ip, insn, disp, offsets, how, deps):
        base = self.image.linear(cs, disp)
        for offset in offsets:
            pos = self.image.linear(cs, (disp + offset) & 0xFFFF)
            target = self.segword(pos)
            if target is None:
                self.outside.append((cs, (disp + offset) & 0xFFFF))
                continue
            self.table_bytes.update((pos, pos + 1))
            self.add(cs, target, "function" if insn.mnemonic == "call" else None)
        self.tables.append((cs, ip, base, len(offsets), how))
        self.deductions.append((cs, ip, deps, len(self.tables) - 1, f"jump table at {cs:04x}:{ip:04x} ({how})"))

    @staticmethod
    def writes(insn):
        return {insn.reg_name(r) for r in insn.regs_access()[1]}

    @staticmethod
    def is_doubling(insn, reg):
        ops = insn.operands
        if len(ops) != 2 or ops[0].type != x86.X86_OP_REG or insn.reg_name(ops[0].reg) != reg:
            return False
        if insn.mnemonic == "shl":
            return ops[1].type == x86.X86_OP_IMM and ops[1].imm == 1
        if insn.mnemonic == "add":
            return ops[1].type == x86.X86_OP_REG and insn.reg_name(ops[1].reg) == reg
        return False

    def table_bound(self, recent, index_reg):
        """Returns (byte offsets, position of the first instruction relied on) for a table jump, or None.

        The table jump sits on the fall-through path of the branch after the cmp, so only "ja" and "jae"
        bound it from above. A cmp before the doubling compares an entry number. A cmp after the doubling
        compares a byte offset, which is only known to be even when the doubling is seen before the cmp.
        Any other write to the index register between the cmp and the jump makes the bound unusable.
        """
        seq = list(recent)
        doubled_after_cmp = False
        for i in range(len(seq) - 1, -1, -1):
            insn = seq[i]
            ops = insn.operands
            if insn.mnemonic in BARRIERS:
                return None
            if insn.mnemonic == "cmp" and len(ops) == 2 and ops[0].type == x86.X86_OP_REG \
                    and insn.reg_name(ops[0].reg) == index_reg and ops[1].type == x86.X86_OP_IMM:
                branch = seq[i + 1].mnemonic if i + 1 < len(seq) else ""
                if branch not in UPPER_BOUND_BRANCHES:
                    return None
                limit = (ops[1].imm & 0xFFFF) + (1 if UPPER_BOUND_BRANCHES[branch] else 0)
                start = i
                if doubled_after_cmp:
                    offsets = [2 * k for k in range(limit)]
                else:
                    start = self.doubled_before(seq[:i], index_reg)
                    if start is None:
                        return None
                    offsets = list(range(0, limit, 2))
                return (offsets, start) if 0 < len(offsets) <= MAX_TABLE_ENTRIES else None
            if family(index_reg) in {family(w) for w in self.writes(insn)}:
                if self.is_doubling(insn, index_reg) and not doubled_after_cmp:
                    doubled_after_cmp = True
                else:
                    return None
        return None

    def doubled_before(self, seq, reg):
        """Returns the position of the doubling that last wrote reg before the end of seq, or None."""
        for i in range(len(seq) - 1, -1, -1):
            insn = seq[i]
            if insn.mnemonic in BARRIERS:
                return None
            if family(reg) in {family(w) for w in self.writes(insn)}:
                return i if self.is_doubling(insn, reg) else None
        return None

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

    @staticmethod
    def stored_operands(insn):
        """Returns the memory operands an instruction writes.

        Capstone marks them with CS_AC_WRITE. Should a Capstone version leave the access flags empty, a memory
        destination counts as written unless the mnemonic only reads it, and the memory operand of an exchange
        counts as written in any position. This errs towards reporting a store.
        """
        stored = []
        for n, op in enumerate(insn.operands):
            if op.type != x86.X86_OP_MEM:
                continue
            if op.access & capstone.CS_AC_WRITE:
                stored.append(op)
            elif op.access == 0 and (insn.mnemonic in EXCHANGES or (n == 0 and insn.mnemonic not in READ_ONLY_DESTINATIONS)):
                stored.append(op)
        return stored

    def variable_values(self):
        """Maps the linear address of each word variable to the set of values written to it, or to None when a write is not an immediate.

        Real-mode segments overlap, so stores are keyed by linear address: a store in one segment and a variable
        in another meet when they name the same bytes. A store of n bytes at X touches the word variables X-1 to
        X+n-1, so all of them are affected.
        """
        values = {}
        for cs, ip, insn in self.code.values():
            for op in self.stored_operands(insn):
                mem = op.mem
                if mem.segment != x86.X86_REG_CS or mem.base != 0 or mem.index != 0:
                    continue
                at = self.image.linear(cs, mem.disp & 0xFFFF)
                exact = insn.mnemonic == "mov" and op.size == 2 and insn.operands[1].type == x86.X86_OP_IMM
                for var in range(at - 1, at + max(op.size, 1)):
                    var &= 0xFFFFF
                    if var == at and exact:
                        current = values.get(var, set())
                        if current is not None:
                            current.add(insn.operands[1].imm & 0xFFFF)
                            values[var] = current
                    else:
                        values[var] = None
        return values

    def known_values(self, values, var, cs=None):
        """Returns the values the word variable at cs:var can hold, including its initial value in the image, or None when unknown."""
        cs = self.image.cs if cs is None else cs
        written = values.get(self.image.linear(cs, var), set())
        if written is None:
            return None
        initial = self.segword(self.image.linear(cs, var))
        if initial is None:
            return None
        return written | {initial}

    def resolve_deferred(self):
        """Sizes tables indexed by a settings variable from the values that variable can hold, and returns True when that found new code."""
        values = self.variable_values()
        still = []
        found = False
        for cs, ip, insn, disp, var, deps in self.deferred:
            known = self.known_values(values, var, cs)
            if known and all(v % 2 == 0 and v < 2 * MAX_TABLE_ENTRIES for v in known):
                self.resolve_table(cs, ip, insn, disp, sorted(known), f"indexed by cs:[{var:04x}] in {sorted(known)}", deps)
                self.resolved_vars[(cs, var)] = set(known)
                found = True
            else:
                still.append((cs, ip, insn, disp, var, deps))
        self.deferred = still
        return found

    def find_alias_risks(self):
        """Lists stores that could write a settings variable without naming its address.

        These are stores through CS with a base or index register, and stores through DS or ES in a program
        that ever loads DS or ES from CS. The check only recognises the two common ways of loading CS into DS
        or ES (push cs / pop, and mov reg, cs / mov). An empty result is therefore no proof, and tables sized
        from settings variables always stay assumptions that the translated program checks at run time.
        """
        cs_register_stores = []
        data_stores = []
        loads_cs_into_data_segment = False
        previous = None
        for lin in sorted(self.code):
            cs, ip, insn = self.code[lin]
            ops = insn.operands
            if insn.mnemonic == "pop" and ops and ops[0].type == x86.X86_OP_REG and insn.reg_name(ops[0].reg) in ("ds", "es") \
                    and previous is not None and previous.mnemonic == "push" and previous.operands \
                    and previous.operands[0].type == x86.X86_OP_REG and previous.reg_name(previous.operands[0].reg) == "cs":
                loads_cs_into_data_segment = True
            if insn.mnemonic == "mov" and len(ops) == 2 and ops[0].type == x86.X86_OP_REG and ops[1].type == x86.X86_OP_REG \
                    and insn.reg_name(ops[0].reg) in ("ds", "es") and previous is not None and previous.mnemonic == "mov" \
                    and len(previous.operands) == 2 and previous.operands[1].type == x86.X86_OP_REG \
                    and previous.reg_name(previous.operands[1].reg) == "cs":
                loads_cs_into_data_segment = True
            for op in self.stored_operands(insn):
                seg = op.mem.segment
                if seg == x86.X86_REG_CS:
                    if op.mem.base != 0 or op.mem.index != 0:
                        cs_register_stores.append((cs, ip, insn.mnemonic + " " + insn.op_str))
                elif seg in (x86.X86_REG_DS, x86.X86_REG_ES) or (seg == 0 and op.mem.base not in (x86.X86_REG_BP, x86.X86_REG_SP)):
                    # Without an override, [bp] and [sp] addresses use SS. An explicit ds: or es: override counts in every form.
                    data_stores.append((cs, ip, insn.mnemonic + " " + insn.op_str))
            previous = insn
        return cs_register_stores + (data_stores if loads_cs_into_data_segment else [])

    def last_write(self, recent, names):
        """Returns the latest instruction on this path that writes one of the given registers, or None if unknown."""
        for insn in reversed(recent):
            if insn.mnemonic in BARRIERS:
                return None
            if self.writes(insn) & names:
                return insn
        return None

    def reg_value(self, recent, reg):
        """Returns ("imm", value, instruction) or ("cs", None, instruction) for a 16-bit register on this path, or None."""
        names = {n for n, f in _FAMILIES.items() if f == reg} | {reg}
        insn = self.last_write(recent, names)
        if insn is None or insn.mnemonic != "mov" or len(insn.operands) != 2:
            return None
        dst, src = insn.operands
        if dst.type != x86.X86_OP_REG or insn.reg_name(dst.reg) != reg:
            return None
        if src.type == x86.X86_OP_IMM:
            return ("imm", src.imm & 0xFFFF, insn)
        if src.type == x86.X86_OP_REG and insn.reg_name(src.reg) == "cs":
            return ("cs", None, insn)
        return None

    def segment_value(self, recent, seg, cs):
        """Returns (load-relative value, first instruction relied on) for DS or ES on this path, or None."""
        seq = list(recent)
        for i in range(len(seq) - 1, -1, -1):
            insn = seq[i]
            if insn.mnemonic in BARRIERS:
                return None
            if seg not in self.writes(insn):
                continue
            ops = insn.operands
            if insn.mnemonic == "pop":
                before = seq[i - 1] if i > 0 else None
                if before is not None and before.mnemonic == "push" and before.operands \
                        and before.operands[0].type == x86.X86_OP_REG and before.reg_name(before.operands[0].reg) == "cs":
                    return cs, before
                return None
            if insn.mnemonic == "mov" and len(ops) == 2 and ops[1].type == x86.X86_OP_REG:
                value = self.reg_value(seq[:i], insn.reg_name(ops[1].reg))
                if value is None:
                    return None
                if value[0] == "cs":
                    return cs, value[2]
                # An immediate is only a segment when the loader relocates it, which makes it load-relative.
                imm_lin = self.image.linear(cs, value[2].address) + value[2].size - 2
                return (value[1], value[2]) if imm_lin in self.reloc_words else None
            return None
        return None

    def note_cs_store(self, cs, ip, insn):
        if not insn.operands:
            return
        dst = insn.operands[0]
        writes = insn.mnemonic not in READ_ONLY_DESTINATIONS
        if writes and dst.type == x86.X86_OP_MEM and dst.mem.segment == x86.X86_REG_CS:
            target = None
            if dst.mem.base == 0 and dst.mem.index == 0:
                target = self.image.linear(cs, dst.mem.disp & 0xFFFF)
            self.cs_stores.append((cs, ip, insn.mnemonic + " " + insn.op_str, target))

    def last_ah(self, recent):
        """Returns (value of AH, index of the instruction that set it) on this path, or (None, None) when unknown."""
        seq = list(recent)
        for i in range(len(seq) - 1, -1, -1):
            insn = seq[i]
            if insn.mnemonic in BARRIERS:
                return None, None
            written = self.writes(insn)
            if not written & {"ah", "ax", "eax"}:
                continue
            ops = insn.operands
            if insn.mnemonic == "mov" and len(ops) == 2 and ops[0].type == x86.X86_OP_REG and ops[1].type == x86.X86_OP_IMM:
                name = insn.reg_name(ops[0].reg)
                if name == "ah":
                    return ops[1].imm & 0xFF, i
                if name == "ax":
                    return (ops[1].imm >> 8) & 0xFF, i
            return None, None
        return None, None

    def report(self, name):
        covered = len(self.owner)
        total = len(self.image.data)
        share = f"{100 * covered / total:.1f} %" if total else "no image"
        lines = [
            f"Program: {name} ({self.image.packed_with or 'not packed'}), {total} bytes, {len(self.image.relocs)} relocations",
            f"Entry {self.image.cs:04x}:{self.image.ip:04x}. Reachable code: {len(self.code)} instructions, {covered} bytes ({share} of the image).",
            f"Functions found through direct calls: {len(self.functions)}. Interrupt handlers installed through INT 21h AH=25h: {len(self.int_handlers)}.",
            f"Control transfers with a run-time target: {len(self.indirect)}.",
        ]
        for cs, ip, text in self.indirect:
            lines.append(f"  {cs:04x}:{ip:04x}  {text}")
        bounded = sum(1 for t in self.tables if t[4].startswith("bounded by cmp"))
        unproven = sum(1 for t in self.tables if t[4].endswith("(unproven)"))
        lines.append(f"Jump tables resolved: {len(self.tables)} ({bounded} bounded by a cmp, {len(self.tables) - bounded} indexed by a settings variable, "
                     f"{unproven} of them unproven because another path enters the checked sequence).")
        for var, n in collections.Counter(t[4] for t in self.tables if not t[4].startswith("bounded by cmp")).items():
            lines.append(f"  {n} tables {var}")
        if self.resolved_vars:
            if self.alias_risks:
                lines.append(f"Tables sized from settings variables are assumptions: {len(self.alias_risks)} stores could write those variables "
                             "without naming them, so the translated program must check each such index at run time.")
            else:
                lines.append("Tables sized from settings variables are assumptions: no aliasing store was detected, but the check does not cover "
                             "every way a store can reach the code segment, so the translated program must check each such index at run time.")
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
    try:
        image = mz.load(args.program)
    except mz.MzError as error:
        sys.exit(f"Cannot load {args.program}: {error}")
    disc = Discovery(image)
    disc.run()
    print(disc.report(os.path.basename(args.program)))
    if args.listing:
        with open(args.listing, "w", encoding="ascii") as f:
            f.write(disc.listing())


if __name__ == "__main__":
    main()
