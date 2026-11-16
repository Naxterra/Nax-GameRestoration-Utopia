"""Translates the reachable 16-bit code of a program into a C++ module for the native runtime.

Usage: python -I emit.py <program> <module-name> <output.cpp> [--flat] [--seeds <file>]

Every decoded instruction becomes a case of one dispatch switch with a label of its own. Direct jumps go
straight to their label, while returns, computed jumps and interrupts set CS:IP and go through the dispatch,
so the program's own stack and jump tables keep working exactly as before. Reaching an address that was not
translated stops the program with a message that names it.

With --flat the program is a raw binary loaded at offset 0 of its segment, as the sound drivers are.
A seeds file lists extra entry points, one "segment:offset" in hex per line, relative to the load segment.
A hooks file lists "segment:offset function" lines: the named C++ function, which the game provides, is called
with the machine just before the instruction at that address runs, so it can observe the program's state.
"""
import argparse
import binascii
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capstone import x86  # noqa: E402

import discover  # noqa: E402
import mz  # noqa: E402

REGS = {
    "ax": "c.a.x", "al": "c.a.l", "ah": "c.a.h", "bx": "c.b.x", "bl": "c.b.l", "bh": "c.b.h",
    "cx": "c.c.x", "cl": "c.c.l", "ch": "c.c.h", "dx": "c.d.x", "dl": "c.d.l", "dh": "c.d.h",
    "si": "c.si", "di": "c.di", "bp": "c.bp", "sp": "c.sp",
    "cs": "c.cs", "ds": "c.ds", "es": "c.es", "ss": "c.ss",
}
SEG_PREFIX = {0x2E: "c.cs", 0x36: "c.ss", 0x3E: "c.ds", 0x26: "c.es"}
CONDITIONS = {
    "jo": "c.of", "jno": "!c.of", "jb": "c.cf", "jae": "!c.cf", "je": "c.zf", "jne": "!c.zf",
    "jbe": "(c.cf || c.zf)", "ja": "(!c.cf && !c.zf)", "js": "c.sf", "jns": "!c.sf",
    "jl": "(c.sf != c.of)", "jge": "(c.sf == c.of)", "jle": "(c.zf || c.sf != c.of)", "jg": "(!c.zf && c.sf == c.of)",
    "jcxz": "(c.c.x == 0)",
}
# Slots of Cpu::org, which follows where the value in each byte register came from.
BYTE_SLOTS = {"al": [0], "cl": [1], "dl": [2], "bl": [3], "ah": [4], "ch": [5], "dh": [6], "bh": [7],
              "ax": [0, 4], "cx": [1, 5], "dx": [2, 6], "bx": [3, 7], "eax": [0, 4], "ecx": [1, 5], "edx": [2, 6], "ebx": [3, 7]}
# Instructions whose origin handling is part of their own translation.
COPIES = ("mov", "lodsb", "lodsw", "stosb", "stosw", "movsb", "movsw")
ARITH = {"add": "add", "adc": "add", "sub": "sub", "sbb": "sub", "cmp": "sub"}
LOGIC = {"and": "&", "or": "|", "xor": "^", "test": "&"}
SHIFTS = {"shl": "shl", "sal": "shl", "shr": "shr", "sar": "sar", "rol": "rol", "ror": "ror", "rcl": "rcl", "rcr": "rcr"}


class EmitError(Exception):
    pass


class Emitter:
    def __init__(self, disc, name, trace=False, hooks=None):
        self.trace = trace
        self.hooks = hooks or {}
        self.disc = disc
        self.image = disc.image
        self.name = name
        self.reloc = disc.reloc_words
        self.addrs = {(cs, ip) for cs, ip, _ in disc.code.values()}

    # Operands ------------------------------------------------------------------------------------------

    def mem_parts(self, insn, op):
        mem = op.mem
        base = insn.reg_name(mem.base) if mem.base else None
        index = insn.reg_name(mem.index) if mem.index else None
        if mem.segment:
            seg = REGS[insn.reg_name(mem.segment)]
        else:
            seg = "c.ss" if base == "bp" else "c.ds"
        terms = [REGS[r] for r in (base, index) if r]
        if terms:
            if mem.disp:
                terms.append(str(mem.disp))
            off = "(uint16_t)(" + " + ".join(terms) + ")"
        else:
            off = "0x%04x" % (mem.disp & 0xFFFF)
        return seg, off

    def operand(self, insn, op, lin, temps):
        """Returns (read expression, write function, size in bytes) for one operand."""
        if op.type == x86.X86_OP_REG:
            name = insn.reg_name(op.reg)
            if name not in REGS:
                raise EmitError(f"unsupported register {name}")
            target = REGS[name]
            return target, (lambda v, t=target: f"{t} = {v};"), op.size
        if op.type == x86.X86_OP_IMM:
            value = op.imm & (0xFF if op.size == 1 else 0xFFFF)
            text = "0x%x" % value
            # A 16-bit immediate is always the last word of an instruction, so a relocation there belongs to it.
            if lin + insn.size - 2 in self.reloc and op.size == 2:
                text = "(uint16_t)(B + 0x%04x)" % value
            return text, None, op.size
        if op.type == x86.X86_OP_MEM:
            seg, off = self.mem_parts(insn, op)
            n = len(temps)
            temps.append(f"const uint16_t s{n} = {seg}; const uint16_t o{n} = {off};")
            bits = 8 if op.size == 1 else 16
            return f"m.rd{bits}(s{n}, o{n})", (lambda v, b=bits, n=n: f"m.wr{b}(s{n}, o{n}, {v});"), op.size
        raise EmitError("unknown operand type")

    def check_relocs(self, insn, lin):
        for word in range(lin, lin + insn.size - 1):
            if word in self.reloc and word != lin + insn.size - 2:
                raise EmitError(f"relocation inside instruction at {lin:05x} that is not its last immediate")

    # Control flow --------------------------------------------------------------------------------------

    def label(self, cs, ip):
        return f"L_{cs:04x}_{ip:04x}"

    def goto(self, cs, ip, here_ip, same_seg=True):
        """Jumps to a translated address, polling for interrupts on backward jumps (here_ip None never polls)."""
        if same_seg and (cs, ip) in self.addrs:
            if here_ip is not None and ip <= here_ip:
                return f"c.ip = 0x{ip:04x}; if (m.poll()) goto dispatch; goto {self.label(cs, ip)};"
            return f"goto {self.label(cs, ip)};"
        return f"c.ip = 0x{ip:04x}; goto dispatch;"

    # Instructions --------------------------------------------------------------------------------------

    def translate(self, cs, ip, insn, lin):
        self.check_relocs(insn, lin)
        m = insn.mnemonic
        ops = insn.operands
        nxt = (ip + insn.size) & 0xFFFF
        temps = []
        body = self.body(cs, ip, insn, lin, m, ops, nxt, temps)
        writer = [f"m.writer = 0x{lin:05x};"] if self.trace else []
        hook = [f"{self.hooks[(cs, ip)]}(m);"] if (cs, ip) in self.hooks else []
        return " ".join(writer + hook + temps + self.forget_origins(insn) + [body])

    def slots(self, insn, op):
        return BYTE_SLOTS.get(insn.reg_name(op.reg), []) if op.type == x86.X86_OP_REG else []

    def forget_origins(self, insn):
        """Any instruction other than a plain copy computes the registers it writes, so their origin is lost."""
        if insn.mnemonic.replace("rep ", "") in COPIES:
            return []
        written = {s for r in insn.regs_access()[1] for s in BYTE_SLOTS.get(insn.reg_name(r), [])}
        return [f"c.org[{s}] = 0;" for s in sorted(written)]

    def body(self, cs, ip, insn, lin, m, ops, nxt, temps):
        rep = m.startswith("rep ")
        if rep:
            m = m[4:]
        if m == "nop":
            return ""
        if m == "mov":
            src, _, size = self.operand(insn, ops[1], lin, temps)
            _, put, _ = self.operand(insn, ops[0], lin, temps)
            src_slots, dst_slots = self.slots(insn, ops[1]), self.slots(insn, ops[0])
            bits = 8 if size == 1 else 16
            if ops[0].type == x86.X86_OP_MEM:
                if not src_slots:
                    return put(src)
                n = len(temps) - 1
                origins = ", ".join(f"c.org[{s}]" for s in src_slots)
                return f"m.wr{bits}o(s{n}, o{n}, {src}, {origins});"
            if not dst_slots:
                return put(src)
            if ops[1].type == x86.X86_OP_MEM:
                origins = [f"m.org(s0, (uint16_t)(o0 + {k}))" for k in range(len(dst_slots))]
            elif src_slots:
                origins = [f"c.org[{s}]" for s in src_slots]
            else:
                origins = ["0"] * len(dst_slots)
            keep = " ".join(f"const uint32_t g{k} = {v};" for k, v in enumerate(origins))
            assign = " ".join(f"c.org[{s}] = g{k};" for k, s in enumerate(dst_slots))
            return f"{{ {keep} {put(src)} {assign} }}"
        if m in ARITH or m in LOGIC:
            a, put, size = self.operand(insn, ops[0], lin, temps)
            b, _, _ = self.operand(insn, ops[1], lin, temps)
            bits = 8 if size == 1 else 16
            if m in ARITH:
                carry = ", c.cf" if m in ("adc", "sbb") else ", false"
                expr = f"nax::{ARITH[m]}{bits}(c, {a}, {b}{carry})"
            else:
                expr = f"nax::logic{bits}(c, (uint{bits}_t)({a} {LOGIC[m]} {b}))"
            return f"(void){expr};" if m in ("cmp", "test") else put(expr)
        if m in ("inc", "dec", "neg", "not"):
            a, put, size = self.operand(insn, ops[0], lin, temps)
            bits = 8 if size == 1 else 16
            if m == "not":
                return put(f"(uint{bits}_t)~{a}")
            return put(f"nax::{m}{bits}(c, {a})")
        if m in SHIFTS:
            a, put, size = self.operand(insn, ops[0], lin, temps)
            count, _, _ = self.operand(insn, ops[1], lin, temps)
            bits = 8 if size == 1 else 16
            return put(f"nax::{SHIFTS[m]}{bits}(c, {a}, (unsigned)({count}))")
        if m in ("mul", "imul", "div", "idiv"):
            if len(ops) != 1:
                if m != "imul":
                    raise EmitError(f"unsupported form of {m}")
                _, put, _ = self.operand(insn, ops[0], lin, temps)
                a, _, _ = self.operand(insn, ops[-2], lin, temps)
                b, _, _ = self.operand(insn, ops[-1], lin, temps)
                return put(f"nax::imul16x(c, {a}, {b})")
            a, _, size = self.operand(insn, ops[0], lin, temps)
            bits = 8 if size == 1 else 16
            return f"nax::{m}{bits}(m, c, {a});"
        if m == "cwde" and insn.bytes[0] == 0x98:
            return "c.a.x = (uint16_t)(int16_t)(int8_t)c.a.l;"
        if m in ("cdq", "cwd") and insn.bytes[0] == 0x99:
            return "c.d.x = (c.a.x & 0x8000) ? 0xFFFF : 0;"
        if m == "xchg":
            a, put_a, _ = self.operand(insn, ops[0], lin, temps)
            b, put_b, _ = self.operand(insn, ops[1], lin, temps)
            return f"{{ const auto t = {a}; {put_a(b)} {put_b('t')} }}"
        if m == "push":
            a, _, _ = self.operand(insn, ops[0], lin, temps)
            if ops[0].type == x86.X86_OP_REG and insn.reg_name(ops[0].reg) == "sp":
                return "m.push((uint16_t)(c.sp - 2));"
            return f"m.push({a});"
        if m == "pop":
            _, put, _ = self.operand(insn, ops[0], lin, temps)
            return f"{{ const uint16_t t = m.pop(); {put('t')} }}"
        if m in ("les", "lds"):
            _, put, _ = self.operand(insn, ops[0], lin, temps)
            seg, off = self.mem_parts(insn, ops[1])
            sreg = "c.es" if m == "les" else "c.ds"
            return (f"{{ const uint16_t s = {seg}; const uint16_t o = {off}; const uint16_t lo = m.rd16(s, o); "
                    f"{sreg} = m.rd16(s, (uint16_t)(o + 2)); {put('lo')} }}")
        if m == "lea":
            _, put, _ = self.operand(insn, ops[0], lin, temps)
            _, off = self.mem_parts(insn, ops[1])
            return put(off)
        if m == "pushf":
            return "m.push(c.flags());"
        if m == "popf":
            return "c.set_flags(m.pop());"
        if m == "lahf":
            return "c.a.h = (uint8_t)c.flags();"
        if m == "sahf":
            return "c.set_flags((uint16_t)((c.flags() & 0xFF00) | c.a.h));"
        if m in ("lodsb", "lodsw", "stosb", "stosw", "movsb", "movsw"):
            return self.string_op(insn, m, rep)
        if m in ("clc", "stc", "cli", "sti", "cld", "std", "cmc"):
            return {
                "clc": "c.cf = false;", "stc": "c.cf = true;", "cmc": "c.cf = !c.cf;", "cli": "c.iflag = false;",
                "sti": "c.iflag = true;", "cld": "c.df = false;", "std": "c.df = true;",
            }[m]
        if m in ("in", "out"):
            if m == "in":
                dst, port = ops[0], ops[1]
            else:
                port, dst = ops[0], ops[1]
            p = "c.d.x" if port.type == x86.X86_OP_REG else "0x%x" % (port.imm & 0xFF)
            reg = insn.reg_name(dst.reg)
            if m == "in":
                return f"c.a.l = m.in8({p});" if reg == "al" else f"c.a.x = m.in16({p});"
            return f"m.out8({p}, c.a.l);" if reg == "al" else f"m.out16({p}, c.a.x);"
        if m in CONDITIONS:
            target = ops[0].imm & 0xFFFF
            return f"if ({CONDITIONS[m]}) {{ {self.goto(cs, target, ip)} }}" + self.fall(cs, ip, nxt)
        if m in ("loop", "loope", "loopne"):
            target = ops[0].imm & 0xFFFF
            extra = {"loop": "", "loope": " && c.zf", "loopne": " && !c.zf"}[m]
            return f"if (--c.c.x != 0{extra}) {{ {self.goto(cs, target, ip)} }}" + self.fall(cs, ip, nxt)
        if m in ("jmp", "call"):
            op = ops[0]
            push = f"m.push(0x{nxt:04x}); " if m == "call" else ""
            if op.type == x86.X86_OP_IMM:
                return push + self.goto(cs, op.imm & 0xFFFF, ip if m == "jmp" else None)
            a, _, _ = self.operand(insn, op, lin, temps)
            return f"{{ const uint16_t t = {a}; {push}c.ip = t; goto dispatch; }}"
        if m in ("ljmp", "lcall"):
            push = f"m.push(c.cs); m.push(0x{nxt:04x}); " if m == "lcall" else ""
            if len(ops) == 2 and all(o.type == x86.X86_OP_IMM for o in ops):
                seg = ops[0].imm & 0xFFFF
                seg_text = "(uint16_t)(B + 0x%04x)" % seg if lin + insn.size - 2 in self.reloc else "0x%04x" % seg
                return f"{push}c.cs = {seg_text}; c.ip = 0x{ops[1].imm & 0xFFFF:04x}; goto dispatch;"
            seg, off = self.mem_parts(insn, ops[0])
            temps.append(f"const uint16_t s0 = {seg}; const uint16_t o0 = {off};")
            return (f"{{ const uint16_t to = m.rd16(s0, o0); const uint16_t ts = m.rd16(s0, (uint16_t)(o0 + 2)); "
                    f"{push}c.cs = ts; c.ip = to; goto dispatch; }}")
        if m in ("ret", "retf"):
            extra = ops[0].imm & 0xFFFF if ops else 0
            far = " c.cs = m.pop();" if m == "retf" else ""
            add = f" c.sp = (uint16_t)(c.sp + {extra});" if extra else ""
            return f"c.ip = m.pop();{far}{add} goto dispatch;"
        if m == "iret":
            return "m.iret(); goto dispatch;"
        if m == "int":
            return f"c.ip = 0x{nxt:04x}; m.interrupt(0x{ops[0].imm & 0xFF:02x}); goto dispatch;"
        if m == "hlt":
            return f"c.ip = 0x{nxt:04x}; m.halt_cpu(); goto dispatch;"
        raise EmitError(f"unsupported instruction {insn.mnemonic} {insn.op_str}")

    def fall(self, cs, ip, nxt):
        return "" if (cs, nxt) in self.addrs else f" c.ip = 0x{nxt:04x}; goto dispatch;"

    def string_op(self, insn, m, rep):
        width = 1 if m.endswith("b") else 2
        bits = width * 8
        src_seg = SEG_PREFIX.get(insn.prefix[1], "c.ds")
        step = f"(c.df ? -{width} : {width})"
        if m.startswith("lods"):
            reg = "c.a.l" if width == 1 else "c.a.x"
            origins = "c.org[0] = m.org(%s, c.si);" % src_seg
            if width == 2:
                origins += " c.org[4] = m.org(%s, (uint16_t)(c.si + 1));" % src_seg
            one = f"{reg} = m.rd{bits}({src_seg}, c.si); {origins} c.si = (uint16_t)(c.si + {step});"
        elif m.startswith("stos"):
            reg = "c.a.l" if width == 1 else "c.a.x"
            origins = "c.org[0]" if width == 1 else "c.org[0], c.org[4]"
            one = f"m.wr{bits}o(c.es, c.di, {reg}, {origins}); c.di = (uint16_t)(c.di + {step});"
        else:
            origins = f"m.org({src_seg}, c.si)"
            if width == 2:
                origins += f", m.org({src_seg}, (uint16_t)(c.si + 1))"
            one = (f"m.wr{bits}o(c.es, c.di, m.rd{bits}({src_seg}, c.si), {origins}); "
                   f"c.si = (uint16_t)(c.si + {step}); c.di = (uint16_t)(c.di + {step});")
        if rep:
            return f"while (c.c.x != 0) {{ {one} c.c.x--; }}"
        return one

    # Module --------------------------------------------------------------------------------------------

    def module(self, source_name, source_crc, flat, signature):
        entries = sorted(self.disc.code.values(), key=lambda e: (e[0], e[1]))
        segs = sorted({cs for cs, _, _ in entries})
        out = [
            f"// Generated by tools/recomp/emit.py from {source_name}. This file is derived from the original program,",
            "// so it stays on the machine of the user who owns that program and is never committed or distributed.",
            '#include "runtime/cpu_ops.h"',
            "",
        ] + [f"void {name}(nax::Machine& m);" for name in sorted(set(self.hooks.values()))] + [
            "",
            f"static void run_{self.name}(nax::Machine& m, uint16_t B) {{",
            "    nax::Cpu& c = m.c;",
            "    for (;;) {",
            "    dispatch:",
            "        if (m.service_pending()) return;",
            "        {",
            "        const uint16_t rel = (uint16_t)(c.cs - B);",
            "        if (" + " && ".join(f"rel != 0x{s:04x}" for s in segs) + ") return;",
            "        switch (((uint32_t)rel << 16) | c.ip) {",
        ]
        for cs, ip, insn in entries:
            lin = self.image.linear(cs, ip)
            try:
                code = self.translate(cs, ip, insn, lin)
            except EmitError as error:
                raise EmitError(f"{cs:04x}:{ip:04x} {insn.mnemonic} {insn.op_str}: {error}") from None
            nxt = (ip + insn.size) & 0xFFFF
            tail = "" if (cs, nxt) in self.addrs or self.terminal(insn) else f" c.ip = 0x{nxt:04x}; goto dispatch;"
            out.append(f"        case 0x{cs:04x}{ip:04x}: {self.label(cs, ip)}: {{ {code}{tail} }} // {insn.mnemonic} {insn.op_str}")
        out += [
            "        default:",
            "            m.untranslated(B);",
            "            return;",
            "        }",
            "        }",
            "    }",
            "}",
            "",
            "static const uint16_t kSegments[] = {" + ", ".join(f"0x{s:04x}" for s in segs) + "};",
            "static const uint8_t kSignature[] = {" + (", ".join(str(b) for b in signature) or "0") + "};",
            f"static const nax::ModuleInfo kInfo = {{\"{source_name}\", run_{self.name}, kSegments, {len(segs)}, "
            f"0x{source_crc:08x}u, {len(self.image.data)}u, {'true' if flat else 'false'}, kSignature, {len(signature)}u, "
            f"0x{self.image.cs:04x}, 0x{self.image.ip:04x}, 0x{self.image.ss:04x}, 0x{self.image.sp:04x}, {self.image.min_extra_paragraphs}u}};",
            f"const nax::ModuleInfo* nax_module_{self.name}() {{ return &kInfo; }}",
            "",
        ]
        return "\n".join(out)

    @staticmethod
    def terminal(insn):
        m = insn.mnemonic
        if m in ("jmp", "ljmp", "ret", "retf", "iret", "int", "hlt", "call", "lcall"):
            return True
        return m in CONDITIONS or m.startswith("loop")


def read_seeds(path):
    seeds = []
    if path and os.path.exists(path):
        with open(path, encoding="ascii") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    seg, off = line.split(":")
                    seeds.append((int(seg, 16), int(off, 16)))
    return seeds


def read_hooks(path):
    hooks = {}
    if path and os.path.exists(path):
        with open(path, encoding="ascii") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    address, name = line.split()
                    seg, off = address.split(":")
                    if not name.isidentifier():
                        raise ValueError(f"bad hook name {name}")
                    hooks[(int(seg, 16), int(off, 16))] = name
    return hooks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program")
    parser.add_argument("name")
    parser.add_argument("output")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--seeds")
    parser.add_argument("--hooks")
    parser.add_argument("--trace", action="store_true", help="record which instruction wrote each byte, for analysis")
    args = parser.parse_args()
    with open(args.program, "rb") as f:
        raw = f.read()
    if args.flat:
        image = mz.MzImage(bytearray(raw), [], 0, 0, 0, 0)
    else:
        image = mz.load(args.program)
    disc = discover.Discovery(image)
    for seg, off in read_seeds(args.seeds):
        disc.add(seg, off, "function")
    disc.run()
    hooks = read_hooks(args.hooks)
    missing = [f"{s:04x}:{o:04x}" for s, o in hooks if (s, o) not in {(cs, ip) for cs, ip, _ in disc.code.values()}]
    if missing:
        sys.exit(f"Cannot translate {args.program}: hook addresses are not instructions: {', '.join(missing)}")
    emitter = Emitter(disc, args.name, args.trace, hooks)
    signature = raw[:64] if args.flat else b""
    try:
        text = emitter.module(os.path.basename(args.program), binascii.crc32(raw), args.flat, signature)
    except EmitError as error:
        sys.exit(f"Cannot translate {args.program}: {error}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"{args.name}: {len(disc.code)} instructions in {len({cs for cs, _, _ in disc.code.values()})} segments, "
          f"{len(disc.indirect)} run-time targets left to the dispatch")


if __name__ == "__main__":
    main()
