"""Loads a DOS MZ executable into a load-relative memory image.

Segment values in the returned image are relative to the load segment, so a program loaded at
segment 0 can be analysed without knowing where DOS would have placed it. Executables compressed
with Microsoft EXEPACK are unpacked transparently, including their packed relocation table.
"""
import struct
from dataclasses import dataclass, field


class MzError(Exception):
    pass


@dataclass
class MzImage:
    data: bytearray
    relocs: list = field(default_factory=list)  # (segment, offset) pairs of words that hold a segment value
    cs: int = 0
    ip: int = 0
    ss: int = 0
    sp: int = 0
    min_extra_paragraphs: int = 0
    packed_with: str = ""

    def linear(self, seg, off):
        return (seg * 16 + off) & 0xFFFFF

    def reloc_linear_set(self):
        return {self.linear(s, o) for s, o in self.relocs}


EXEPACK_SIGNATURE = b"RB"
EXEPACK_ERROR_TEXT = b"Packed file is corrupt"


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] not in (b"MZ", b"ZM"):
        raise MzError(f"{path} is not an MZ executable")
    if len(raw) < 28:
        raise MzError(f"{path} is shorter than an MZ header")
    (last_page, pages, nrel, hdr_par, min_alloc, _max_alloc, ss, sp, _checksum, ip, cs, rel_off) = struct.unpack_from("<12H", raw, 2)
    size = pages * 512 - ((512 - last_page) if last_page else 0)
    if size > len(raw) or hdr_par * 16 > size:
        raise MzError(f"{path} is truncated or its header size is larger than the program")
    if rel_off + 4 * nrel > len(raw):
        raise MzError(f"{path} has a relocation table that runs past the end of the file")
    body = bytearray(raw[hdr_par * 16:size])
    relocs = [struct.unpack_from("<HH", raw, rel_off + 4 * i)[::-1] for i in range(nrel)]
    image = MzImage(body, relocs, cs, ip, ss, sp, min_alloc)
    stub = cs * 16
    if ip in (0x10, 0x12) and body[stub + ip - 2:stub + ip] == EXEPACK_SIGNATURE:
        return _unpack_exepack(image, stub)
    return image


def _unpack_exepack(image, stub):
    body = image.data
    if stub + 16 > len(body):
        raise MzError("EXEPACK header runs past the end of the program")
    fields = struct.unpack_from("<8H", body, stub)
    real_ip, real_cs, _mem_start, exepack_size, real_sp, real_ss, dest_len, skip_len = fields
    if image.ip == 0x10:
        # The older header variant has no skip_len field and keeps the signature at offset 14.
        skip_len = 1
    compressed_len = stub - (skip_len - 1) * 16
    if compressed_len < 0 or compressed_len > len(body):
        raise MzError("EXEPACK header describes an impossible compressed length")
    dest_size = dest_len * 16
    buf = bytearray(max(dest_size, compressed_len))
    buf[:compressed_len] = body[:compressed_len]

    src = compressed_len
    for _ in range(16):
        if src > 0 and buf[src - 1] == 0xFF:
            src -= 1
    dst = dest_size
    # Output is written downwards from the end of the buffer while input is read downwards below it.
    # Every write must stay at or above the input that is still unread, or it would destroy that input.
    while True:
        if src < 3:
            raise MzError("EXEPACK stream ends before its final command")
        command = buf[src - 1]
        length = buf[src - 3] | (buf[src - 2] << 8)
        src -= 3
        kind = command & 0xFE
        if kind == 0xB0:
            if src < 1:
                raise MzError("EXEPACK fill command has no fill byte")
            fill = buf[src - 1]
            src -= 1
            if dst - length < src:
                raise MzError("EXEPACK fill would overwrite compressed data that is not yet read")
            buf[dst - length:dst] = bytes([fill]) * length
            dst -= length
        elif kind == 0xB2:
            if src - length < 0 or dst < src:
                raise MzError("EXEPACK copy runs past the start of the data or over unread input")
            # Copying backwards byte by byte keeps overlapping in-place moves correct.
            for _ in range(length):
                src -= 1
                dst -= 1
                buf[dst] = buf[src]
        else:
            raise MzError(f"unknown EXEPACK command 0x{command:02x}")
        if command & 1:
            break

    marker = body.find(EXEPACK_ERROR_TEXT, stub, stub + exepack_size)
    if marker < 0:
        raise MzError("EXEPACK relocation table not found")
    pos = marker + len(EXEPACK_ERROR_TEXT)
    end = min(stub + exepack_size, len(body))
    relocs = []
    for group in range(16):
        if pos + 2 > end:
            raise MzError("EXEPACK relocation table runs past the packer stub")
        (count,) = struct.unpack_from("<H", body, pos)
        pos += 2
        if pos + 2 * count > end:
            raise MzError("EXEPACK relocation table runs past the packer stub")
        for _ in range(count):
            (off,) = struct.unpack_from("<H", body, pos)
            pos += 2
            relocs.append((group * 0x1000, off))
    # The packed program asked DOS for its own file size plus min_alloc. Whatever of that the unpacked
    # image does not fill is the memory the original program needs beyond its image.
    packed_paragraphs = (len(body) + 15) // 16 + image.min_extra_paragraphs
    min_extra = max(0, packed_paragraphs - dest_len)
    return MzImage(buf[:dest_size], relocs, real_cs, real_ip, real_ss, real_sp, min_extra, "EXEPACK")
