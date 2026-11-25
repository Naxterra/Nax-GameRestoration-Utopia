// Writes uncompressed PNG pictures (screenshots and sprites), enlarged by a whole factor.
#pragma once

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace nax {

// With alpha the picture keeps its transparency (the top byte of each pixel), otherwise it is stored as RGB.
inline bool write_png(const std::filesystem::path& path, const uint32_t* rgba, unsigned w, unsigned h, unsigned scale, bool alpha = false) {
    auto crc = [](const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
        for (size_t i = 0; i < n; i++) {
            c ^= p[i];
            for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
        }
        return c;
    };
    const unsigned sw = w * scale, sh = h * scale;
    std::vector<uint8_t> raw;
    raw.reserve(size_t(sh) * (sw * (alpha ? 4 : 3) + 1));
    for (unsigned y = 0; y < sh; y++) {
        raw.push_back(0);
        for (unsigned x = 0; x < sw; x++) {
            const uint32_t p = rgba[(y / scale) * w + x / scale];
            raw.push_back(uint8_t(p));
            raw.push_back(uint8_t(p >> 8));
            raw.push_back(uint8_t(p >> 16));
            if (alpha) raw.push_back(uint8_t(p >> 24));
        }
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t pos = 0; pos < raw.size() || pos == 0;) {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        const bool last = pos + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(uint8_t(n));
        z.push_back(uint8_t(n >> 8));
        z.push_back(uint8_t(~n));
        z.push_back(uint8_t(~n >> 8));
        z.insert(z.end(), raw.begin() + long(pos), raw.begin() + long(pos + n));
        pos += n;
        if (last) break;
    }
    const uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8) z.push_back(uint8_t(adler >> s));
    const std::filesystem::path temporary = path.wstring() + L".part";
    FILE* f = _wfopen(temporary.c_str(), L"wbN");
    if (!f) return false;
    auto be32 = [](uint8_t* p, uint32_t v) { p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16); p[2] = uint8_t(v >> 8); p[3] = uint8_t(v); };
    bool ok = true;
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        uint8_t len[4];
        be32(len, uint32_t(data.size()));
        ok = ok && std::fwrite(len, 1, 4, f) == 4;
        std::vector<uint8_t> body(type, type + 4);
        body.insert(body.end(), data.begin(), data.end());
        ok = ok && std::fwrite(body.data(), 1, body.size(), f) == body.size();
        uint8_t c[4];
        be32(c, ~crc(body.data(), body.size()));
        ok = ok && std::fwrite(c, 1, 4, f) == 4;
    };
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    ok = ok && std::fwrite(signature, 1, 8, f) == 8;
    std::vector<uint8_t> ihdr(13);
    be32(&ihdr[0], sw);
    be32(&ihdr[4], sh);
    ihdr[8] = 8;
    ihdr[9] = alpha ? 6 : 2;
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    ok = ok && std::ferror(f) == 0 && std::fflush(f) == 0;
    ok = std::fclose(f) == 0 && ok;
    if (!ok || !MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
        return false;
    }
    return true;
}

// Reads an uncompressed 8-bit RGBA PNG written by write_png(..., alpha=true).
inline bool read_uncompressed_png(const std::filesystem::path& path, unsigned* w, unsigned* h, std::vector<uint32_t>* rgba) {
    FILE* f = _wfopen(path.c_str(), L"rbN");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    if (size < 33 || size > 4 * 1024 * 1024) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    const size_t nbytes = static_cast<size_t>(size);
    std::vector<uint8_t> bytes(nbytes);
    const bool ok_read = std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    std::fclose(f);
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    if (!ok_read || std::memcmp(bytes.data(), sig, 8) != 0) return false;
    auto rd32 = [&](size_t off) {
        return (uint32_t(bytes[off]) << 24) | (uint32_t(bytes[off + 1]) << 16) | (uint32_t(bytes[off + 2]) << 8) | bytes[off + 3];
    };
    unsigned width = 0, height = 0;
    std::vector<uint8_t> idat;
    for (size_t pos = 8; pos + 12 <= bytes.size();) {
        const uint32_t len = rd32(pos);
        if (size_t(len) > bytes.size() - pos - 12) return false;
        const uint8_t* type = bytes.data() + pos + 4;
        const uint8_t* data = bytes.data() + pos + 8;
        if (std::memcmp(type, "IHDR", 4) == 0 && len == 13) {
            width = rd32(pos + 8);
            height = rd32(pos + 12);
            if (width == 0 || height == 0 || width > 512 || height > 512 || data[8] != 8 || data[9] != 6 || data[10] != 0 || data[11] != 0 ||
                data[12] != 0)
                return false;
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), data, data + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + size_t(len);
    }
    if (!width || !height || idat.size() < 7) return false;
    const size_t expected = size_t(height) * (size_t(width) * 4 + 1);
    std::vector<uint8_t> raw;
    raw.reserve(expected);
    size_t zpos = 2;  // skip 2-byte zlib header
    while (zpos + 5 <= idat.size()) {
        const uint8_t hdr = idat[zpos++];
        if ((hdr & 0x06) != 0) return false;  // only uncompressed stored blocks
        const uint16_t len = uint16_t(idat[zpos]) | (uint16_t(idat[zpos + 1]) << 8);
        const uint16_t nlen = uint16_t(idat[zpos + 2]) | (uint16_t(idat[zpos + 3]) << 8);
        zpos += 4;
        if (uint16_t(len ^ nlen) != 0xFFFFu || zpos + len > idat.size()) return false;
        raw.insert(raw.end(), idat.begin() + long(zpos), idat.begin() + long(zpos + len));
        zpos += len;
        if (hdr & 1) break;
    }
    if (raw.size() != expected) return false;
    rgba->resize(size_t(width) * size_t(height));
    for (unsigned y = 0; y < height; y++) {
        const uint8_t* row = raw.data() + size_t(y) * (size_t(width) * 4 + 1);
        if (row[0] != 0) return false;  // only filter type 0
        for (unsigned x = 0; x < width; x++) {
            const uint8_t* p = row + 1 + size_t(x) * 4;
            (*rgba)[size_t(y) * width + x] = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
        }
    }
    *w = width;
    *h = height;
    return true;
}

}  // namespace nax
