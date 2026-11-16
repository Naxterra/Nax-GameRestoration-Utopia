// Writes uncompressed PNG pictures (screenshots and sprites), enlarged by a whole factor.
#pragma once

#include <cstdint>
#include <cstdio>
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
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    auto be32 = [](uint8_t* p, uint32_t v) { p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16); p[2] = uint8_t(v >> 8); p[3] = uint8_t(v); };
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        uint8_t len[4];
        be32(len, uint32_t(data.size()));
        std::fwrite(len, 1, 4, f);
        std::vector<uint8_t> body(type, type + 4);
        body.insert(body.end(), data.begin(), data.end());
        std::fwrite(body.data(), 1, body.size(), f);
        uint8_t c[4];
        be32(c, ~crc(body.data(), body.size()));
        std::fwrite(c, 1, 4, f);
    };
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    std::fwrite(signature, 1, 8, f);
    std::vector<uint8_t> ihdr(13);
    be32(&ihdr[0], sw);
    be32(&ihdr[4], sh);
    ihdr[8] = 8;
    ihdr[9] = alpha ? 6 : 2;
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    std::fclose(f);
    return true;
}

}  // namespace nax
