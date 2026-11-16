#include "hd_sprites.h"

#include <cstdio>

#include "machine.h"
#include "png.h"

namespace nax {

namespace {

constexpr uint32_t kVideo = 0xA0000;
constexpr int kPixels = 320 * 200;
// The 64-bit FNV-1a hash that identifies each sprite.
constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x100000001b3ull;

uint8_t nibble(const Machine& m, uint32_t row, int x) {
    const uint8_t byte = m.mem[(row + uint32_t(x / 2)) & 0xFFFFF];
    return (x & 1) ? (byte & 0x0F) : (byte >> 4);
}

}  // namespace

HdSprites::HdSprites() : marks_(0x100000) {}

void HdSprites::set_folder(const std::filesystem::path& folder) {
    std::lock_guard<std::mutex> lock(mutex_);
    folder_ = folder;
    std::error_code ec;
    std::filesystem::create_directories(folder / L"original", ec);
    std::filesystem::create_directories(folder / L"upscaled", ec);
}

std::filesystem::path HdSprites::original_file(uint64_t key) const {
    wchar_t name[32];
    swprintf(name, 32, L"%016llx.png", static_cast<unsigned long long>(key));
    return folder_ / L"original" / name;
}

std::filesystem::path HdSprites::upscaled_file(uint64_t key) const {
    wchar_t name[32];
    swprintf(name, 32, L"%016llx.png", static_cast<unsigned long long>(key));
    return folder_ / L"upscaled" / name;
}

void HdSprites::draw(const Machine& m, uint32_t data, int width, int rows, uint32_t dest, uint8_t base) {
    if (width <= 0 || width > 256 || rows <= 0 || rows > 256) return;
    const uint32_t stride = uint32_t(width / 2);
    // The key covers the pixels and the colours they are drawn in, so a sprite shown with another palette
    // (another planet, for example) is a sprite of its own.
    uint64_t id = kFnvBasis;
    auto mix = [&id](uint8_t v) { id = (id ^ v) * kFnvPrime; };
    for (uint32_t i = 0; i < stride * uint32_t(rows); i++) mix(m.mem[(data + i) & 0xFFFFF]);
    mix(uint8_t(width));
    for (int c = 1; c < 16; c++)
        for (int k = 0; k < 3; k++) mix(m.dac[uint8_t(base + c)][k]);

    uint32_t slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_key_.find(id);
        if (it != by_key_.end()) {
            slot = it->second;
        } else {
            slot = uint32_t(slots_.size());
            State state = State::missing;
            if (!folder_.empty()) {
                std::error_code ec;
                const auto original = original_file(id);
                if (!std::filesystem::exists(original, ec)) {
                    // Stored top row first, as pictures are, with transparent pixels where colour 0 is.
                    std::vector<uint32_t> rgba(size_t(width) * size_t(rows));
                    for (int y = 0; y < rows; y++) {
                        const uint32_t row = data + uint32_t(rows - 1 - y) * stride;
                        for (int x = 0; x < width; x++) {
                            const uint8_t c = nibble(m, row, x);
                            const uint8_t* d = m.dac[uint8_t(base + c)];
                            auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
                            rgba[size_t(y) * size_t(width) + size_t(x)] =
                                c ? (expand(d[0]) | (expand(d[1]) << 8) | (expand(d[2]) << 16) | 0xFF000000u) : 0;
                        }
                    }
                    write_png(original, rgba.data(), unsigned(width), unsigned(rows), 1, true);
                }
                if (std::filesystem::exists(upscaled_file(id), ec)) state = State::pending;
            }
            slots_.push_back({id, width, rows, state, 0});
            by_key_.emplace(id, slot);
        }
    }

    for (int r = 0; r < rows; r++) {
        const uint32_t row = data + uint32_t(r) * stride;
        const uint32_t line = dest - uint32_t(r) * 320;
        for (int x = 0; x < width; x++) {
            const uint8_t c = nibble(m, row, x);
            if (!c) continue;
            Mark& mark = marks_[(line + uint32_t(x)) & 0xFFFFF];
            mark.slot = slot + 1;
            mark.x = uint8_t(x);
            mark.y = uint8_t(rows - 1 - r);
            mark.colour = uint8_t(base + c);
        }
    }
}

void HdSprites::frame(const Machine& m, uint32_t* refs) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kPixels; i++) {
        refs[i] = kNone;
        const uint32_t from = m.origin[kVideo + uint32_t(i)];
        if (!from) continue;
        const Mark& mark = marks_[(from - 1) & 0xFFFFF];
        if (!mark.slot || mark.slot > slots_.size() || m.mem[kVideo + uint32_t(i)] != mark.colour) continue;
        const Slot& s = slots_[mark.slot - 1];
        if (s.state != State::placed) continue;
        refs[i] = ((s.atlas & 0xFFFF) + mark.x) | (((s.atlas >> 16) + mark.y) << 16);
    }
}

std::vector<HdSprites::Pending> HdSprites::take_pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Pending> out;
    if (folder_.empty()) return out;
    // Pictures made while the game runs are picked up a moment later.
    const auto now = std::chrono::steady_clock::now();
    const bool scan = now >= next_scan_;
    if (scan) next_scan_ = now + std::chrono::seconds(2);
    for (uint32_t i = 0; i < slots_.size(); i++) {
        Slot& s = slots_[i];
        if (s.state == State::missing && scan) {
            std::error_code ec;
            if (std::filesystem::exists(upscaled_file(s.key), ec)) s.state = State::pending;
        }
        if (s.state == State::pending) out.push_back({i, upscaled_file(s.key), s.width, s.rows});
    }
    return out;
}

void HdSprites::placed(uint32_t slot, uint16_t x, uint16_t y) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slots_.size()) return;
    slots_[slot].state = State::placed;
    slots_[slot].atlas = uint32_t(x) | (uint32_t(y) << 16);
}

void HdSprites::failed(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < slots_.size()) slots_[slot].state = State::failed;
}

size_t HdSprites::sprite_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return slots_.size();
}

}  // namespace nax
