#include "hd_sprites.h"

#include <cstdio>
#include <cstring>

#include "machine.h"
#include "png.h"

namespace nax {

namespace {

constexpr uint32_t kVideo = 0xA0000;
constexpr int kPixels = 320 * 200;
constexpr int64_t kSettleMs = 500;
constexpr size_t kMaxDeferred = 8;
// The 64-bit FNV-1a hash that identifies each sprite.
constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x100000001b3ull;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint8_t nibble(const Machine& m, uint32_t row, int x) {
    const uint8_t byte = m.mem[(row + uint32_t(x / 2)) & 0xFFFFF];
    return (x & 1) ? (byte & 0x0F) : (byte >> 4);
}

uint32_t colour_of(const Machine& m, uint8_t index) {
    auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
    const uint8_t* d = m.dac[index];
    return expand(d[0]) | (expand(d[1]) << 8) | (expand(d[2]) << 16) | 0xFF000000u;
}

// The colours of the given palette entries, as one number.
uint64_t palette_of(const Machine& m, const std::bitset<256>& colours) {
    uint64_t h = kFnvBasis;
    for (int c = 0; c < 256; c++)
        if (colours[size_t(c)])
            for (int k = 0; k < 3; k++) h = (h ^ m.dac[c][k]) * kFnvPrime;
    return h;
}

}  // namespace

HdSprites::HdSprites() : sprite_marks_(0x100000), image_marks_(0x100000) {}

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

bool HdSprites::settled(const std::bitset<256>& colours) const {
    const int64_t now = now_ms();
    for (int c = 0; c < 256; c++)
        if (colours[size_t(c)] && now - colour_changed_[c].load() < kSettleMs) return false;
    return true;
}

// Called without the lock held: the lock is taken only to look the sprite up and to add it, and the picture is
// saved in between, so neither thread waits for a file to be written.
uint32_t HdSprites::slot_for(uint64_t id, int width, int rows, const std::bitset<256>& colours, uint64_t palette,
                             const std::function<std::vector<uint32_t>()>& picture) {
    std::filesystem::path original, upscaled;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_key_.find(id);
        if (it != by_key_.end()) return it->second;
        if (!folder_.empty()) {
            original = original_file(id);
            upscaled = upscaled_file(id);
        }
    }
    State state = State::missing;
    if (!original.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(original, ec)) write_png(original, picture().data(), unsigned(width), unsigned(rows), 1, true);
        if (std::filesystem::exists(upscaled, ec)) state = State::pending;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_key_.find(id);
    if (it != by_key_.end()) return it->second;
    const uint32_t slot = uint32_t(slots_.size());
    slots_.push_back({id, width, rows, state, 0, colours, palette, true});
    by_key_.emplace(id, slot);
    return slot;
}

void HdSprites::draw(const Machine& m, uint32_t data, int width, int rows, uint32_t dest, uint8_t base) {
    if (width <= 0 || width > 256 || rows <= 0 || rows > 256) return;
    std::bitset<256> colours;
    for (int c = 1; c < 16; c++) colours.set(uint8_t(base + c));
    // While the colours fade, every step would look like another sprite, so sprites wait until they settle.
    if (!settled(colours)) return;
    const uint32_t stride = uint32_t(width / 2);
    // The key covers the pixels and the colours they are drawn in, so a sprite shown with another palette
    // (another planet, for example) is a sprite of its own.
    uint64_t id = kFnvBasis;
    auto mix = [&id](uint8_t v) { id = (id ^ v) * kFnvPrime; };
    for (uint32_t i = 0; i < stride * uint32_t(rows); i++) mix(m.mem[(data + i) & 0xFFFFF]);
    mix(uint8_t(width));
    for (int c = 1; c < 16; c++)
        for (int k = 0; k < 3; k++) mix(m.dac[uint8_t(base + c)][k]);

    const uint32_t slot = slot_for(id, width, rows, colours, palette_of(m, colours), [&] {
        // Stored top row first, as pictures are, with transparent pixels where colour 0 is.
        std::vector<uint32_t> rgba(size_t(width) * size_t(rows));
        for (int y = 0; y < rows; y++) {
            const uint32_t row = data + uint32_t(rows - 1 - y) * stride;
            for (int x = 0; x < width; x++) {
                const uint8_t c = nibble(m, row, x);
                rgba[size_t(y) * size_t(width) + size_t(x)] = c ? colour_of(m, uint8_t(base + c)) : 0;
            }
        }
        return rgba;
    });
    // Sprites drawn later lie on top. The count runs from 1 to 4095 and starts again; 0 marks pictures.
    const uint16_t order = uint16_t(draw_count_ % 4095 + 1);
    draw_count_++;

    for (int r = 0; r < rows; r++) {
        const uint32_t row = data + uint32_t(r) * stride;
        const uint32_t line = dest - uint32_t(r) * 320;
        for (int x = 0; x < width; x++) {
            const uint8_t c = nibble(m, row, x);
            if (!c) continue;
            sprite_marks_[(line + uint32_t(x)) & 0xFFFFF] = {slot + 1, uint16_t(x), uint16_t(rows - 1 - r), uint8_t(base + c), order};
        }
    }
}

void HdSprites::image(const Machine& m, uint32_t data, int width, int rows, uint32_t stride, uint32_t dest, uint32_t dest_stride) {
    if (width <= 0 || width > 1024 || rows <= 0 || rows > 1024) return;
    Deferred d{std::vector<uint8_t>(size_t(width) * size_t(rows)), width, rows, dest, dest_stride};
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < width; x++) d.pixels[size_t(y) * size_t(width) + size_t(x)] = m.mem[(data + uint32_t(y) * stride + uint32_t(x)) & 0xFFFFF];
    std::lock_guard<std::mutex> lock(mutex_);
    // Only the latest few wait; an older picture would be covered by a newer one at the same place anyway.
    if (deferred_.size() >= kMaxDeferred) deferred_.erase(deferred_.begin());
    deferred_.push_back(std::move(d));
}

void HdSprites::take_over(const Machine& m, const Deferred& d) {
    std::bitset<256> colours;
    for (uint8_t c : d.pixels) colours.set(c);
    uint64_t id = kFnvBasis;
    auto mix = [&id](uint8_t v) { id = (id ^ v) * kFnvPrime; };
    mix(0xFF);  // whole pictures never share a key with sprites
    mix(uint8_t(d.width));
    mix(uint8_t(d.width >> 8));
    for (uint8_t c : d.pixels) {
        mix(m.dac[c][0]);
        mix(m.dac[c][1]);
        mix(m.dac[c][2]);
    }
    const uint32_t slot = slot_for(id, d.width, d.rows, colours, palette_of(m, colours), [&] {
        std::vector<uint32_t> rgba(d.pixels.size());
        for (size_t i = 0; i < rgba.size(); i++) rgba[i] = colour_of(m, d.pixels[i]);
        return rgba;
    });
    for (int y = 0; y < d.rows; y++)
        for (int x = 0; x < d.width; x++)
            image_marks_[(d.dest + uint32_t(y) * d.dest_stride + uint32_t(x)) & 0xFFFFF] = {
                slot + 1, uint16_t(x), uint16_t(y), d.pixels[size_t(y) * size_t(d.width) + size_t(x)], 0};
}

void HdSprites::set_screen_source(uint32_t linear) { screen_source_ = linear; }

void HdSprites::frame(const Machine& m, uint32_t* refs) {
    const int64_t now = now_ms();
    for (int c = 0; c < 256; c++)
        if (std::memcmp(last_dac_[c], m.dac[c], 3) != 0) {
            std::memcpy(last_dac_[c], m.dac[c], 3);
            colour_changed_[c] = now;
        }

    // Pictures whose colours have settled are taken over, outside the lock.
    std::vector<Deferred> ready;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Deferred> waiting;
        for (auto& d : deferred_) {
            std::bitset<256> colours;
            bool visible = false;
            for (uint8_t c : d.pixels) {
                colours.set(c);
                visible = visible || (m.dac[c][0] | m.dac[c][1] | m.dac[c][2]);
            }
            // A picture shown on a black screen waits until it is faded in.
            if (visible && settled(colours)) ready.push_back(std::move(d));
            else waiting.push_back(std::move(d));
        }
        deferred_ = std::move(waiting);
    }
    for (const auto& d : ready) take_over(m, d);

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& s : slots_)
        if (s.state == State::placed) s.current = palette_of(m, s.colours) == s.palette;
    auto ref_of = [&](const Mark& mark, uint8_t colour) -> uint32_t {
        if (!mark.slot || mark.slot > slots_.size() || colour != mark.colour) return kNone;
        const Slot& s = slots_[mark.slot - 1];
        if (s.state != State::placed || !s.current) return kNone;
        return encode((s.atlas & 0xFFFF) + mark.x, (s.atlas >> 16) + mark.y, mark.order);
    };
    const uint32_t source = screen_source_;
    for (int i = 0; i < kPixels; i++) {
        refs[i] = kNone;
        // Where the screen pixel was drawn: in the game's own picture buffer when it copies that to the screen
        // as a whole, otherwise wherever its value came from.
        uint32_t at;
        if (source) {
            at = source + uint32_t(i);
        } else {
            const uint32_t from = m.origin[kVideo + uint32_t(i)];
            if (!from) continue;
            at = from - 1;
        }
        at &= 0xFFFFF;
        const uint8_t colour = m.mem[kVideo + uint32_t(i)];
        // A sprite drawn over a picture wins while it is still there; when the game restores the picture
        // underneath, the colours match the picture again.
        refs[i] = ref_of(sprite_marks_[at], colour);
        if (refs[i] == kNone) refs[i] = ref_of(image_marks_[at], colour);
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

void HdSprites::atlas_cleared() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& s : slots_)
        if (s.state == State::placed) s.state = State::pending;
}

size_t HdSprites::sprite_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return slots_.size();
}

size_t HdSprites::missing_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = 0;
    for (const auto& s : slots_) n += s.state == State::missing ? 1 : 0;
    return n;
}

}  // namespace nax
