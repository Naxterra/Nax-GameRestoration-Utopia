// High-resolution replacements for a game's sprites.
//
// Game-specific hooks report every sprite the game draws. Each distinct sprite (its pixels together with the
// colours it is drawn in) is saved once as a PNG in <folder>/original, and every pixel it covers is marked with
// the sprite and the position inside it. When a picture with the same name exists in <folder>/upscaled, for
// example one made by tools/ai/upscale_sprites.py, the presenter places it in a texture atlas, and each frame
// tells it, for every screen pixel, which part of that picture to draw there instead of the original pixel.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nax {

class Machine;

class HdSprites {
public:
    static constexpr int kScale = 8;                 // the atlas holds pictures eight times as large (four is doubled)
    static constexpr uint32_t kNone = 0xFFFFFFFFu;  // a screen pixel without a high-resolution picture

    HdSprites();

    // The folder for the original and upscaled pictures. Without one, sprites are still tracked but not saved.
    void set_folder(const std::filesystem::path& folder);

    // Called on the game thread when the game draws a sprite of 4-bit pixels, two per byte with the left pixel
    // in the high half. The sprite is `width` pixels wide and `rows` high, stored from the bottom row up, and
    // is drawn upward from linear address `dest` (the bottom row's first pixel) with a line length of 320.
    // Colour 0 is transparent, and every other colour is drawn as `base` plus the colour.
    void draw(const Machine& m, uint32_t data, int width, int rows, uint32_t dest, uint8_t base);

    // Called for each frame: for each of the 320x200 screen pixels, the atlas cell (x in the low 16 bits, y in
    // the high 16 bits, in original pixels) whose upscaled picture belongs there, or kNone.
    void frame(const Machine& m, uint32_t* refs);

    // Upscaled pictures that exist on disk but are not in the atlas yet. The presenter loads each one and
    // reports its position with placed(), or with failed() when it cannot be used.
    struct Pending {
        uint32_t slot;
        std::filesystem::path file;
        int width, rows;
    };
    std::vector<Pending> take_pending();
    void placed(uint32_t slot, uint16_t x, uint16_t y);
    void failed(uint32_t slot);

    size_t sprite_count();

private:
    struct Mark {
        uint32_t slot = 0;  // 1 + the sprite's slot, or 0 for no sprite
        uint8_t x = 0, y = 0;
        uint8_t colour = 0;  // the colour the sprite put there, to notice when something else was drawn over it
        uint8_t unused = 0;
    };
    enum class State : uint8_t { missing, pending, placed, failed };
    struct Slot {
        uint64_t key;
        int width, rows;
        State state;
        uint32_t atlas;  // x | y << 16 once placed
    };

    std::filesystem::path original_file(uint64_t key) const;
    std::filesystem::path upscaled_file(uint64_t key) const;

    std::vector<Mark> marks_;  // one per byte of memory
    std::mutex mutex_;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, uint32_t> by_key_;
    std::filesystem::path folder_;
    std::chrono::steady_clock::time_point next_scan_{};
};

}  // namespace nax
