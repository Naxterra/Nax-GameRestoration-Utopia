// High-resolution replacements for a game's sprites.
//
// Game-specific hooks report every sprite the game draws. Each distinct sprite (its pixels together with the
// colours it is drawn in) is saved once as a PNG in <folder>/original, and every pixel it covers is marked with
// the sprite and the position inside it. When a picture with the same name exists in <folder>/upscaled, for
// example one made by tools/ai/upscale_sprites.py, the presenter places it in a texture atlas, and each frame
// tells it, for every screen pixel, which part of that picture to draw there instead of the original pixel.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
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

    // Called on the game thread when the game has put a whole picture of 8-bit pixels (a background or a panel)
    // at linear address `data`, `width` pixels per row with `stride` bytes between rows, top row first, which
    // is shown at `dest` with `dest_stride` bytes between rows. Sprites drawn over it take precedence where
    // they are; where the game later restores the picture, the picture's upscaled version shows again.
    // Games often unpack a picture while the screen is still dark and fade it in afterwards, so the picture is
    // only taken over once the palette has stayed the same for half a second and is not black.
    void image(const Machine& m, uint32_t data, int width, int rows, uint32_t stride, uint32_t dest, uint32_t dest_stride);

    // For games that draw into a 320x200 buffer and copy it to the screen as a whole: the buffer's linear
    // address. Each screen pixel is then looked up at the same place in that buffer.
    void set_screen_source(uint32_t linear);

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
        uint16_t x = 0, y = 0;
        uint8_t colour = 0;  // the colour the sprite put there, to notice when something else was drawn over it
    };
    enum class State : uint8_t { missing, pending, placed, failed };
    struct Slot {
        uint64_t key;
        int width, rows;
        State state;
        uint32_t atlas;  // x | y << 16 once placed
    };

    struct Deferred {
        std::vector<uint8_t> pixels;
        int width, rows;
        uint32_t dest, dest_stride;
    };
    uint32_t slot_for(uint64_t id, int width, int rows, const std::function<std::vector<uint32_t>()>& picture);  // mutex held
    void take_over(const Machine& m, const Deferred& d);  // mutex held
    bool palette_settled() const;
    std::filesystem::path original_file(uint64_t key) const;
    std::filesystem::path upscaled_file(uint64_t key) const;

    std::vector<Mark> sprite_marks_, image_marks_;  // one of each per byte of memory
    std::atomic<uint32_t> screen_source_{0};
    std::mutex mutex_;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, uint32_t> by_key_;
    std::filesystem::path folder_;
    std::chrono::steady_clock::time_point next_scan_{};
    std::vector<Deferred> deferred_;
    uint8_t last_dac_[256][3]{};
    std::atomic<int64_t> palette_changed_{0};  // steady clock in milliseconds
};

}  // namespace nax
