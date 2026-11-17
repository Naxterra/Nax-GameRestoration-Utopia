// High-resolution replacements for a game's sprites.
//
// Game-specific hooks report every sprite the game draws. Each distinct sprite (its pixels together with the
// colours it is drawn in) is saved once as a PNG in <folder>/original, and every pixel it covers is marked with
// the sprite and the position inside it. When a picture with the same name exists in <folder>/upscaled, for
// example one made by tools/ai/upscale_sprites.py, the presenter places it in a texture atlas, and each frame
// tells it, for every screen pixel, which part of that picture to draw there instead of the original pixel.
//
// An upscaled picture has the colours of the moment it was saved. Whenever the palette entries a sprite uses
// differ from those colours, for example while the game fades the screen, its original pixels are shown.
#pragma once

#include <atomic>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nax {

class Machine;

// Decides whether a full texture atlas is emptied to make room. Emptying helps when the screen has moved on
// to other pictures, but not when the pictures on screen alone fill the atlas: then each emptying would only
// make every upscaled picture flicker back to its original pixels while the same set is loaded again. So the
// atlas is only emptied when the pictures on screen need at most half of it, and at most once every 10
// seconds. If it runs full again within those 10 seconds while the screen needs about as much as it did at
// the emptying (between three quarters and five quarters of it), the emptying did not help, and for the next
// minute the atlas is not emptied for a screen that needs about that much. A screen that needs clearly less
// or clearly more counts as a different screen and is not held back by this.
class AtlasClearing {
public:
    explicit AtlasClearing(uint64_t capacity) : capacity_(capacity) {}
    // A picture found no room. `on_screen` is the room the pictures on screen need, in the same unit as the
    // capacity. Returns whether the atlas is to be emptied now.
    bool overflow(int64_t now_ms, uint64_t on_screen);

private:
    struct Held {
        uint64_t need;  // what the screen needed when emptying did not help
        int64_t until;  // when it may be emptied for again
    };
    uint64_t capacity_;
    int64_t cleared_at_ = INT64_MIN / 2;
    uint64_t cleared_for_ = 0;  // what the screen needed when the atlas was last emptied
    std::vector<Held> held_;
};

// Which sprites the game draws side by side, so that tools/ai/remaster.py can redraw tiles that belong together
// (the four tiles of a landing pad, or the pieces of a building on several tiles) as one picture instead of
// giving each piece edges of its own. Two sprites count as neighbours when they have the same width, are both
// flat (at most half as high as wide) or both tall, lie next to each other on an isometric tile grid (half
// their width across and a quarter of their width up or down, measured at their bottom rows) and were drawn
// in the same or in consecutive frames. A building or a unit over a ground tile is therefore never a
// neighbour of it. Tall sprites side by side are, whether they are pieces of one building or a unit standing
// next to a building; remaster.py tells these apart. A pair is reported once it has been seen in three
// frames: right after the game scrolls, sprites at their old places can briefly meet sprites at their new
// places, and that happens only in a frame or two.
class NeighbourLog {
public:
    // Sprite `b` lies `dx` pixels to the right of and `dy` pixels below sprite `a`, measured from the first
    // pixel of each sprite's bottom row. The sprite with the smaller key is always `a`.
    struct Pair {
        uint64_t a, b;
        int dx, dy;
        bool operator<(const Pair& o) const { return std::tie(a, b, dx, dy) < std::tie(o.a, o.b, o.dx, o.dy); }
    };
    // A sprite was drawn with the first pixel of its bottom row at linear address `dest` (line length 320)
    // during presenter frame `frame`. Returns the pairs that have now been seen often enough to be recorded.
    std::vector<Pair> drawn(uint64_t key, int width, int rows, uint32_t dest, uint32_t frame);
    // A pair that is recorded already, for example in the file of an earlier run.
    void known(const Pair& pair);

private:
    struct Recent {
        uint64_t key;
        int width, rows;
        uint32_t dest, frame;
    };
    struct Seen {
        uint32_t count = 0, frame = 0;
    };
    static constexpr size_t kRecent = 64;
    Recent recent_[kRecent]{};
    size_t recent_count_ = 0, recent_next_ = 0;
    std::map<Pair, Seen> seen_;  // pairs not reported yet
    std::set<Pair> reported_;
};

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
    // Colour 0 is transparent, and every other colour is drawn as `base` plus the colour. While the colours it
    // uses are changing (a fade), the sprite is not recorded, so no picture is saved for a passing shade.
    void draw(const Machine& m, uint32_t data, int width, int rows, uint32_t dest, uint8_t base);

    // Called on the game thread when the game has put a whole picture of 8-bit pixels (a background or a panel)
    // at linear address `data`, `width` pixels per row with `stride` bytes between rows, top row first, which
    // is shown at `dest` with `dest_stride` bytes between rows. Sprites drawn over it take precedence where
    // they are; where the game later restores the picture, the picture's upscaled version shows again.
    // Games often unpack a picture while the screen is still dark and fade it in afterwards, so the picture is
    // only taken over once the colours it uses have stayed the same for half a second and are not all black.
    void image(const Machine& m, uint32_t data, int width, int rows, uint32_t stride, uint32_t dest, uint32_t dest_stride);

    // For games that draw into a 320x200 buffer and copy it to the screen as a whole: the buffer's linear
    // address. Each screen pixel is then looked up at the same place in that buffer.
    void set_screen_source(uint32_t linear);

    // Called for each frame: for each of the 320x200 screen pixels, the atlas cell whose upscaled picture
    // belongs there, or kNone. A reference holds the cell's x in bits 0 to 9 and its y in bits 10 to 19 (in
    // original pixels; the atlas is 1024 cells wide and high), and in bits 20 to 31 the order in which sprites
    // were drawn: 0 for a picture underneath everything, then 1 to 4095 counting up and starting again, so the
    // presenter can tell which of two neighbouring sprites lies on top where their smooth outlines meet.
    static uint32_t encode(uint32_t x, uint32_t y, uint32_t order) { return (x & 0x3FF) | ((y & 0x3FF) << 10) | (order << 20); }
    void frame(const Machine& m, uint32_t* refs);

    // Upscaled pictures that exist on disk and were seen on screen, but are not in the atlas yet. Pictures are
    // only loaded once they show, so the atlas holds what the game has displayed, not every picture on disk.
    // The presenter loads each one and reports its position with placed(), with failed() when the picture
    // cannot be used, or with no_room() when the atlas is full: such a picture is asked for again the next
    // time it shows.
    struct Pending {
        uint32_t slot;
        std::filesystem::path file;
        int width, rows;
    };
    std::vector<Pending> take_pending();
    void placed(uint32_t slot, uint16_t x, uint16_t y);
    void failed(uint32_t slot);
    void no_room(uint32_t slot);
    // The presenter emptied its full atlas: every picture is loaded again when it next shows on screen.
    void atlas_cleared();

    // Sprites the game draws side by side are written to <folder>/layout.txt, one pair per line: the two
    // sprites' names and where the second lies from the first (see NeighbourLog).

    // The atlas cells, gaps included, that the pictures seen on screen in the last frame need.
    uint64_t on_screen_cells();

    size_t sprite_count();
    size_t missing_count();  // saved sprites without an upscaled picture yet

private:
    struct Mark {
        uint32_t slot = 0;  // 1 + the sprite's slot, or 0 for no sprite
        uint16_t x = 0, y = 0;
        uint8_t colour = 0;  // the colour the sprite put there, to notice when something else was drawn over it
        uint16_t order = 0;  // see encode()
    };
    // missing: no upscaled picture on disk. unloaded: a picture on disk that is not in the atlas and is asked
    // for as soon as it shows. pending: asked for. placed: in the atlas. failed: the picture cannot be used.
    enum class State : uint8_t { missing, unloaded, pending, placed, failed };
    struct Slot {
        uint64_t key;
        int width, rows;
        State state;
        uint32_t atlas;           // x | y << 16 once placed
        std::bitset<256> colours;  // the palette entries the sprite's pixels use
        uint64_t palette;          // their colours when the sprite was saved
        bool current = true;       // whether those entries still have those colours (updated each frame)
        uint32_t seen = 0;         // the last frame in which the sprite was on screen
    };
    struct Deferred {
        std::vector<uint8_t> pixels;
        int width, rows;
        uint32_t dest, dest_stride;
    };

    uint32_t slot_for(uint64_t id, uint64_t legacy, int width, int rows, const std::bitset<256>& colours, uint64_t palette,
                      const std::function<std::vector<uint32_t>()>& picture);
    void take_over(const Machine& m, const Deferred& d);
    bool settled(const std::bitset<256>& colours) const;
    std::filesystem::path original_file(uint64_t key) const;
    std::filesystem::path upscaled_file(uint64_t key) const;
    void record_neighbours(uint64_t key, int width, int rows, uint32_t dest);

    std::vector<Mark> sprite_marks_, image_marks_;  // one of each per byte of memory
    std::atomic<uint32_t> screen_source_{0};
    uint32_t draw_count_ = 0;  // only used on the game thread
    uint32_t frame_count_ = 0;
    uint64_t on_screen_cells_ = 0;
    std::mutex mutex_;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, uint32_t> by_key_;
    std::filesystem::path folder_;
    std::chrono::steady_clock::time_point next_scan_{};
    std::vector<Deferred> deferred_;
    uint8_t last_dac_[256][3]{};
    std::atomic<int64_t> colour_changed_[256]{};  // per palette entry, steady clock in milliseconds
    std::atomic<uint32_t> frames_{0};            // frames so far, for the neighbour log
    NeighbourLog neighbours_;                    // only used on the game thread
    bool layout_loaded_ = false;                 // only used on the game thread
};

}  // namespace nax
