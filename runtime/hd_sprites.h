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

#include <array>
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
    static constexpr int kScale = 8;                    // the atlas holds pictures eight times as large (four is doubled)
    static constexpr uint32_t kNone = 0xFFFFFFFFu;      // a screen pixel without a high-resolution picture
    static constexpr uint32_t kPopupHole = 0xFFFFFFFEu; // unmasked background gap inside an L-shaped popup bounding box

    HdSprites();

    // The folder for the original and upscaled pictures. Without one, sprites are still tracked but not saved.
    void set_folder(const std::filesystem::path& folder);

    // Called on the game thread when the game draws a sprite of 4-bit pixels, two per byte with the left pixel
    // in the high half. The sprite is `width` pixels wide and `rows` high. When `bottom_up` is true, rows are
    // stored from the bottom row up and drawn upward from linear address `dest`; when false, rows are stored top
    // row first and drawn downward from `dest` (line length 320). Colour 0 is transparent, and every other colour
    // is drawn as `base` plus the colour. While the colours it uses are changing (a fade), the sprite is not
    // recorded, so no picture is saved for a passing shade.
    void draw(const Machine& m, uint32_t data, int width, int rows, uint32_t dest, uint8_t base, bool bottom_up = true,
              bool track_neighbours = true);

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
    uint32_t screen_source() const { return screen_source_.load(std::memory_order_acquire); }

    // Called on the game thread right before the game starts copying its 320x200 offscreen buffer to video
    // memory (0xA0000), freezing a coherent copy of live video memory so the render thread never sees a torn
    // mid-copy frame while `rep movsw` runs.
    void begin_present(const Machine& m);

    // Called when the game has finished copying its 320x200 offscreen buffer to video memory (0xA0000).
    // Snapshots the buffer's sprite and backdrop marks into a double-buffered front copy so subsequent
    // offscreen redraws for the next frame never race against the displayed VRAM frame.
    void present_buffer(uint32_t linear);

    // Called when the game fills a rectangular UI box or panel (line length 320) over existing sprites and
    // backdrop, so covered sprites and backdrop pixels cannot accidentally bleed through on colour collisions.
    void clear_rect(uint32_t dest, int width, int rows);

    // Called when the game restores a previously saved copy of its map/backdrop over a rectangle (line length
    // 320), unmasking the covered sprite and backdrop marks so their HD pictures show again.
    void restore_rect(uint32_t dest, int width, int rows);

    // Called when the game overwrites a rectangular map region in its offscreen buffer with a clean saved
    // backdrop before drawing the next frame's tiles, clearing stale sprite marks while keeping backdrop marks.
    void restore_backdrop_rect(uint32_t dest, int width, int rows);

    // Called after the game clears a text area with its stippled background tile, updating the active HD
    // backdrop's expected pixel colours to the newly written clear pattern so cleared areas stay seamless while
    // text drawn afterwards still renders on top.
    void sync_image_rect(const Machine& m, uint32_t dest, int width, int rows);

    // Called for each frame: for each of the 320x200 screen pixels, the atlas cell whose upscaled picture
    // belongs there, or kNone. A reference holds the cell's x in bits 0 to 9, y in bits 10 to 19 (in original
    // pixels; the atlas is 1024 cells wide and high), drawing order in bits 20 to 30 (0 for a picture
    // underneath everything, 1 to 2046 wrapping for sprites), and bit 31 set when the slot is dimmed by ambient
    // lighting so the presenter tints only dimmed HD layers without double-tinting undimmed or DOS pixels.
    static uint32_t encode(uint32_t x, uint32_t y, uint32_t order, bool dimmed = false) {
        return (x & 0x3FF) | ((y & 0x3FF) << 10) | ((order & 0x7FF) << 20) | (dimmed ? 0x80000000u : 0u);
    }
    // When `vram_snapshot` is provided (64,000 bytes), pixel colours are read from that snapshot rather than
    // live video memory so the RGBA frame and sprite references always come from the exact same instant.
    void frame(const Machine& m, uint32_t* refs, const uint8_t* vram_snapshot = nullptr);
    // Atomically snapshots coherent video memory (64,000 bytes) into `vram_out` and fills `refs` under the
    // same lock as `begin_present()` and `present_buffer()`, preventing mid-draw flicker and half-copy tearing.
    // An optional `preprocess_vram` callback may normalize `vram_out` (for example erasing a software cursor)
    // before sprite references are built, without double-counting frames or duplicating work.
    void snapshot_and_frame(const Machine& m, uint8_t* vram_out, uint32_t* refs,
                            const std::function<void(uint8_t*)>& preprocess_vram = {});

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

    // Active ambient RGB lighting multiplier (1, 1, 1 in full daylight, or the dimmed night/weather tint
    // detected when the game dims its VGA palette uniformly across the active HD screen/sprites).
    void ambient_tint(float out[3]);

    // Lightweight atlas lookup for the expanded ultrawide world view: resolves a 4-bit game sprite (already in
    // the atlas or available in upscaled/ on disk) without writing to original/ or layout.txt.
    struct WorldSpriteInfo {
        bool placed = false;
        bool dimmed = false;
        uint16_t atlas_x = 0, atlas_y = 0;
        uint64_t key = 0;
    };
    void begin_world_frame();
    WorldSpriteInfo resolve_world_sprite(const Machine& m, uint32_t data, int width, int rows, uint8_t base,
                                         bool bottom_up = true);
    // True when the live offscreen colony map buffer is being presented. When optional popup output pointers
    // are supplied and a popup masks part of the map area, returns true and populates its tight bounding box
    // instead of forcing a fallback to 4:3 (full-screen VRAM menus still return false).
    bool is_showing_offscreen_map(int* popup_x0 = nullptr, int* popup_y0 = nullptr, int* popup_x1 = nullptr,
                                  int* popup_y1 = nullptr, bool* popup_has_holes = nullptr);
    // Marks unmasked pixels inside an L-shaped popup bounding box with kPopupHole in `refs` so the popup
    // shader discards leaked 4:3 backdrop/map gaps (such as below the portrait on Intelligence Reports).
    void mark_popup_holes(uint32_t* refs, int x0, int y0, int x1, int y1);

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
        uint64_t legacy;  // the name earlier versions gave the sprite, after all 15 colours of its bank
        int width, rows;
        State state;
        uint32_t atlas;           // x | y << 16 once placed
        std::bitset<256> colours;  // the palette entries the sprite's pixels use
        uint64_t palette;          // their colours when the sprite was saved
        std::array<std::array<uint8_t, 3>, 256> ref_dac{};  // 6-bit VGA DAC colours when the sprite was saved
        bool current = true;       // whether those entries still have those colours (or a uniform ambient dim)
        bool dimmed = false;       // true when current matched via ambient dimming rather than exact palette
        float tint[3] = {1.0f, 1.0f, 1.0f};
        uint64_t shape = 0;        // canonical shape hash in disk_shapes_ (for sprites <= 256x256)
        uint64_t miss_shape_version = 0;  // last disk_shapes_version_ checked when state == missing
        bool miss_with_hint = false;      // whether that miss check already had a dimmed ambient hint
        float miss_hint[3] = {1.0f, 1.0f, 1.0f};
        uint32_t seen = 0;         // the last frame in which the sprite was on screen
        uint32_t world_seen = 0;   // the last expanded-world frame in which the sprite was visible
    };
    struct DiskShape {
        uint64_t key;
        int width, rows;
        bool has_upscaled;
        std::vector<std::array<uint8_t, 3>> colours;  // 6-bit daytime RGB per canonical non-transparent index
    };
    struct Deferred {
        std::vector<uint8_t> pixels;
        int width, rows;
        uint32_t dest, dest_stride;
        uint64_t epoch = 0;
        uint64_t seq = 0;
        uint64_t present_seq = 0;
        bool settling = false;
        std::vector<uint8_t> expected;  // updated when a VRAM text-clear tile overwrites pixels before settle
        std::vector<uint8_t> masked;    // non-zero where clear_rect() covered pixels before settle
    };

    uint32_t slot_for(uint64_t id, uint64_t legacy, int width, int rows, const std::bitset<256>& colours, uint64_t palette,
                      const std::array<std::array<uint8_t, 3>, 256>& ref_dac,
                      const std::function<std::vector<uint32_t>()>& picture);
    void index_disk_shapes();
    void take_over(const Machine& m, const Deferred& d, uint64_t matched_key = 0,
                   const std::array<std::array<uint8_t, 3>, 256>* matched_dac = nullptr);
    void frame_impl(const Machine& m, uint32_t* refs, const uint8_t* vram_snapshot, uint8_t* vram_out,
                    const std::function<void(uint8_t*)>& preprocess_vram = {});
    bool settled(const std::bitset<256>& colours) const;
    std::filesystem::path original_file(uint64_t key) const;
    std::filesystem::path upscaled_file(uint64_t key) const;
    void record_neighbours(uint64_t key, int width, int rows, uint32_t dest);
    void rename_in_layout(const std::filesystem::path& file, uint64_t from, uint64_t to);

    std::vector<Mark> sprite_marks_, image_marks_;  // one of each per byte of memory
    std::vector<Mark> front_sprite_marks_, front_image_marks_;  // double-buffered snapshot of presented offscreen buffer
    std::vector<uint8_t> stable_vram_;            // coherent VRAM snapshot held while rep movsw copies to 0xA0000
    bool has_front_ = false;                      // true once present_buffer() snapshotted front_*_marks_
    bool presenting_ = false;                     // true between begin_present() and present_buffer()
    uint32_t front_source_ = 0;                   // linear address whose marks live in front_*_marks_
    uint64_t present_seq_ = 0;                    // monotonic count incremented on every offscreen present_buffer()
    std::vector<uint8_t> vram_masked_;            // non-zero where a direct-VRAM UI fill hides the offscreen buffer
    std::atomic<uint32_t> screen_source_{0};
    std::atomic<bool> vram_dirty_{false};  // whether any sprite, picture or mask has been written directly in 0xA0000..0xAFA00
    uint64_t vram_epoch_ = 0;              // incremented whenever VRAM overlays are cleared (guarded by mutex_)
    uint64_t deferred_seq_ = 0;            // monotonic id for each pending picture (guarded by mutex_)
    uint32_t draw_count_ = 0;  // only used on the game thread
    uint32_t frame_count_ = 0;
    uint32_t world_frame_count_ = 0;
    uint64_t on_screen_cells_ = 0;
    uint64_t world_on_screen_cells_ = 0;
    std::mutex mutex_;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, uint32_t> by_key_;
    std::unordered_map<uint64_t, std::vector<DiskShape>> disk_shapes_;
    std::unordered_map<uint64_t, std::vector<DiskShape>> disk_images_;  // upscaled full-screen backdrops indexed by 256-colour shape
    struct WorldResolveCache {
        uint32_t slot = UINT32_MAX;  // UINT32_MAX marks a cached miss
        uint64_t shapes_version = 0;
        float hint[3] = {1.0f, 1.0f, 1.0f};
    };
    std::unordered_map<uint64_t, WorldResolveCache> world_resolve_cache_;
    uint64_t disk_shapes_version_ = 1;
    float ambient_tint_[3] = {1.0f, 1.0f, 1.0f};
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
