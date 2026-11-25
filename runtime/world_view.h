// Expanded native-resolution isometric world viewport and docked HUD layout for widescreen and ultrawide displays.
#pragma once

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nax {

// One static map tile or dynamic overlay sprite to render in back-to-front isometric order.
struct WorldTile {
    float row = 0.0f;
    float col = 0.0f;
    int width = 32;
    int rows = 15;
    int pixel_dx = 0;  // fine sub-tile horizontal offset in DOS pixels (for dynamic sprites)
    int pixel_dy = 0;  // fine sub-tile vertical offset in DOS pixels (for dynamic sprites)
    bool has_hd = false;
    bool dimmed = false;
    uint16_t atlas_x = 0;
    uint16_t atlas_y = 0;
    uint32_t fallback_id = 0;  // 1-based index into WorldFallbackAtlas when has_hd is false
};

// A docked region of the original 320x200 screen (such as the top status bar, right toolbar, compass, or message bar).
struct HudStrip {
    RECT dst{};
    int src_x = 0, src_y = 0, src_w = 0, src_h = 0;
    bool solid_backdrop = false;   // when true, also draw a dark full-width header/footer backing bar behind the strip
    bool key_out_backdrop = false; // when true, discard full-screen backdrop (order 0) pixels so crisp HUD text floats cleanly
    bool diamond_clip = false;     // when true, clip the strip to an isometric diamond (removes stone corners around the compass)
};

// Result of mapping a client-window pointer position onto the expanded ultrawide view.
struct WorldHit {
    enum class Kind : uint8_t { none, hud, tile };
    Kind kind = Kind::none;
    int dos_x = 0, dos_y = 0;
    int tile_row = 0, tile_col = 0;
    bool need_cam_shift = false;
    int shift_cam_row = 0, shift_cam_col = 0;
};

// CPU-side cache of decoded 32xN fallback sprites (for un-remastered tiles on the 100x100 planet map) packed into
// a 1024x1024 RGBA texture uploaded to Direct3D 12 on demand.
class WorldFallbackAtlas {
public:
    static constexpr int kSize = 1024;
    struct Entry {
        uint16_t x = 0, y = 0, w = 0, h = 0;
    };

    WorldFallbackAtlas();
    void clear();
    // Returns a 1-based entry ID, decoding and packing the sprite once per unique key.
    uint32_t get_or_add(uint64_t key, int w, int h, const uint32_t* rgba);
    const Entry* entry(uint32_t id) const;
    const uint32_t* pixels() const { return pixels_.data(); }
    uint64_t version() const { return version_; }
    uint64_t clear_count() const { return clear_count_; }
    int free_rows() const { return std::max(0, kSize - shelf_y_ - shelf_h_ - 1); }

private:
    std::vector<uint32_t> pixels_;
    std::vector<Entry> entries_;
    std::unordered_map<uint64_t, uint32_t> by_key_;
    int shelf_x_ = 1, shelf_y_ = 1, shelf_h_ = 0;
    uint64_t version_ = 1;
    uint64_t clear_count_ = 0;
};

// Complete frame payload handed from the game state to Presenter for native ultrawide rendering.
struct WorldRenderData {
    int win_w = 0, win_h = 0;
    float center_row = 0.0f, center_col = 0.0f;
    float tile_scale_x = 4.0f;  // native pixels per DOS horizontal pixel
    float tile_scale_y = 4.8f;  // native pixels per DOS vertical pixel (preserves 4:3 CRT aspect ratio)
    float origin_x = 0.0f;      // screen X of (center_row, center_col) diamond center
    float origin_y = 0.0f;      // screen Y of (center_row, center_col) diamond center
    float ambient_tint[3] = {1.0f, 1.0f, 1.0f};
    bool has_hover = false;
    int hover_row = 0, hover_col = 0;
    bool has_popup = false;
    bool popup_has_holes = false;  // true for L-shaped multi-box popups such as Intelligence Reports
    int popup_src_x = 0, popup_src_y = 0, popup_src_w = 0, popup_src_h = 0;
    RECT popup_dst{};
    bool has_bottom_message = true;  // false when Utopia's bottom message console is empty (auto-hides the blank bar)
    int bottom_msg_src_h = 0;        // when > 0, fits the bottom message toast tightly to active text rows (y = 170..)
    bool has_game_cursor = false;
    float cursor_x = 0.0f, cursor_y = 0.0f;
    float cursor_scale_x = 3.0f, cursor_scale_y = 3.6f;
    uint32_t cursor_fallback_id = 0;
    std::vector<WorldTile> tiles;
    std::vector<HudStrip> hud_strips;
    const WorldFallbackAtlas* fallback = nullptr;
};

// Pure geometry, camera, zoom, HUD docking, and click-to-tile controller for the expanded isometric world view.
class WorldView {
public:
    // Utopia's planet map in u.mdg (ds:0xbaa8, stride 0xcc) is 75 rows by 100 columns (0x4b x 0x64).
    static constexpr int kMapRows = 75;
    static constexpr int kMapCols = 100;
    static constexpr int kGameViewSize = 8;
    static constexpr int kMaxGameCamRow = kMapRows - kGameViewSize;  // 67 (0x43 in u.mdg)
    static constexpr int kMaxGameCamCol = kMapCols - kGameViewSize;  // 92 (0x5c in u.mdg)
    static constexpr float kMinZoom = 0.45f;
    static constexpr float kMaxZoom = 2.00f;

    // Synchronizes our smooth ultrawide camera when the game itself jumps the camera (save load, minimap jump,
    // or scroll arrows) without snapping back on our own click-induced logic-window shifts.
    void sync_game_camera(int game_cam_row, int game_cam_col);
    // Records an expected logic-window shift initiated by clicking outside the 8x8 core so sync_game_camera()
    // keeps the visual ultrawide camera completely steady instead of jumping.
    void note_click_shift(int target_cam_row, int target_cam_col);
    void pan_screen_pixels(float dx_px, float dy_px, int win_w, int win_h);
    void pan_tiles(float drow, float dcol);
    void adjust_zoom(float factor);
    void reset_zoom() { zoom_ = 0.90f; }

    float center_row() const { return center_row_; }
    float center_col() const { return center_col_; }
    float zoom() const { return zoom_; }
    int game_cam_row() const { return last_game_row_; }
    int game_cam_col() const { return last_game_col_; }

    // Computes pixel scale factors and the visual centre of the map viewport (centred in the area left of the
    // right toolbar and between the top status bar and bottom message bar).
    void compute_projection(int win_w, int win_h, float* scale_x, float* scale_y, float* origin_x, float* origin_y) const;
    // Computes the docked HUD rectangles for the given native client size.
    static std::vector<HudStrip> compute_hud_strips(int win_w, int win_h);
    // Computes the centred floating rectangle for an in-game popup (such as SELECT A BUILDING.. or building info)
    // inside the open map area left of the docked sidebar and between the top and (optionally) bottom bars.
    static RECT compute_popup_rect(int win_w, int win_h, int src_w, int src_h, bool hide_bottom_bar = false,
                                   bool hide_top_bar = false);
    // Crops the top button row (y = 59..78) off the right command sidebar strip when an L-shaped advisor portrait
    // popup overlaps the upper right corner (y < 14, right edge > 268), leaving the 4 visible lower rows intact.
    static bool crop_sidebar_for_popup(HudStrip& strip, const WorldRenderData* active_world);
    // Crops the bottom message strip (strips[1]) into a compact floating dark-glass toast sized to active text rows.
    static bool fit_bottom_strip_for_message(HudStrip& strip, const WorldRenderData* active_world);
    // Projects a tile (row, col) of size (width x rows) into native client pixel coordinates.
    static void project_tile(const WorldRenderData& data, float row, float col, int width, int rows, int pixel_dx, int pixel_dy,
                             float* x0, float* y0, float* x1, float* y1);
    // Inverse-projects a native client pixel into fractional planet map coordinates (row, col).
    void screen_to_map(float px, float py, int win_w, int win_h, float* row_out, float* col_out) const;
    // Maps a client window point to either a docked HUD strip, an active floating popup, or a planet tile.
    WorldHit hit_test(float px, float py, int win_w, int win_h, int game_cam_row, int game_cam_col,
                      const WorldRenderData* active_world = nullptr) const;
    // Converts a local tile offset (dr, dc) inside Utopia's 8x8 diamond (0..7, 0..7) to the centre pixel in 320x200.
    static void game_tile_center_dos(int dr, int dc, int* dos_x, int* dos_y);
    // Computes the inclusive planet tile bounds [r0..r1, c0..c1] visible in the current viewport (with margin for tall roofs).
    void visible_bounds(int win_w, int win_h, int* r0, int* r1, int* c0, int* c1) const;

private:
    void clamp_camera();

    bool initialized_ = false;
    float center_row_ = 37.0f;
    float center_col_ = 51.0f;
    float zoom_ = 0.90f;
    int last_game_row_ = -1;
    int last_game_col_ = -1;
    bool has_click_shift_ = false;
    int click_shift_row_ = -1;
    int click_shift_col_ = -1;
};

}  // namespace nax
