#include "world_view.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nax {

WorldFallbackAtlas::WorldFallbackAtlas() : pixels_(size_t(kSize) * size_t(kSize), 0) {}

void WorldFallbackAtlas::clear() {
    std::fill(pixels_.begin(), pixels_.end(), 0u);
    entries_.clear();
    by_key_.clear();
    shelf_x_ = 1;
    shelf_y_ = 1;
    shelf_h_ = 0;
    version_++;
    clear_count_++;
}

uint32_t WorldFallbackAtlas::get_or_add(uint64_t key, int w, int h, const uint32_t* rgba) {
    if (w <= 0 || h <= 0 || w > kSize - 2 || h > kSize - 2 || !rgba) return 0;
    auto it = by_key_.find(key);
    if (it != by_key_.end()) return it->second;
    if (shelf_x_ + w + 1 > kSize) {
        shelf_x_ = 1;
        shelf_y_ += shelf_h_ + 1;
        shelf_h_ = 0;
    }
    if (shelf_y_ + h + 1 > kSize) {
        clear();
    }
    const int x0 = shelf_x_, y0 = shelf_y_;
    for (int y = 0; y < h; y++)
        std::memcpy(&pixels_[size_t(y0 + y) * size_t(kSize) + size_t(x0)], rgba + size_t(y) * size_t(w), size_t(w) * 4);
    shelf_x_ += w + 1;
    shelf_h_ = std::max(shelf_h_, h);
    entries_.push_back({uint16_t(x0), uint16_t(y0), uint16_t(w), uint16_t(h)});
    const uint32_t id = uint32_t(entries_.size());
    by_key_.emplace(key, id);
    version_++;
    return id;
}

const WorldFallbackAtlas::Entry* WorldFallbackAtlas::entry(uint32_t id) const {
    if (id == 0 || id > entries_.size()) return nullptr;
    return &entries_[id - 1];
}

void WorldView::clamp_camera() {
    center_row_ = std::clamp(center_row_, 0.0f, float(kMapRows - 1));
    center_col_ = std::clamp(center_col_, 0.0f, float(kMapCols - 1));
}

void WorldView::sync_game_camera(int game_cam_row, int game_cam_col) {
    game_cam_row = std::clamp(game_cam_row, 0, kMaxGameCamRow);
    game_cam_col = std::clamp(game_cam_col, 0, kMaxGameCamCol);
    if (!initialized_) {
        initialized_ = true;
        center_row_ = float(game_cam_row) + 3.5f;
        center_col_ = float(game_cam_col) + 3.5f;
        last_game_row_ = game_cam_row;
        last_game_col_ = game_cam_col;
        return;
    }
    if (game_cam_row == last_game_row_ && game_cam_col == last_game_col_) return;
    if (has_click_shift_ && game_cam_row == click_shift_row_ && game_cam_col == click_shift_col_) {
        has_click_shift_ = false;
        last_game_row_ = game_cam_row;
        last_game_col_ = game_cam_col;
        return;
    }
    has_click_shift_ = false;
    const int dr = game_cam_row - last_game_row_;
    const int dc = game_cam_col - last_game_col_;
    // Small 1..2 tile steps (scroll arrows) nudge the ultrawide camera smoothly; large jumps (minimap click or
    // save load) recenter immediately on the destination.
    if (std::abs(dr) <= 2 && std::abs(dc) <= 2) {
        center_row_ += float(dr);
        center_col_ += float(dc);
    } else {
        center_row_ = float(game_cam_row) + 3.5f;
        center_col_ = float(game_cam_col) + 3.5f;
    }
    clamp_camera();
    last_game_row_ = game_cam_row;
    last_game_col_ = game_cam_col;
}

void WorldView::note_click_shift(int target_cam_row, int target_cam_col) {
    has_click_shift_ = true;
    click_shift_row_ = std::clamp(target_cam_row, 0, kMaxGameCamRow);
    click_shift_col_ = std::clamp(target_cam_col, 0, kMaxGameCamCol);
}

void WorldView::pan_screen_pixels(float dx_px, float dy_px, int win_w, int win_h) {
    float sx = 1.0f, sy = 1.0f, ox = 0.0f, oy = 0.0f;
    compute_projection(win_w, win_h, &sx, &sy, &ox, &oy);
    // Moving the grabbed map by (+dx_px, +dy_px) moves the camera center by (-dx_px, -dy_px).
    const float u = -dx_px / (16.0f * sx);
    const float v = -dy_px / (8.0f * sy);
    center_col_ += (v + u) * 0.5f;
    center_row_ += (v - u) * 0.5f;
    clamp_camera();
}

void WorldView::pan_tiles(float drow, float dcol) {
    center_row_ += drow;
    center_col_ += dcol;
    clamp_camera();
}

void WorldView::adjust_zoom(float factor) {
    zoom_ = std::clamp(zoom_ * factor, kMinZoom, kMaxZoom);
}

std::vector<HudStrip> WorldView::compute_hud_strips(int win_w, int win_h) {
    std::vector<HudStrip> strips;
    if (win_w <= 0 || win_h <= 0) return strips;
    // Scale HUD strips cleanly from window height (keeping 4:3 CRT aspect 1.2x vertical) capped so they stay
    // crisp and compact on ultrawide displays without crowding the expanded colony map.
    const float hud_sy = std::clamp(float(win_h) / 260.0f, 1.5f, 5.2f);
    const float hud_sx = hud_sy / 1.2f;
    auto rect_of = [](int x, int y, int w, int h) -> RECT { return RECT{x, y, x + w, y + h}; };

    // 1. Top status bar (Date / GR / Q.O.L.): compact floating dark-glass pill centred at the top so the
    // ultrawide colony terrain extends edge-to-edge on the left and right with zero full-width black bars.
    const int top_w = std::max(1, std::min(win_w - 24, int(std::round(320.0f * hud_sx))));
    const int top_h = int(std::round(14.0f * hud_sy));
    const int top_x = (win_w - top_w) / 2;
    const int top_y = 6;
    strips.push_back({rect_of(top_x, top_y, top_w, top_h), 0, 0, 320, 14, false, true, false});

    // 2. Bottom message bar: compact framed message box centred at the bottom edge with zero dark side wings.
    const int bot_w = std::max(1, std::min(win_w - 24, int(std::round(320.0f * hud_sx))));
    const int bot_h = int(std::round(32.0f * hud_sy));
    const int bot_x = (win_w - bot_w) / 2;
    const int bot_y = std::max(top_y + top_h + 4, win_h - bot_h - 6);
    strips.push_back({rect_of(bot_x, bot_y, bot_w, bot_h), 0, 168, 320, 32, false, false, false});

    // 3. Right column (Compass + Command Sidebar): docked on the right edge between top and bottom bars,
    // tightly cropped to the exact isometric compass diamond (256..320, 18..56) and 2x5 button bezel (268..316, 59..159)
    // so zero red-stripe stone floor scraps peek around the UI.
    const int avail_h = std::max(64, bot_y - (top_y + top_h));
    const float side_fit = std::min(1.0f, float(avail_h) / (146.0f * hud_sy));
    const float s_sx = hud_sx * side_fit;
    const float s_sy = hud_sy * side_fit;
    const int bar_w = int(std::round(48.0f * s_sx));
    const int bar_h = int(std::round(100.0f * s_sy));
    const int comp_w = int(std::round(64.0f * s_sx));
    const int comp_h = int(std::round(38.0f * s_sy));
    const int total_side_h = comp_h + bar_h + 8;
    const int side_top = top_y + top_h + std::max(4, (avail_h - total_side_h) / 2);
    const int margin_r = 10;

    const int col_right = win_w - margin_r;
    const int comp_x = col_right - comp_w;
    const int bar_x = col_right - (comp_w + bar_w) / 2;
    strips.push_back({rect_of(comp_x, side_top, comp_w, comp_h), 256, 18, 64, 38, false, false, true});
    strips.push_back({rect_of(bar_x, side_top + comp_h + 8, bar_w, bar_h), 268, 59, 48, 100, false, false, false});

    // 4. Compact scroll-arrow pad tightly cropped to the 4 golden arrows + base (216..262, 122..153).
    // On narrower windows where the pad would overlap the centred bottom message bar, lift it above the bar.
    const int pad_w = int(std::round(46.0f * s_sx));
    const int pad_h = int(std::round(31.0f * s_sy));
    const int pad_x = std::max(0, std::min(comp_x, bar_x) - pad_w - 10);
    const int pad_y = (pad_x < bot_x + bot_w + 10)
                          ? std::max(top_y + top_h + 4, bot_y - pad_h - 10)
                          : std::max(top_y + top_h + 4, win_h - pad_h - 10);
    strips.push_back({rect_of(pad_x, pad_y, pad_w, pad_h), 216, 122, 46, 31, false, false, false});

    return strips;
}

bool WorldView::crop_sidebar_for_popup(HudStrip& strip, const WorldRenderData* active_world) {
    if (!active_world || !active_world->has_popup || active_world->popup_src_w <= 0 || active_world->popup_src_h <= 0)
        return false;
    if (active_world->popup_src_y >= 14 || active_world->popup_src_x + active_world->popup_src_w <= 268)
        return false;
    // Advisor / Intelligence Report portraits cover y = 0..78 at the top right, leaving button rows 2..5 (y = 79..158).
    if (strip.src_y < 79 && strip.src_y + strip.src_h > 79) {
        const int orig_h = strip.src_h;
        const int crop_rows = 79 - strip.src_y;
        const int dst_h = int(strip.dst.bottom - strip.dst.top);
        const int new_dst_h = std::max(1, int(std::round(float(dst_h) * float(orig_h - crop_rows) / float(orig_h))));
        strip.src_y = 79;
        strip.src_h = orig_h - crop_rows;
        strip.dst.top = strip.dst.bottom - new_dst_h;
        return true;
    }
    return false;
}

bool WorldView::fit_bottom_strip_for_message(HudStrip& strip, const WorldRenderData* active_world) {
    if (!active_world || !active_world->has_bottom_message || active_world->bottom_msg_src_h <= 0) return false;
    const int fit_h = std::clamp(active_world->bottom_msg_src_h, 13, 28);
    const int orig_h = std::max(1, strip.src_h);
    const int orig_w = std::max(1, strip.src_w);
    const int dst_h = int(strip.dst.bottom - strip.dst.top);
    const int dst_w = int(strip.dst.right - strip.dst.left);
    const int new_dst_h = std::max(1, int(std::round(float(dst_h) * float(fit_h) / float(orig_h))));
    const int inset_x = int(std::round(float(dst_w) * 4.0f / float(orig_w)));
    strip.src_x = 4;
    strip.src_w = 312;
    strip.src_y = 170;
    strip.src_h = fit_h;
    strip.dst.left += inset_x;
    strip.dst.right -= inset_x;
    strip.dst.top = strip.dst.bottom - new_dst_h;
    strip.key_out_backdrop = true;
    return true;
}

RECT WorldView::compute_popup_rect(int win_w, int win_h, int src_w, int src_h, bool hide_bottom_bar, bool hide_top_bar) {
    if (win_w <= 0 || win_h <= 0 || src_w <= 0 || src_h <= 0) return RECT{0, 0, 0, 0};
    const auto strips = compute_hud_strips(win_w, win_h);
    const int top_y = (!hide_top_bar && strips.size() >= 1) ? int(strips[0].dst.bottom) : 0;
    const int bot_y = (!hide_bottom_bar && strips.size() >= 2) ? int(strips[1].dst.top) : win_h;
    const int right_x = strips.size() >= 4 ? int(std::min(strips[2].dst.left, strips[3].dst.left)) : win_w;
    const int margin = 20;
    const int avail_w = std::max(64, right_x - margin * 2);
    const int avail_h = std::max(64, bot_y - top_y - margin * 2);
    // Scale popup cleanly with 4:3 CRT aspect (1.2x vertical) while fitting comfortably inside the open map region.
    float sy = std::clamp(float(win_h) / 215.0f, 2.0f, 5.2f);
    float sx = sy / 1.2f;
    const float fit = std::min({1.0f, float(avail_w) / (float(src_w) * sx), float(avail_h) / (float(src_h) * sy)});
    sx *= fit;
    sy *= fit;
    const int pw = std::max(1, int(std::round(float(src_w) * sx)));
    const int ph = std::max(1, int(std::round(float(src_h) * sy)));
    const int px = margin + (avail_w - pw) / 2;
    const int py = top_y + margin + (avail_h - ph) / 2;
    return RECT{px, py, px + pw, py + ph};
}

void WorldView::compute_projection(int win_w, int win_h, float* scale_x, float* scale_y, float* origin_x, float* origin_y) const {
    const float base_sy = std::max(2.0f, float(win_h) / 225.0f) * zoom_;
    const float base_sx = base_sy / 1.2f;  // exact 4:3 CRT pixel aspect ratio (320x200 -> 4:3)
    const auto strips = compute_hud_strips(win_w, win_h);
    const float top_h = strips.size() >= 1 ? float(strips[0].dst.bottom) : 0.0f;
    const float bot_y = strips.size() >= 2 ? float(strips[1].dst.top) : float(win_h);
    const float right_x = strips.size() >= 4 ? float(strips[3].dst.left) : float(win_w);
    *scale_x = base_sx;
    *scale_y = base_sy;
    *origin_x = right_x * 0.5f;
    *origin_y = (top_h + bot_y) * 0.5f;
}

void WorldView::project_tile(const WorldRenderData& data, float row, float col, int width, int rows, int pixel_dx, int pixel_dy,
                             float* x0, float* y0, float* x1, float* y1) {
    const float dr = row - data.center_row;
    const float dc = col - data.center_col;
    // In Utopia's isometric grid (verified from ds:0x617e), moving +dc shifts (+16, +8) and +dr shifts (-16, +8).
    // A 32x15 ground tile's bottom row lies 7.5 DOS pixels below its diamond centre, and its left edge lies 16 px left.
    const float cx = data.origin_x + ((dc - dr) * 16.0f + float(pixel_dx)) * data.tile_scale_x;
    const float cy = data.origin_y + ((dc + dr) * 8.0f + float(pixel_dy)) * data.tile_scale_y;
    const float bottom_y = cy + 7.5f * data.tile_scale_y;
    *x0 = cx - (float(width) * 0.5f) * data.tile_scale_x;
    *x1 = *x0 + float(width) * data.tile_scale_x;
    *y1 = bottom_y;
    *y0 = bottom_y - float(rows) * data.tile_scale_y;
}

void WorldView::screen_to_map(float px, float py, int win_w, int win_h, float* row_out, float* col_out) const {
    float sx = 1.0f, sy = 1.0f, ox = 0.0f, oy = 0.0f;
    compute_projection(win_w, win_h, &sx, &sy, &ox, &oy);
    const float u = (px - ox) / (16.0f * sx);
    const float v = (py - oy) / (8.0f * sy);
    *col_out = center_col_ + (v + u) * 0.5f;
    *row_out = center_row_ + (v - u) * 0.5f;
}

void WorldView::game_tile_center_dos(int dr, int dc, int* dos_x, int* dos_y) {
    // From u.mdg ds:0x617e: tile (0,0) bottom-left is (112, 38), so its 32x15 diamond centre is (128, 31).
    *dos_x = std::clamp(128 + (dc - dr) * 16, 0, 319);
    *dos_y = std::clamp(31 + (dc + dr) * 8, 0, 199);
}

WorldHit WorldView::hit_test(float px, float py, int win_w, int win_h, int game_cam_row, int game_cam_col,
                             const WorldRenderData* active_world) const {
    WorldHit hit;
    const bool has_popup = active_world && active_world->has_popup && active_world->popup_src_w > 0 && active_world->popup_src_h > 0;
    const bool hide_top = has_popup && (active_world->popup_src_y < 14);
    const bool hide_bottom = (has_popup && (active_world->popup_src_y + active_world->popup_src_h > 170)) ||
                             (active_world && !active_world->has_bottom_message);
    const auto strips = compute_hud_strips(win_w, win_h);
    for (size_t si = 0; si < strips.size(); si++) {
        // Hide scroll-arrow pad while any modal popup is open, hide compass when a portrait popup overlaps it,
        // and hide the top/bottom bars when a tall popup covers them or the bottom message box is empty.
        if (has_popup && si == 4) continue;
        if (has_popup && hide_top && si == 0) continue;
        if (has_popup && hide_top && si == 2) continue;
        if (hide_bottom && si == 1) continue;
        HudStrip s = strips[si];
        if (si == 1) fit_bottom_strip_for_message(s, active_world);
        if (si == 3) crop_sidebar_for_popup(s, active_world);
        if (px >= float(s.dst.left) && px < float(s.dst.right) && py >= float(s.dst.top) && py < float(s.dst.bottom)) {
            const float fx = (px - float(s.dst.left)) / std::max(1.0f, float(s.dst.right - s.dst.left));
            const float fy = (py - float(s.dst.top)) / std::max(1.0f, float(s.dst.bottom - s.dst.top));
            if (s.diamond_clip && std::fabs(fx - 0.5f) + std::fabs(fy - 0.5f) > 0.5f) continue;
            hit.kind = WorldHit::Kind::hud;
            hit.dos_x = std::clamp(s.src_x + int(fx * float(s.src_w)), 0, 319);
            hit.dos_y = std::clamp(s.src_y + int(fy * float(s.src_h)), 0, 199);
            return hit;
        }
        // Opaque full-width header/footer backing bars must block clicks and hover highlights on hidden tiles beneath them.
        if (s.solid_backdrop && py >= float(s.dst.top) && py < float(s.dst.bottom)) return hit;
    }
    if (has_popup) {
        const RECT& pr = active_world->popup_dst;
        if (px >= float(pr.left) && px < float(pr.right) && py >= float(pr.top) && py < float(pr.bottom)) {
            const float fx = (px - float(pr.left)) / std::max(1.0f, float(pr.right - pr.left));
            const float fy = (py - float(pr.top)) / std::max(1.0f, float(pr.bottom - pr.top));
            const int dos_x = std::clamp(active_world->popup_src_x + int(fx * float(active_world->popup_src_w)), 0, 319);
            const int dos_y = std::clamp(active_world->popup_src_y + int(fy * float(active_world->popup_src_h)), 0, 199);
            // Ignore clicks falling inside the transparent L-cutout beneath an advisor portrait so they cannot hit hidden buttons.
            if (active_world->popup_has_holes && dos_x >= 216 && dos_y >= 80) return hit;
            hit.kind = WorldHit::Kind::hud;
            hit.dos_x = dos_x;
            hit.dos_y = dos_y;
        }
        return hit;
    }
    float frow = 0.0f, fcol = 0.0f;
    screen_to_map(px, py, win_w, win_h, &frow, &fcol);
    const int tr = int(std::floor(frow + 0.5f));
    const int tc = int(std::floor(fcol + 0.5f));
    if (tr < 0 || tr >= kMapRows || tc < 0 || tc >= kMapCols) return hit;

    hit.kind = WorldHit::Kind::tile;
    hit.tile_row = tr;
    hit.tile_col = tc;
    int cam_r = std::clamp(game_cam_row, 0, kMaxGameCamRow);
    int cam_c = std::clamp(game_cam_col, 0, kMaxGameCamCol);
    // Keep at least a 1-tile inner margin when shifting Utopia's 8x8 logic window so 2x2 building footprints fit.
    const bool inside = tr >= cam_r && tr < cam_r + kGameViewSize && tc >= cam_c && tc < cam_c + kGameViewSize;
    if (!inside) {
        cam_r = std::clamp(tr - 3, 0, kMaxGameCamRow);
        cam_c = std::clamp(tc - 3, 0, kMaxGameCamCol);
        hit.need_cam_shift = true;
        hit.shift_cam_row = cam_r;
        hit.shift_cam_col = cam_c;
    }
    game_tile_center_dos(tr - cam_r, tc - cam_c, &hit.dos_x, &hit.dos_y);
    return hit;
}

void WorldView::visible_bounds(int win_w, int win_h, int* r0, int* r1, int* c0, int* c1) const {
    float rs[4], cs[4];
    screen_to_map(0.0f, 0.0f, win_w, win_h, &rs[0], &cs[0]);
    screen_to_map(float(win_w), 0.0f, win_w, win_h, &rs[1], &cs[1]);
    screen_to_map(0.0f, float(win_h), win_w, win_h, &rs[2], &cs[2]);
    screen_to_map(float(win_w), float(win_h), win_w, win_h, &rs[3], &cs[3]);
    const float min_r = std::min({rs[0], rs[1], rs[2], rs[3]});
    const float max_r = std::max({rs[0], rs[1], rs[2], rs[3]});
    const float min_c = std::min({cs[0], cs[1], cs[2], cs[3]});
    const float max_c = std::max({cs[0], cs[1], cs[2], cs[3]});
    // Include 3 tiles of margin so tall buildings (30 px high) near the bottom edge still draw their roofs.
    *r0 = std::clamp(int(std::floor(min_r)) - 3, 0, kMapRows - 1);
    *r1 = std::clamp(int(std::ceil(max_r)) + 4, 0, kMapRows - 1);
    *c0 = std::clamp(int(std::floor(min_c)) - 3, 0, kMapCols - 1);
    *c1 = std::clamp(int(std::ceil(max_c)) + 4, 0, kMapCols - 1);
}

}  // namespace nax
