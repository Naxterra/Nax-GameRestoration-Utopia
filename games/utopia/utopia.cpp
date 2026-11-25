// Utopia: The Creation of a Nation (Gremlin Graphics, 1992).
//
// The original launcher UTOPIA.EXE asks for graphics, sound and language and then starts I.MDG (the intro)
// and U.MDG (the game) with a six-character command tail: graphics letter, sound letter, language letter,
// "M", the video mode to restore on exit, and "P" for the game or a space for the intro. This file takes the
// place of the launcher. The translated modules are generated from the user's own copy at build time.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/game.h"
#include "runtime/hd_sprites.h"
#include "runtime/world_view.h"

const nax::ModuleInfo* nax_module_i();
const nax::ModuleInfo* nax_module_u();
const nax::ModuleInfo* nax_module_beeper();
const nax::ModuleInfo* nax_module_adlib();
const nax::ModuleInfo* nax_module_mt32();

// The hooks listed in hooks-u.txt. U.MDG runs with CS = load segment + 0FFEh, and it draws every frame into a
// 320x200 buffer at load segment + 5872h, which it then copies to the screen as a whole.
namespace {

uint16_t load_base(const nax::Machine& m) { return uint16_t(m.c.cs - 0x0ffe); }
uint16_t picture_buffer(const nax::Machine& m) { return uint16_t(load_base(m) + 0x5872); }

struct DynamicSprite {
    uint32_t data = 0;
    int width = 32, rows = 15;
    uint8_t base = 0;
    bool bottom_up = true;
    int dos_x = 0, dos_y = 0;  // bottom-left pixel (if bottom_up) or top-left pixel (if !bottom_up) in 320x200
    int cam_row = 0, cam_col = 0;
};

struct WorldBridge {
    std::mutex mutex;
    std::atomic<bool> in_game{false};
    std::atomic<uint16_t> u_base{0};
    bool back_has_map = false;
    bool front_has_map = false;
    bool back_popup = false;
    bool front_popup = false;
    int back_popup_x0 = 320, back_popup_y0 = 200, back_popup_x1 = 0, back_popup_y1 = 0;
    int front_popup_x0 = 320, front_popup_y0 = 200, front_popup_x1 = 0, front_popup_y1 = 0;
    bool front_has_cam = false;
    int front_cam_row = 0, front_cam_col = 0;
    std::vector<DynamicSprite> back_dynamics;
    std::vector<DynamicSprite> front_dynamics;

    // Pending UI-thread requests applied safely on the game thread at 0ffe:1085.
    bool has_cursor_req = false;
    int req_cursor_x = 0, req_cursor_y = 0;
    bool has_cam_req = false;
    int req_cam_row = 0, req_cam_col = 0;
    bool has_btn_req = false;
    int req_buttons = 0;
} g_world;

void record_dynamic_sprite(const nax::Machine& m, uint32_t data, int width, int rows, uint16_t dest_off, uint8_t base,
                           bool bottom_up) {
    if (width <= 0 || width > 256 || rows <= 0 || rows > 256) return;
    const uint16_t b = load_base(m);
    const uint16_t ds = uint16_t(b + 0x0020);
    const int dos_y = int(dest_off / 320);
    const int dos_x = int(dest_off % 320);
    // Only record sprites drawn inside the isometric map diamond area (not sidebar buttons or HUD).
    if (dos_x >= 256 || dos_y >= 165) return;
    const int cam_col = std::clamp(int(m.rd16(ds, 0x2ce9)), 0, nax::WorldView::kMaxGameCamCol);
    const int cam_row = std::clamp(int(m.rd16(ds, 0x2ceb)), 0, nax::WorldView::kMaxGameCamRow);
    std::lock_guard<std::mutex> lock(g_world.mutex);
    if (g_world.back_dynamics.size() < 256)
        g_world.back_dynamics.push_back({data, width, rows, base, bottom_up, dos_x, dos_y, cam_row, cam_col});
}

// Each of the game's three map sprite routines has set up a sprite: DS:SI points to its 4-bit pixels, DS:BX to
// its entry in the sprite table (whose first byte is the colour base), and DI is where its bottom row starts.
// The sprites are 32 pixels wide.
void report_sprite(nax::Machine& m, uint16_t rows, uint16_t target_segment, bool is_overlay) {
    if (!m.hd) return;
    const uint16_t base_seg = load_base(m);
    g_world.u_base.store(base_seg, std::memory_order_release);
    const uint8_t colour_base = m.rd8(m.c.ds, m.c.b.x);
    const uint32_t data = nax::Machine::linear(m.c.ds, m.c.si);
    if (target_segment == picture_buffer(m)) {
        m.hd->set_screen_source(nax::Machine::linear(target_segment, 0));
        if (is_overlay) {
            record_dynamic_sprite(m, data, 32, rows, m.c.di, colour_base, true);
        } else {
            std::lock_guard<std::mutex> lock(g_world.mutex);
            g_world.back_has_map = true;
        }
    }
    m.hd->draw(m, data, 32, rows, nax::Machine::linear(target_segment, m.c.di), colour_base);
}

// The game unpacks its screen pictures (the stone backdrop, options, advisors, maps and panels) with one routine,
// from DS:SI to ES:DI. Full lines unpacked into either the picture buffer or directly into VGA memory are reported.
uint16_t unpack_segment = 0, unpack_start = 0;
uint16_t intro_unpack_segment = 0, intro_unpack_start = 0;

void decode_4bit_rgba(const nax::Machine& m, uint32_t data, int width, int rows, uint8_t base, bool bottom_up,
                      std::vector<uint32_t>* out) {
    out->resize(size_t(width) * size_t(rows));
    const uint32_t stride = uint32_t(width / 2);
    auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
    for (int y = 0; y < rows; y++) {
        const uint32_t row = data + uint32_t(bottom_up ? rows - 1 - y : y) * stride;
        for (int x = 0; x < width; x++) {
            const uint8_t byte = m.mem[(row + uint32_t(x / 2)) & 0xFFFFF];
            const uint8_t c = (x & 1) ? (byte & 0x0F) : (byte >> 4);
            if (!c) {
                (*out)[size_t(y) * size_t(width) + size_t(x)] = 0;
                continue;
            }
            const uint8_t* d = m.dac[uint8_t(base + c)];
            (*out)[size_t(y) * size_t(width) + size_t(x)] =
                expand(d[0]) | (expand(d[1]) << 8) | (expand(d[2]) << 16) | 0xFF000000u;
        }
    }
}

}  // namespace

void utopia_sprite_rows_in_cx(nax::Machine& m) { report_sprite(m, m.c.c.x, m.c.es, false); }
void utopia_sprite_rows_in_cx_es_in_ax(nax::Machine& m) { report_sprite(m, m.c.c.x, m.c.a.x, false); }
void utopia_sprite_rows_in_bp(nax::Machine& m) { report_sprite(m, m.c.bp, m.c.es, true); }
void utopia_sprite_dynamic(nax::Machine& m) {
    if (!m.hd) return;
    const int16_t step = static_cast<int16_t>(m.rd16(m.c.cs, 0xf4ab));
    if (step != 320 && step != -320) return;
    const uint32_t data = nax::Machine::linear(m.c.ds, m.c.si);
    const bool bottom_up = step < 0;
    if (m.c.es == picture_buffer(m)) {
        m.hd->set_screen_source(nax::Machine::linear(m.c.es, 0));
        record_dynamic_sprite(m, data, 32, m.c.c.x, m.c.di, 0, bottom_up);
    }
    m.hd->draw(m, data, 32, m.c.c.x, nax::Machine::linear(m.c.es, m.c.di), 0, bottom_up, false);
}
void utopia_button_buffer(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->set_screen_source(nax::Machine::linear(m.c.es, 0));
    m.hd->draw(m, nax::Machine::linear(m.c.ds, m.c.si), 16, 16, nax::Machine::linear(m.c.es, m.c.di), 0x00, false, false);
}
void utopia_button_vram(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->draw(m, nax::Machine::linear(m.c.ds, m.c.si), 16, 16, nax::Machine::linear(m.c.es, m.c.di), 0x10, false, false);
}
void utopia_begin_present(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->begin_present(m);
}
void utopia_present_buffer(nax::Machine& m) {
    if (!m.hd) return;
    const uint16_t b = load_base(m);
    g_world.u_base.store(b, std::memory_order_release);
    const uint16_t ds = uint16_t(b + 0x0020);
    const int cam_col = int(m.rd16(ds, 0x2ce9));
    const int cam_row = int(m.rd16(ds, 0x2ceb));
    {
        std::lock_guard<std::mutex> lock(g_world.mutex);
        g_world.front_has_map = g_world.back_has_map;
        g_world.front_popup = g_world.back_popup;
        g_world.front_popup_x0 = g_world.back_popup_x0;
        g_world.front_popup_y0 = g_world.back_popup_y0;
        g_world.front_popup_x1 = g_world.back_popup_x1;
        g_world.front_popup_y1 = g_world.back_popup_y1;
        g_world.front_dynamics = g_world.back_dynamics;
        if (cam_row >= 0 && cam_row <= nax::WorldView::kMaxGameCamRow &&
            cam_col >= 0 && cam_col <= nax::WorldView::kMaxGameCamCol) {
            g_world.front_has_cam = true;
            g_world.front_cam_row = cam_row;
            g_world.front_cam_col = cam_col;
        }
    }
    m.hd->present_buffer(nax::Machine::linear(picture_buffer(m), 0));
}
void utopia_panel_fill_8_buffer(nax::Machine& m) {
    if (!m.hd) return;
    const int px = int(m.c.di % 320);
    const int py = int(m.c.di / 320);
    const int pw = int(m.c.bp) * 8;
    const int ph = 8;
    if (pw <= 0 || px >= 256 || py + ph <= 14) return;
    // Below y = 171 lies the persistent bottom message bar unless a tall popup (such as SELECT A BUILDING..)
    // started above y = 171 and continues down to y = 190 with matching horizontal bounds.
    bool is_popup_row = py < 171;
    if (!is_popup_row) {
        std::lock_guard<std::mutex> lock(g_world.mutex);
        if (g_world.back_popup && px == g_world.back_popup_x0 && px + pw == g_world.back_popup_x1 &&
            py <= g_world.back_popup_y1 + 2) {
            is_popup_row = true;
        }
    }
    if (!is_popup_row) return;
    {
        std::lock_guard<std::mutex> lock(g_world.mutex);
        if (!g_world.back_popup) {
            g_world.back_popup = true;
            g_world.back_popup_x0 = px;
            g_world.back_popup_y0 = py;
            g_world.back_popup_x1 = px + pw;
            g_world.back_popup_y1 = py + ph;
        } else {
            g_world.back_popup_x0 = std::min(g_world.back_popup_x0, px);
            g_world.back_popup_y0 = std::min(g_world.back_popup_y0, py);
            g_world.back_popup_x1 = std::max(g_world.back_popup_x1, px + pw);
            g_world.back_popup_y1 = std::max(g_world.back_popup_y1, py + ph);
        }
    }
    m.hd->clear_rect(nax::Machine::linear(picture_buffer(m), m.c.di), pw, ph);
}
void utopia_panel_fill_8_buffer_cell(nax::Machine& m) {
    if (!m.hd || m.c.di < 171u * 320u) return;
    {
        const int px = int(m.c.di % 320);
        const int py = int(m.c.di / 320);
        std::lock_guard<std::mutex> lock(g_world.mutex);
        if (g_world.back_popup && px >= g_world.back_popup_x0 && px < g_world.back_popup_x1 &&
            py >= g_world.back_popup_y0 && py < g_world.back_popup_y1)
            return;
    }
    // Outside an active popup, unmask any leftover tall-popup bits on the bottom message bar before syncing.
    const uint32_t dest = nax::Machine::linear(picture_buffer(m), m.c.di);
    m.hd->restore_rect(dest, 8, 8);
    m.hd->sync_image_rect(m, dest, 8, 8);
}
void utopia_vram_text_clear_8x8(nax::Machine& m) {
    if (!m.hd) return;
    const int px = int(m.c.di % 320);
    const int py = int(m.c.di / 320);
    // Routine 0ffe:720f/736f draws both direct-VRAM modal dialog panels over the live colony map (when
    // screen_source != 0xA0000) and text clears on direct-VRAM full-screen menus (such as option.vgg).
    if (m.hd->screen_source() != 0xA0000u && px < 256 && py >= 14 && py < 171) {
        m.hd->clear_rect(nax::Machine::linear(0xA000, m.c.di), 8, 8);
        return;
    }
    m.hd->sync_image_rect(m, nax::Machine::linear(0xA000, m.c.di), 8, 8);
}
void utopia_vram_text_clear_8x6(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->sync_image_rect(m, nax::Machine::linear(0xA000, m.c.di), 8, 6);
}
void utopia_vram_cell_8x8(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->clear_rect(nax::Machine::linear(0xA000, m.c.di), 8, 8);
}
void utopia_vram_cell_8x8_unshifted(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->clear_rect(nax::Machine::linear(0xA000, uint16_t(m.c.di << 3)), 8, 8);
}
void utopia_vram_planet_map(nax::Machine& m) {
    if (!m.hd) return;
    m.hd->clear_rect(nax::Machine::linear(0xA000, 0x0a08), 200, 150);
}
void utopia_restore_map_backdrop(nax::Machine& m) {
    if (!m.hd) return;
    bool was_popup = false;
    int prev_y1 = 0;
    {
        std::lock_guard<std::mutex> lock(g_world.mutex);
        was_popup = g_world.back_popup;
        prev_y1 = g_world.back_popup_y1;
        g_world.back_popup = false;
        g_world.back_popup_x0 = 320;
        g_world.back_popup_y0 = 200;
        g_world.back_popup_x1 = 0;
        g_world.back_popup_y1 = 0;
        g_world.back_dynamics.clear();
    }
    const uint16_t buf = picture_buffer(m);
    m.hd->restore_backdrop_rect(nax::Machine::linear(buf, 0), 256, 159);
    // Tall popups such as SELECT A BUILDING.. extend below y = 158 down to y = 190; unmask those lower rows
    // too so closing the popup never leaves orphaned 0x80000000u mask bits in front_image_marks_.
    if (was_popup && prev_y1 > 159) {
        const int extra_h = std::min(41, prev_y1 - 159);
        m.hd->restore_rect(nax::Machine::linear(buf, uint16_t(159 * 320)), 256, extra_h);
    }
}
void utopia_unpack_start(nax::Machine& m) {
    unpack_segment = m.c.es;
    unpack_start = m.c.di;
}
void utopia_unpack_end(nax::Machine& m) {
    if (!m.hd || m.c.es != unpack_segment) return;
    const bool to_buffer = unpack_segment == picture_buffer(m);
    const bool to_vram = unpack_segment == 0xA000;
    if (!to_buffer && !to_vram) return;
    const uint16_t length = uint16_t(m.c.di - unpack_start);
    if (length == 0 || length % 320 != 0 || unpack_start % 320 != 0) return;
    if (to_buffer) {
        std::lock_guard<std::mutex> lock(g_world.mutex);
        g_world.back_has_map = false;
        g_world.back_popup = false;
        g_world.back_popup_x0 = 320;
        g_world.back_popup_y0 = 200;
        g_world.back_popup_x1 = 0;
        g_world.back_popup_y1 = 0;
        g_world.back_dynamics.clear();
    }
    m.hd->set_screen_source(nax::Machine::linear(unpack_segment, 0));
    const uint32_t at = nax::Machine::linear(unpack_segment, unpack_start);
    m.hd->image(m, at, 320, length / 320, 320, at, 320);
}
void utopia_intro_unpack_start(nax::Machine& m) {
    intro_unpack_segment = m.c.es;
    intro_unpack_start = m.c.di;
}
void utopia_intro_unpack_end(nax::Machine& m) {
    if (!m.hd || m.c.es != intro_unpack_segment || intro_unpack_segment != 0xA000) return;
    const uint16_t length = uint16_t(m.c.di - intro_unpack_start);
    if (length == 0 || length % 320 != 0 || intro_unpack_start % 320 != 0) return;
    m.hd->set_screen_source(0xA0000);
    const uint32_t at = nax::Machine::linear(intro_unpack_segment, intro_unpack_start);
    m.hd->image(m, at, 320, length / 320, 320, at, 320);
}
void utopia_after_mouse_poll(nax::Machine& m) {
    std::lock_guard<std::mutex> lock(g_world.mutex);
    const bool set_cur = g_world.has_cursor_req;
    const bool set_cam = g_world.has_cam_req;
    const bool have_btn = g_world.has_btn_req;
    if (!set_cur && !set_cam && !have_btn) return;
    const uint16_t ds = uint16_t(load_base(m) + 0x0020);
    if (set_cur) {
        m.wr16(ds, 0x6590, uint16_t(std::clamp(g_world.req_cursor_x, 0, 319)));
        m.wr16(ds, 0x6592, uint16_t(std::clamp(g_world.req_cursor_y, 0, 199)));
        g_world.has_cursor_req = false;
    }
    if (set_cam) {
        const uint16_t applied_cr = uint16_t(std::clamp(g_world.req_cam_row, 0, nax::WorldView::kMaxGameCamRow));
        const uint16_t applied_cc = uint16_t(std::clamp(g_world.req_cam_col, 0, nax::WorldView::kMaxGameCamCol));
        g_world.has_cam_req = false;
        if (m.rd16(ds, 0x2ceb) != applied_cr || m.rd16(ds, 0x2ce9) != applied_cc) {
            m.wr16(ds, 0x2ce9, applied_cc);
            m.wr16(ds, 0x2ceb, applied_cr);
            // Bit 3 of ds:[0x2e1d] is Utopia's native map-redraw dirty flag (same as 0ffe:05c6 after scrolling).
            m.wr8(ds, 0x2e1d, uint8_t(m.rd8(ds, 0x2e1d) | 0x08));
        }
        g_world.front_has_cam = true;
        g_world.front_cam_row = int(applied_cr);
        g_world.front_cam_col = int(applied_cc);
    }
    if (have_btn) {
        m.mouse.buttons = g_world.req_buttons;
        g_world.has_btn_req = false;
    }
}

namespace game {

const wchar_t* title() { return L"Utopia: The Creation of a Nation"; }
const wchar_t* data_folder_name() { return L"utopia"; }
const wchar_t* required_file() { return L"u.mdg"; }

void register_modules(nax::Machine& m) {
    for (auto info : {nax_module_i(), nax_module_u(), nax_module_beeper(), nax_module_adlib(), nax_module_mt32()}) m.add_module(info);
}

int run(nax::Machine& m, const std::map<std::wstring, std::wstring>& options) {
    auto option = [&](const wchar_t* key, const wchar_t* fallback) {
        auto it = options.find(key);
        return it == options.end() ? std::wstring(fallback) : it->second;
    };
    const std::wstring sound = option(L"sound", L"adlib");
    const std::wstring language = option(L"language", L"en");
    const char sound_letter = sound == L"beeper" ? 'B' : (sound == L"roland" ? 'R' : 'A');
    const char language_letter = language == L"fr" ? 'f' : (language == L"de" ? 'g' : (language == L"it" ? 'i' : 'e'));
    std::string tail = {'V', sound_letter, language_letter, 'M', char(3), ' '};
    m.log("Utopia: sound %c, language %c", sound_letter, language_letter);
    g_world.in_game.store(false, std::memory_order_release);
    if (!options.count(L"skip-intro")) {
        m.run_program(nax_module_i(), tail);
        if (m.hd) m.hd->present_buffer(0);
        if (m.stopped()) return -1;
    }
    tail[5] = 'P';
    g_world.in_game.store(true, std::memory_order_release);
    const int rc = m.run_program(nax_module_u(), tail);
    g_world.in_game.store(false, std::memory_order_release);
    return rc;
}

void queue_world_input(bool set_cursor, int cursor_x, int cursor_y, bool set_cam, int cam_row, int cam_col, int set_buttons) {
    std::lock_guard<std::mutex> lock(g_world.mutex);
    // Never let a cursor-only hover/park update clobber the coordinates of an unconsumed click or camera shift.
    if (set_cursor && (set_cam || set_buttons >= 0 || (!g_world.has_cam_req && !g_world.has_btn_req))) {
        g_world.has_cursor_req = true;
        g_world.req_cursor_x = cursor_x;
        g_world.req_cursor_y = cursor_y;
    }
    if (set_cam) {
        g_world.has_cam_req = true;
        g_world.req_cam_row = cam_row;
        g_world.req_cam_col = cam_col;
    }
    if (set_buttons >= 0) {
        g_world.has_btn_req = true;
        g_world.req_buttons = set_buttons;
    }
}

void release_world_buttons(nax::Machine& m, int clear_bits) {
    std::lock_guard<std::mutex> lock(g_world.mutex);
    if (g_world.has_btn_req) {
        g_world.req_buttons &= ~clear_bits;
        if (g_world.req_buttons == 0) g_world.has_btn_req = false;
    }
    m.mouse.buttons &= ~clear_bits;
}

bool erase_vram_cursor(const nax::Machine& m, uint8_t* vram_copy) {
    if (!vram_copy || !g_world.in_game.load(std::memory_order_acquire)) return false;
    const uint16_t base_seg = g_world.u_base.load(std::memory_order_acquire);
    if (!base_seg) return false;
    const uint16_t ds = uint16_t(base_seg + 0x0020);
    const uint16_t saved_off = m.rd16(ds, 0x5e3a);
    if (saved_off == 0xFFFF || saved_off >= 64000) return false;
    const int y0 = int(saved_off / 320), x0 = int(saved_off % 320);
    bool changed = false;
    for (int row = 0; row < 10; row++) {
        const int y = y0 + row;
        if (y >= 200) break;
        for (int col = 0; col < 10; col++) {
            const int x = x0 + col;
            if (x >= 320) break;
            const uint16_t idx = uint16_t(row * 10 + col);
            const uint8_t cur_px = m.rd8(ds, uint16_t(0x5e68 + idx));
            const size_t off = size_t(y) * 320 + size_t(x);
            if (cur_px != 0 && vram_copy[off] == cur_px) {
                vram_copy[off] = m.rd8(ds, uint16_t(0x5f6c + idx));
                changed = true;
            }
        }
    }
    return changed;
}

bool build_world_render(const nax::Machine& m, nax::HdSprites& hd, nax::WorldView& view, nax::WorldFallbackAtlas& fallback,
                        int win_w, int win_h, nax::WorldRenderData* out) {
    if (!out || win_w <= 0 || win_h <= 0 || m.video_mode != 0x13) return false;
    if (!g_world.in_game.load(std::memory_order_acquire)) return false;
    const uint16_t base_seg = g_world.u_base.load(std::memory_order_acquire);
    if (!base_seg) return false;
    int vx0 = 0, vy0 = 0, vx1 = 0, vy1 = 0;
    bool pop_holes = false;
    if (!hd.is_showing_offscreen_map(&vx0, &vy0, &vx1, &vy1, &pop_holes)) return false;

    std::vector<DynamicSprite> dynamics;
    bool have_cam = false;
    bool has_front_popup = false;
    int fpx0 = 320, fpy0 = 200, fpx1 = 0, fpy1 = 0;
    int game_cam_row = 0, game_cam_col = 0;
    {
        std::lock_guard<std::mutex> lock(g_world.mutex);
        if (!g_world.front_has_map) return false;
        dynamics = g_world.front_dynamics;
        have_cam = g_world.front_has_cam;
        game_cam_row = g_world.front_cam_row;
        game_cam_col = g_world.front_cam_col;
        has_front_popup = g_world.front_popup;
        fpx0 = g_world.front_popup_x0;
        fpy0 = g_world.front_popup_y0;
        fpx1 = g_world.front_popup_x1;
        fpy1 = g_world.front_popup_y1;
    }

    const uint16_t ds = uint16_t(base_seg + 0x0020);
    if (!have_cam) {
        game_cam_col = int(m.rd16(ds, 0x2ce9));
        game_cam_row = int(m.rd16(ds, 0x2ceb));
    }
    if (game_cam_col < 0 || game_cam_col > nax::WorldView::kMaxGameCamCol ||
        game_cam_row < 0 || game_cam_row > nax::WorldView::kMaxGameCamRow)
        return false;

    view.sync_game_camera(game_cam_row, game_cam_col);
    hd.begin_world_frame();
    out->win_w = win_w;
    out->win_h = win_h;
    out->center_row = view.center_row();
    out->center_col = view.center_col();
    view.compute_projection(win_w, win_h, &out->tile_scale_x, &out->tile_scale_y, &out->origin_x, &out->origin_y);
    hd.ambient_tint(out->ambient_tint);
    out->hud_strips = nax::WorldView::compute_hud_strips(win_w, win_h);
    out->fallback = &fallback;

    // Combine offscreen and direct-VRAM popup bounding boxes so dialogs like SELECT A BUILDING.. and MISSILE LAUNCHER
    // float cleanly over the live ultrawide colony instead of snapping back to the pillarboxed 4:3 screen.
    int pop_x0 = 320, pop_y0 = 200, pop_x1 = 0, pop_y1 = 0;
    if (has_front_popup && fpx1 > fpx0 && fpy1 > fpy0) {
        pop_x0 = std::min(pop_x0, fpx0);
        pop_y0 = std::min(pop_y0, fpy0);
        pop_x1 = std::max(pop_x1, fpx1);
        pop_y1 = std::max(pop_y1, fpy1);
    }
    if (vx1 > vx0 && vy1 > vy0) {
        pop_x0 = std::min(pop_x0, vx0);
        pop_y0 = std::min(pop_y0, vy0);
        pop_x1 = std::max(pop_x1, vx1);
        pop_y1 = std::max(pop_y1, vy1);
    }
    if (pop_x1 > pop_x0 && pop_y1 > pop_y0) {
        // Expand an edge by 1 px only when that outer line in VRAM is a genuine solid black dialog border,
        // keeping Utopia's crisp outer frame while never grabbing a 1 px sliver of main_scr.vgg stone floor.
        const uint8_t* vram = m.vram();
        auto is_dark = [&](int x, int y) {
            const uint8_t c = vram[size_t(y) * 320 + size_t(x)];
            return m.dac[c][0] < 6 && m.dac[c][1] < 6 && m.dac[c][2] < 6;
        };
        auto row_is_border = [&](int y) {
            if (y < 14 || y >= 196 || pop_x1 - pop_x0 < 12) return false;
            int dark = 0, total = 0;
            for (int x = pop_x0 + 4; x < pop_x1 - 4; x += 2) {
                total++;
                if (is_dark(x, y)) dark++;
            }
            return total > 0 && dark * 5 >= total * 4;
        };
        auto col_is_border = [&](int x) {
            if (x < 0 || x >= 256 || pop_y1 - pop_y0 < 12) return false;
            int dark = 0, total = 0;
            for (int y = pop_y0 + 4; y < pop_y1 - 4; y += 2) {
                total++;
                if (is_dark(x, y)) dark++;
            }
            return total > 0 && dark * 5 >= total * 4;
        };
        if (pop_y0 > 14 && row_is_border(pop_y0 - 1)) pop_y0--;
        if (pop_y1 < 196 && row_is_border(pop_y1)) pop_y1++;
        if (pop_x0 > 0 && col_is_border(pop_x0 - 1)) pop_x0--;
        if (pop_x1 < 256 && col_is_border(pop_x1)) pop_x1++;
    }
    if (pop_x1 > pop_x0 && pop_y1 > pop_y0) {
        out->has_popup = true;
        out->popup_has_holes = pop_holes;
        out->popup_src_x = pop_x0;
        out->popup_src_y = pop_y0;
        out->popup_src_w = pop_x1 - pop_x0;
        out->popup_src_h = pop_y1 - pop_y0;
        const bool hide_top = pop_y0 < 14;
        const bool hide_bottom = pop_y1 > 170;
        out->popup_dst = nax::WorldView::compute_popup_rect(win_w, win_h, out->popup_src_w, out->popup_src_h, hide_bottom, hide_top);
    } else {
        out->has_popup = false;
        out->popup_has_holes = false;
        out->popup_src_x = out->popup_src_y = out->popup_src_w = out->popup_src_h = 0;
        out->popup_dst = RECT{0, 0, 0, 0};
    }

    // Auto-hide the empty bottom message console so it does not leave a large blank black box over the ultrawide
    // colony map, and size it tightly to active text rows (with 250 ms hold hysteresis) whenever Utopia prints a message.
    // Suppress and clear the hold timer whenever a tall popup overlaps y >= 171 so popup pixels cannot cause a flash on close.
    {
        static ULONGLONG s_last_msg_ms = 0;
        static int s_last_msg_h = 0;
        if (out->has_popup && (out->popup_src_y + out->popup_src_h) > 170) {
            s_last_msg_h = 0;
            out->has_bottom_message = false;
            out->bottom_msg_src_h = 0;
        } else {
            const uint8_t* vram = m.vram();
            const uint16_t cur_off = m.rd16(ds, 0x5e3a);
            const int cur_x = (cur_off != 0xFFFF && cur_off < 64000) ? int(cur_off % 320) : -32;
            const int cur_y = (cur_off != 0xFFFF && cur_off < 64000) ? int(cur_off / 320) : -32;
            int bright_px = 0;
            int max_text_y = 170;
            for (int y = 171; y <= 196; y++) {
                const size_t row_off = size_t(y) * 320;
                for (int x = 8; x <= 311; x++) {
                    if (x >= cur_x && x < cur_x + 10 && y >= cur_y && y < cur_y + 10) continue;
                    const uint8_t c = vram[row_off + size_t(x)];
                    // Empty main_scr.vgg stipple peaks at DAC 14; active yellow/silver/coloured text is >= 22 even at night.
                    const uint8_t hi = std::max({m.dac[c][0], m.dac[c][1], m.dac[c][2]});
                    if (hi >= 22) {
                        bright_px++;
                        if (y > max_text_y) max_text_y = y;
                    }
                }
            }
            const ULONGLONG now_ms = GetTickCount64();
            if (bright_px >= 6) {
                const int fit_h = std::clamp((max_text_y - 170) + 4, 13, 28);
                s_last_msg_ms = now_ms;
                s_last_msg_h = fit_h;
                out->has_bottom_message = true;
                out->bottom_msg_src_h = fit_h;
            } else if (s_last_msg_h > 0 && (now_ms - s_last_msg_ms) < 250) {
                out->has_bottom_message = true;
                out->bottom_msg_src_h = s_last_msg_h;
            } else {
                s_last_msg_h = 0;
                out->has_bottom_message = false;
                out->bottom_msg_src_h = 0;
            }
        }
    }

    // Scale the native game cursor proportionally to the docked HUD scale.
    if (!out->hud_strips.empty() && out->hud_strips[0].src_w > 0 && out->hud_strips[0].src_h > 0) {
        out->cursor_scale_x = float(out->hud_strips[0].dst.right - out->hud_strips[0].dst.left) / float(out->hud_strips[0].src_w);
        out->cursor_scale_y = float(out->hud_strips[0].dst.bottom - out->hud_strips[0].dst.top) / float(out->hud_strips[0].src_h);
    } else {
        out->cursor_scale_x = out->tile_scale_x;
        out->cursor_scale_y = out->tile_scale_y;
    }

    // Pre-clear the fallback atlas before building the frame when free height is low, and also retry once if an
    // unexpected mid-frame wrap clears earlier fallback_id entries so no tile ever draws a stale atlas slot.
    if (fallback.free_rows() < 96) fallback.clear();

    // Determine if Utopia has an active blinking selection tile (ds:[0x2d94] holds packed (local_col << 8) | local_row,
    // or 0xFFFF when unselected; bit 1 of ds:[0x2e1d] is the blink phase).
    const uint16_t sel_word = m.rd16(ds, 0x2d94);
    const bool blink_on = (m.rd8(ds, 0x2e1d) & 0x02) != 0;
    int sel_row = -1, sel_col = -1;
    if (blink_on) {
        const int local_row = int(sel_word & 0xFF);
        const int local_col = int(sel_word >> 8);
        if (local_row < nax::WorldView::kGameViewSize && local_col < nax::WorldView::kGameViewSize) {
            sel_row = game_cam_row + local_row;
            sel_col = game_cam_col + local_col;
        }
    }

    int r0 = 0, r1 = 0, c0 = 0, c1 = 0;
    view.visible_bounds(win_w, win_h, &r0, &r1, &c0, &c1);

    struct CachedTile {
        int rows = 15;
        nax::HdSprites::WorldSpriteInfo info;
        uint32_t fallback_id = 0;
        bool valid = false;
    };
    std::vector<uint32_t> rgba_tmp;

    for (int attempt = 0; attempt < 2; attempt++) {
        const uint64_t clears_before = fallback.clear_count();
        std::unordered_map<uint16_t, CachedTile> tile_cache;
        out->tiles.clear();

        auto resolve_tid = [&](uint16_t tid) -> const CachedTile& {
            auto it = tile_cache.find(tid);
            if (it != tile_cache.end()) return it->second;
            CachedTile ct;
            if (tid > 0 && tid <= 0x3ff) {
                uint16_t bx = uint16_t(((tid - 1) & 0x3ff) << 3);
                uint16_t seg = uint16_t(base_seg + 0x276b);
                if (bx > 0x2f0) {
                    bx = uint16_t(bx - 0x2f8);
                    seg = uint16_t(base_seg + 0x31db);
                }
                const uint8_t colour_base = m.rd8(seg, bx);
                const int rows = int(m.rd16(seg, uint16_t(bx + 2))) + 1;
                const uint16_t data_off = m.rd16(seg, uint16_t(bx + 6));
                if (rows > 0 && rows <= 64) {
                    const uint32_t data_lin = nax::Machine::linear(seg, data_off);
                    ct.rows = rows;
                    ct.info = hd.resolve_world_sprite(m, data_lin, 32, rows, colour_base, true);
                    if (!ct.info.placed) {
                        decode_4bit_rgba(m, data_lin, 32, rows, colour_base, true, &rgba_tmp);
                        ct.fallback_id = fallback.get_or_add(ct.info.key, 32, rows, rgba_tmp.data());
                    }
                    ct.valid = true;
                }
            }
            return tile_cache.emplace(tid, ct).first->second;
        };

        // Build visible planet map tiles in strict back-to-front isometric painter's order (diagonal sum r + c).
        out->tiles.reserve(size_t((r1 - r0 + 1) * (c1 - c0 + 1)) + dynamics.size());
        for (int sum = r0 + c0; sum <= r1 + c1; sum++) {
            const int r_lo = std::max(r0, sum - c1);
            const int r_hi = std::min(r1, sum - c0);
            for (int r = r_lo; r <= r_hi; r++) {
                const int c = sum - r;
                const uint16_t off = uint16_t(0xbaa8 + r * 0xcc + c * 2);
                uint16_t tid = uint16_t(m.rd16(ds, off) & 0x3ff);
                if (r == sel_row && c == sel_col) tid = 0x3b;  // Utopia's blinking selection tile (bx = 0x1d0)
                if (!tid) continue;
                const CachedTile& ct = resolve_tid(tid);
                if (!ct.valid) continue;
                nax::WorldTile wt;
                wt.row = float(r);
                wt.col = float(c);
                wt.width = 32;
                wt.rows = ct.rows;
                wt.has_hd = ct.info.placed;
                wt.dimmed = ct.info.dimmed;
                wt.atlas_x = ct.info.atlas_x;
                wt.atlas_y = ct.info.atlas_y;
                wt.fallback_id = ct.fallback_id;
                out->tiles.push_back(wt);
            }
        }

        // Append dynamic map sprites (beacons 0..8, rovers/tanks, aliens, explosions, construction previews) projected
        // from Utopia's 8x8 pixel coordinates into exact planet map coordinates.
        for (const auto& d : dynamics) {
            const float bottom_y = float(d.bottom_up ? d.dos_y : (d.dos_y + d.rows - 1)) + 0.5f;
            const float cx = float(d.dos_x) + float(d.width) * 0.5f;
            const float cy = bottom_y - 7.5f;
            const float u = (cx - 128.0f) / 16.0f;
            const float v = (cy - 31.0f) / 8.0f;
            const float dc = (v + u) * 0.5f;
            const float dr = (v - u) * 0.5f;
            nax::WorldTile wt;
            wt.row = float(d.cam_row) + dr;
            wt.col = float(d.cam_col) + dc;
            wt.width = d.width;
            wt.rows = d.rows;
            const auto info = hd.resolve_world_sprite(m, d.data, d.width, d.rows, d.base, d.bottom_up);
            if (info.placed) {
                wt.has_hd = true;
                wt.dimmed = info.dimmed;
                wt.atlas_x = info.atlas_x;
                wt.atlas_y = info.atlas_y;
            } else {
                decode_4bit_rgba(m, d.data, d.width, d.rows, d.base, d.bottom_up, &rgba_tmp);
                wt.fallback_id = fallback.get_or_add(info.key, d.width, d.rows, rgba_tmp.data());
            }
            out->tiles.push_back(wt);
        }

        // Decode Utopia's live 10x10 Mode 13h software cursor bitmap from ds:0x5e68 into WorldFallbackAtlas so
        // the expanded view always shows the authentic in-game cursor instead of the Windows arrow.
        uint32_t cursor_rgba[100];
        uint64_t cursor_key = 0xC055070114650FBULL;
        auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
        bool any_cursor_px = false;
        for (int i = 0; i < 100; i++) {
            const uint8_t idx = m.rd8(ds, uint16_t(0x5e68 + i));
            cursor_key = (cursor_key ^ idx) * 1099511628211ULL;
            if (!idx) {
                cursor_rgba[i] = 0;
            } else {
                const uint8_t* d = m.dac[idx];
                cursor_key = (cursor_key ^ d[0]) * 1099511628211ULL;
                cursor_key = (cursor_key ^ d[1]) * 1099511628211ULL;
                cursor_key = (cursor_key ^ d[2]) * 1099511628211ULL;
                cursor_rgba[i] = expand(d[0]) | (expand(d[1]) << 8) | (expand(d[2]) << 16) | 0xFF000000u;
                any_cursor_px = true;
            }
        }
        out->cursor_fallback_id = any_cursor_px ? fallback.get_or_add(cursor_key, 10, 10, cursor_rgba) : 0;

        if (fallback.clear_count() == clears_before) break;
    }

    return true;
}

}  // namespace game
