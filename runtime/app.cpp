// The Windows program: window, input, presentation and the thread the translated game runs on.
//
// Options: --data <folder> (the game files), --sound adlib|beeper|roland, --language en|fr|de|it, --skip-intro, --mute,
// --filter sharp|smooth|crt|ai, --view expanded|classic, --window WxH,
// and for automated checks --script <file>, --shot-every <ms> and --quit-after <ms>.
#include <windows.h>
#include <shellapi.h>
#include <windowsx.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "audio.h"
#include "game.h"
#include "hd_sprites.h"
#include "machine.h"
#include "png.h"
#include "presenter.h"
#include "world_view.h"

namespace {

constexpr UINT kGameFinished = WM_APP + 1;

nax::Machine* g_machine = nullptr;
nax::Presenter* g_presenter = nullptr;
HWND g_hwnd = nullptr;
bool g_fullscreen = false;
bool g_captured = false;
bool g_scripted = false;  // a test script drives the input, so the real mouse is ignored and never captured
double g_rest_x = 0, g_rest_y = 0;
// Mouse movement reaches the game in one of three ways, depending on what the system delivers: relative raw
// input (a normal mouse), absolute raw input (Remote Desktop, virtual machines, some pens and touchpads), or
// plain window messages when raw input never arrives.
std::chrono::steady_clock::time_point g_last_raw{};
double g_pending_x = 0, g_pending_y = 0;  // movement from window messages, held back until raw input stays away
std::chrono::steady_clock::time_point g_pending_since{};
int g_filter = 0;
bool g_expanded_view = true;
bool g_world_active = false;
nax::WorldView g_world_view;
nax::WorldFallbackAtlas g_world_fallback;
nax::WorldRenderData g_world_data;
bool g_hover_valid = false;
int g_hover_row = 0, g_hover_col = 0;
bool g_pan_dragging = false;
UINT g_pan_button = 0;
POINT g_pan_start{}, g_pan_last{};
bool g_pan_moved = false;
bool g_rclick_pulse = false;
std::chrono::steady_clock::time_point g_rclick_release_at{};
bool g_lclick_pulse = false;
std::chrono::steady_clock::time_point g_lclick_release_at{};
POINT g_client_mouse{};
bool g_mouse_in_client = false;
bool g_tracking_leave = false;
bool g_shot_requested = false;
std::filesystem::path g_settings;
std::filesystem::path g_error_file;
std::string g_ai_problem;  // why the AI filter is not available, written to the log once the machine exists

// Shows an error to the player. During a scripted check nobody watches the screen, so the message goes to
// last-error.txt in the save folder instead of a dialog that would wait for a click.
void report_error(const std::wstring& text, const wchar_t* caption) {
    if (g_scripted && !g_error_file.empty()) {
        if (FILE* f = _wfopen(g_error_file.c_str(), L"wN")) {
            std::fwprintf(f, L"%ls\n", text.c_str());
            std::fclose(f);
        }
        return;
    }
    MessageBoxW(nullptr, text.c_str(), caption, MB_ICONERROR);
}

std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }
const wchar_t* const kFilterNames[nax::Presenter::kFilterCount] = {L"sharp", L"smooth", L"crt", L"ai"};
const wchar_t* const kFilterTitles[nax::Presenter::kFilterCount] = {L"sharp pixels", L"smooth edges", L"CRT", L"AI upscaling"};
bool g_abs_baseline = false;
bool g_raw_absolute = false;
double g_last_abs_x = 0, g_last_abs_y = 0;
POINT g_last_client{};
bool g_have_last_client = false;
const char* g_mouse_source = nullptr;
WINDOWPLACEMENT g_placement{sizeof(WINDOWPLACEMENT)};
std::filesystem::path g_shots;
int g_shot_number = 0;

const uint16_t kArrow[32] = {
    0x3FFF, 0x1FFF, 0x0FFF, 0x07FF, 0x03FF, 0x01FF, 0x00FF, 0x007F, 0x003F, 0x001F, 0x01FF, 0x10FF, 0x30FF, 0xF87F, 0xF87F, 0xFC7F,
    0x0000, 0x4000, 0x6000, 0x7000, 0x7800, 0x7C00, 0x7E00, 0x7F00, 0x7F80, 0x7C00, 0x6C00, 0x4600, 0x0600, 0x0300, 0x0300, 0x0000,
};

// Converts video memory to RGBA with the mouse pointer drawn in, and fills refs with the upscaled sprite
// picture for each pixel (none where the pointer is, so it stays on top).
void build_frame(nax::Machine& m, nax::HdSprites& hd, std::vector<uint32_t>& out, std::vector<uint32_t>& refs) {
    std::fill(refs.begin(), refs.end(), nax::HdSprites::kNone);
    std::fill(out.begin(), out.end(), 0xFF000000u);
    if (m.video_mode != 0x13) return;
    uint32_t palette[256];
    for (int i = 0; i < 256; i++) {
        auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
        palette[i] = expand(m.dac[i][0]) | (expand(m.dac[i][1]) << 8) | (expand(m.dac[i][2]) << 16) | 0xFF000000u;
    }
    // Atomically snapshot coherent VRAM and build HD sprite references in a single pass. In Expanded View,
    // erase Utopia's lagging VRAM cursor from our temporary copy before building refs so docked HUD strips
    // and floating popups stay clean beneath our native game cursor without double-counting frames.
    uint8_t vram_copy[nax::Presenter::kWidth * nax::Presenter::kHeight];
    if (g_expanded_view && g_world_active && !g_scripted) {
        hd.snapshot_and_frame(m, vram_copy, refs.data(), [&](uint8_t* v) { game::erase_vram_cursor(m, v); });
    } else {
        hd.snapshot_and_frame(m, vram_copy, refs.data());
    }
    if (g_expanded_view && g_world_active && g_world_data.has_popup && g_world_data.popup_has_holes) {
        hd.mark_popup_holes(refs.data(), g_world_data.popup_src_x, g_world_data.popup_src_y,
                            g_world_data.popup_src_x + g_world_data.popup_src_w,
                            g_world_data.popup_src_y + g_world_data.popup_src_h);
    }
    for (size_t i = 0; i < out.size(); i++) out[i] = palette[vram_copy[i]];
    if (m.cursor_visible != 0) return;
    const uint16_t* shape = m.custom_cursor ? m.cursor_shape : kArrow;
    const int hx = m.custom_cursor ? m.cursor_hot_x / 2 : 0, hy = m.custom_cursor ? m.cursor_hot_y : 0;
    const int x0 = m.mouse.x - hx, y0 = m.mouse.y - hy;
    for (int row = 0; row < 16; row++) {
        const int y = y0 + row;
        if (y < 0 || y >= 200) continue;
        for (int col = 0; col < 16; col++) {
            const int x = x0 + col;
            if (x < 0 || x >= 320) continue;
            const bool screen = (shape[row] >> (15 - col)) & 1, cursor = (shape[16 + row] >> (15 - col)) & 1;
            uint32_t& p = out[size_t(y) * 320 + size_t(x)];
            if (!screen || cursor) refs[size_t(y) * 320 + size_t(x)] = nax::HdSprites::kNone;
            if (!screen) p = cursor ? 0xFFFFFFFFu : 0xFF000000u;
            else if (cursor) p ^= 0x00FFFFFFu;
        }
    }
}

// Runs tools/ai/remaster.py in the background whenever the game has saved sprites that have no redrawn
// picture yet, provided the PyTorch environment described in the README exists. It only uses models that are
// already on the machine. The tool writes each
// picture under a temporary name and moves it into place, and the game picks the pictures up by itself.
class SpriteRedrawer {
public:
    void init(const std::filesystem::path& exe_dir, const std::filesystem::path& hd_folder, nax::Machine& m) {
        wchar_t local[MAX_PATH];
        if (!GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH)) return;
        python_ = std::filesystem::path(local) / L"Nax Game Restoration" / L"ai-env" / L"Scripts" / L"python.exe";
        std::error_code ec;
        for (const auto& root : {exe_dir, exe_dir.parent_path().parent_path()})
            if (std::filesystem::exists(root / L"tools" / L"ai" / L"remaster.py", ec)) root_ = root;
        enabled_ = !root_.empty() && std::filesystem::exists(python_, ec);
        log_ = hd_folder / L"redraw.log";
        std::filesystem::remove(log_, ec);  // the log covers this session only
        machine_ = &m;
        if (enabled_) m.log("New sprites are redrawn in the background with %ls", python_.c_str());
    }
    void tick(nax::HdSprites& hd) {
        if (!enabled_) return;
        const auto now = std::chrono::steady_clock::now();
        if (now < next_check_) return;
        next_check_ = now + std::chrono::seconds(3);
        if (process_) {
            if (WaitForSingleObject(process_, 0) != WAIT_OBJECT_0) return;
            CloseHandle(process_);
            process_ = nullptr;
        }
        // A new run starts when sprites without a picture are waiting and the game saved new ones since the
        // last run, so a sprite the tool cannot redraw does not start it again and again.
        const size_t count = hd.sprite_count();
        if (hd.missing_count() == 0 || count == count_at_last_run_) return;
        count_at_last_run_ = count;
        std::wstring command = L"\"" + python_.wstring() + L"\" -I \"" + (root_ / L"tools" / L"ai" / L"remaster.py").wstring() +
                               L"\" " + game::data_folder_name() + L" --tries 2 --offline";
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        HANDLE out = CreateFileW(log_.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof(si);
        std::vector<uint8_t> attr_buf;
        BOOL inherit = FALSE;
        DWORD flags = CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS;
        bool attr_ready = false;
        if (out != INVALID_HANDLE_VALUE) {
            SIZE_T attr_size = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
            if (attr_size > 0) {
                attr_buf.resize(attr_size);
                si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
                if (InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &attr_size) &&
                    UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &out, sizeof(out), nullptr, nullptr)) {
                    attr_ready = true;
                    inherit = TRUE;
                    flags |= EXTENDED_STARTUPINFO_PRESENT;
                    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
                    si.StartupInfo.hStdOutput = si.StartupInfo.hStdError = out;
                }
            }
        }
        PROCESS_INFORMATION pi{};
        if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, inherit, flags, nullptr, root_.c_str(), &si.StartupInfo, &pi)) {
            CloseHandle(pi.hThread);
            process_ = pi.hProcess;
            machine_->log("Redrawing new sprites in the background");
        } else {
            machine_->log("Could not start the sprite redrawer (error %lu)", GetLastError());
            enabled_ = false;
        }
        if (attr_ready) DeleteProcThreadAttributeList(si.lpAttributeList);
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
    }
    // Ends a run that is still going when the game closes. Finished pictures are already in place, and the
    // tool only ever replaces a picture with a complete one.
    void stop() {
        if (!process_) return;
        TerminateProcess(process_, 1);
        WaitForSingleObject(process_, 5000);
        CloseHandle(process_);
        process_ = nullptr;
    }

private:
    bool enabled_ = false;
    std::filesystem::path python_, root_, log_;
    nax::Machine* machine_ = nullptr;
    HANDLE process_ = nullptr;
    size_t count_at_last_run_ = 0;
    std::chrono::steady_clock::time_point next_check_{};
};

// The next frame is saved as it appears on screen, with the current filter.
void save_shot() { g_shot_requested = true; }

// Writes settings.ini atomically via a temporary sibling file so an interrupted write never corrupts it.
void save_settings() {
    if (g_settings.empty()) return;
    const std::filesystem::path temporary = g_settings.wstring() + L".part";
    FILE* f = _wfopen(temporary.c_str(), L"wN");
    if (!f) return;
    const bool wrote = std::fwprintf(f, L"filter=%ls\nview=%ls\n", kFilterNames[g_filter],
                                     g_expanded_view ? L"expanded" : L"classic") > 0;
    const bool closed = std::fclose(f) == 0;
    if (!wrote || !closed || !MoveFileExW(temporary.c_str(), g_settings.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
    }
}

void show_filter(HWND hwnd) {
    std::wstring title = game::title();
    title += L"  [";
    title += kFilterTitles[g_filter];
    if (g_filter == nax::Presenter::kAiFilter) title += g_expanded_view ? L", Expanded view" : L", Classic 4:3";
    if (g_captured) title += L", Right Ctrl releases mouse";
    title += L"]";
    SetWindowTextW(hwnd, title.c_str());
}

void capture_pointer(HWND hwnd, bool on);

void end_pan_drag(HWND hwnd) {
    g_pan_dragging = false;
    g_pan_button = 0;
    g_pan_moved = false;
    if (hwnd && GetCapture() == hwnd) ReleaseCapture();
}

bool direct_pointer_mode();

void sync_client_cursor_pos(HWND hwnd) {
    if (!hwnd) return;
    if (GetForegroundWindow() != hwnd) {
        g_mouse_in_client = false;
        return;
    }
    POINT pt{};
    RECT client{};
    if (GetCursorPos(&pt) && ScreenToClient(hwnd, &pt) && GetClientRect(hwnd, &client) && PtInRect(&client, pt)) {
        g_client_mouse = pt;
        g_mouse_in_client = true;
    } else {
        g_mouse_in_client = false;
    }
}

bool should_hide_os_cursor(HWND hwnd) {
    if (g_scripted) return false;
    if (g_captured) return true;
    if (!direct_pointer_mode()) return false;
    sync_client_cursor_pos(hwnd);
    if (!g_mouse_in_client) return false;
    if (g_world_active) return true;
    if (g_presenter) {
        const RECT pr = g_presenter->picture_rect();
        return PtInRect(&pr, g_client_mouse) != FALSE;
    }
    return false;
}

// Switches to the next filter and remembers the choice for the next start.
void next_filter(HWND hwnd) {
    end_pan_drag(hwnd);
    g_hover_valid = false;
    g_filter = (g_filter + 1) % nax::Presenter::kFilterCount;
    // The AI filter is offered only when its model has loaded.
    if (g_filter == nax::Presenter::kAiFilter && !(g_presenter && g_presenter->ai_ready()))
        g_filter = (g_filter + 1) % nax::Presenter::kFilterCount;
    if (g_filter == nax::Presenter::kAiFilter && g_expanded_view && g_captured) capture_pointer(hwnd, false);
    if (should_hide_os_cursor(hwnd)) SetCursor(nullptr);
    else if (!direct_pointer_mode() && !g_captured) SetCursor(LoadCursorW(nullptr, IDC_ARROW));
    show_filter(hwnd);
    save_settings();
}

void toggle_expanded_view(HWND hwnd) {
    end_pan_drag(hwnd);
    g_expanded_view = !g_expanded_view;
    g_hover_valid = false;
    if (g_expanded_view && g_captured) capture_pointer(hwnd, false);
    if (should_hide_os_cursor(hwnd)) SetCursor(nullptr);
    else if (!direct_pointer_mode() && !g_captured) SetCursor(LoadCursorW(nullptr, IDC_ARROW));
    show_filter(hwnd);
    save_settings();
}

int filter_from_name(const std::wstring& name) {
    for (int i = 0; i < nax::Presenter::kFilterCount; i++)
        if (name == kFilterNames[i]) return i;
    return -1;
}

bool direct_pointer_mode() {
    return !g_scripted && g_expanded_view && g_filter == nax::Presenter::kAiFilter;
}

void sync_game_cam_to_view_center() {
    const int cr = std::clamp(int(std::round(g_world_view.center_row())) - 4, 0, nax::WorldView::kMaxGameCamRow);
    const int cc = std::clamp(int(std::round(g_world_view.center_col())) - 4, 0, nax::WorldView::kMaxGameCamCol);
    if (cr != g_world_view.game_cam_row() || cc != g_world_view.game_cam_col()) {
        g_world_view.note_click_shift(cr, cc);
        game::queue_world_input(false, 0, 0, true, cr, cc);
    }
}

void map_client_to_dos_direct(POINT p) {
    if (!g_presenter || !g_machine) return;
    const RECT r = g_presenter->picture_rect();
    const int w = std::max<LONG>(1, r.right - r.left), h = std::max<LONG>(1, r.bottom - r.top);
    const int x = std::clamp(int((p.x - r.left) * 320 / w), 0, 319);
    const int y = std::clamp(int((p.y - r.top) * 200 / h), 0, 199);
    g_machine->mouse.x = x;
    g_machine->mouse.y = y;
    game::queue_world_input(true, x, y, false, 0, 0);
}

void toggle_fullscreen(HWND hwnd) {
    const DWORD style = GetWindowLongW(hwnd, GWL_STYLE);
    if (!g_fullscreen) {
        MONITORINFO mi{sizeof(mi)};
        GetWindowPlacement(hwnd, &g_placement);
        GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi);
        SetWindowLongW(hwnd, GWL_STYLE, (style & ~WS_OVERLAPPEDWINDOW) | WS_POPUP);
        SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    } else {
        SetWindowLongW(hwnd, GWL_STYLE, (style & ~WS_POPUP) | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &g_placement);
        SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    g_fullscreen = !g_fullscreen;
    if (g_fullscreen && GetForegroundWindow() == hwnd && !direct_pointer_mode()) capture_pointer(hwnd, true);
    else if (!g_fullscreen || direct_pointer_mode()) capture_pointer(hwnd, false);
}

// Capturing keeps the pointer inside the client area while playing so relative raw input always reaches the
// game. Windowed mode captures only after a click inside the picture, and Right Ctrl, middle click, window
// deactivation, or leaving fullscreen releases the pointer immediately.
void capture_pointer(HWND hwnd, bool on) {
    if (g_scripted) on = false;
    const bool was_captured = g_captured;
    const bool starting = on && !g_captured;
    g_captured = on;
    g_have_last_client = false;
    g_abs_baseline = false;
    g_pending_x = g_pending_y = 0;
    if (!on) {
        ClipCursor(nullptr);
        if (GetCapture() == hwnd) ReleaseCapture();
        g_lclick_pulse = false;
        g_rclick_pulse = false;
        if (g_machine && !g_scripted) game::release_world_buttons(*g_machine, 3);
        if (was_captured) {
            SetCursor(LoadCursorW(nullptr, IDC_ARROW));
            show_filter(hwnd);
        }
        return;
    }
    // When capture begins, the position starts where the pointer is, provided it is inside the window. Renewing
    // the capture after a move, a resize or Alt+Enter keeps the position, because the pointer is unrelated to
    // it while captured.
    POINT p;
    RECT client;
    GetClientRect(hwnd, &client);
    if (starting && g_presenter && g_machine && GetCursorPos(&p) && ScreenToClient(hwnd, &p) && PtInRect(&client, p)) {
        const RECT pr = g_presenter->picture_rect();
        g_machine->mouse_place(int((p.x - pr.left) * 320 / std::max<LONG>(1, pr.right - pr.left)),
                               int((p.y - pr.top) * 200 / std::max<LONG>(1, pr.bottom - pr.top)));
    }
    RECT r;
    GetClientRect(hwnd, &r);
    MapWindowPoints(hwnd, nullptr, reinterpret_cast<POINT*>(&r), 2);
    ClipCursor(&r);
    SetCursor(nullptr);
    if (starting) show_filter(hwnd);
}

void note_source(const char* source) {
    if (g_mouse_source == source || !g_machine) return;
    g_mouse_source = source;
    g_machine->log("Mouse input: %s", source);
}

// Adds a movement in screen pixels, scaled so that crossing the picture moves the game cursor across it.
void add_motion(double px, double py) {
    if (!g_presenter || !g_machine) return;
    const RECT r = g_presenter->picture_rect();
    g_rest_x += px * 320.0 / std::max<LONG>(1, r.right - r.left);
    g_rest_y += py * 200.0 / std::max<LONG>(1, r.bottom - r.top);
    const int mx = int(g_rest_x), my = int(g_rest_y);
    g_rest_x -= mx;
    g_rest_y -= my;
    // While the mouse is captured the absolute position is built from the same movement, so moving the
    // pointer back to the centre of the window never moves the cursor of a game that reads positions.
    g_machine->mouse_motion(mx, my);
}

// Applies movement from window messages once no raw input has followed it for a moment; raw input for the
// same movement replaces it instead, so nothing is counted twice.
void flush_pending_motion() {
    if (g_pending_x == 0 && g_pending_y == 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - g_pending_since < std::chrono::milliseconds(40)) return;
    if (g_last_raw >= g_pending_since) {
        g_pending_x = g_pending_y = 0;
        return;
    }
    note_source("window messages");
    add_motion(g_pending_x, g_pending_y);
    g_pending_x = g_pending_y = 0;
}

void raw_mouse(LPARAM lp) {
    if (!g_captured) return;
    RAWINPUT input;
    UINT size = sizeof(input);
    if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lp), RID_INPUT, &input, &size, sizeof(RAWINPUTHEADER)) == UINT(-1)) return;
    if (input.header.dwType != RIM_TYPEMOUSE) return;
    const RAWMOUSE& rm = input.data.mouse;
    if (!(rm.usFlags & MOUSE_MOVE_ABSOLUTE)) {
        if (rm.lLastX == 0 && rm.lLastY == 0) return;
        g_last_raw = std::chrono::steady_clock::now();
        g_raw_absolute = false;
        note_source("relative raw input");
        add_motion(rm.lLastX, rm.lLastY);
        return;
    }
    // Absolute positions run from 0 to 65535 across the primary screen or the whole virtual desktop.
    const bool desktop = rm.usFlags & MOUSE_VIRTUAL_DESKTOP;
    const double w = GetSystemMetrics(desktop ? SM_CXVIRTUALSCREEN : SM_CXSCREEN);
    const double h = GetSystemMetrics(desktop ? SM_CYVIRTUALSCREEN : SM_CYSCREEN);
    const double x = rm.lLastX * w / 65535.0, y = rm.lLastY * h / 65535.0;
    g_last_raw = std::chrono::steady_clock::now();
    g_raw_absolute = true;
    note_source("absolute raw input");
    if (g_abs_baseline) add_motion(x - g_last_abs_x, y - g_last_abs_y);
    g_last_abs_x = x;
    g_last_abs_y = y;
    g_abs_baseline = true;
}

// Uses window messages when no raw input has arrived recently, and keeps the pointer away from the window
// edges, so that movement never stops at an edge while the game cursor could still move.
void message_mouse(HWND hwnd, LPARAM lp) {
    if (!g_captured || g_scripted) return;
    const POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
    const bool raw_active = std::chrono::steady_clock::now() - g_last_raw < std::chrono::milliseconds(300);
    // While raw input arrives it carries every movement. Otherwise the movement is held back briefly, so that
    // raw input for the same movement can still replace it.
    if (!raw_active && g_have_last_client && (p.x != g_last_client.x || p.y != g_last_client.y)) {
        if (g_pending_x == 0 && g_pending_y == 0) g_pending_since = std::chrono::steady_clock::now();
        g_pending_x += p.x - g_last_client.x;
        g_pending_y += p.y - g_last_client.y;
    }
    g_last_client = p;
    g_have_last_client = true;
    // An absolute device reports where the hand is, so moving the pointer back would not help there.
    if (raw_active && g_raw_absolute) return;
    RECT r;
    GetClientRect(hwnd, &r);
    const LONG mx = (r.right - r.left) / 8, my = (r.bottom - r.top) / 8;
    if (p.x < r.left + mx || p.x > r.right - mx || p.y < r.top + my || p.y > r.bottom - my) {
        POINT centre{(r.left + r.right) / 2, (r.top + r.bottom) / 2};
        g_last_client = centre;
        ClientToScreen(hwnd, &centre);
        SetCursorPos(centre.x, centre.y);
        g_abs_baseline = false;  // the jump back to the centre is not a movement
    }
}

void mouse_to_game(LPARAM lp) {
    if (!g_presenter || !g_machine || g_scripted || g_captured) return;  // while captured, movement sets the position
    const RECT r = g_presenter->picture_rect();
    const int w = std::max<LONG>(1, r.right - r.left), h = std::max<LONG>(1, r.bottom - r.top);
    const int x = (GET_X_LPARAM(lp) - r.left) * 320 / w, y = (GET_Y_LPARAM(lp) - r.top) * 200 / h;
    g_machine->mouse.x = std::clamp(x, 0, 319);
    g_machine->mouse.y = std::clamp(y, 0, 199);
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (g_presenter) g_presenter->resize();
        if (g_captured) capture_pointer(hwnd, true);
        return 0;
    case WM_MOVE:
        if (g_captured) capture_pointer(hwnd, true);
        return 0;
    case WM_ACTIVATE: {
        const bool active = LOWORD(wp) != WA_INACTIVE && !HIWORD(wp);
        if (!active) {
            capture_pointer(hwnd, false);
            end_pan_drag(hwnd);
            g_hover_valid = false;
            g_mouse_in_client = false;
        } else {
            sync_client_cursor_pos(hwnd);
            if (should_hide_os_cursor(hwnd)) SetCursor(nullptr);
            if (g_fullscreen && !direct_pointer_mode()) capture_pointer(hwnd, true);
        }
        break;  // the default handling also gives the window keyboard focus
    }
    case WM_KILLFOCUS:
    case WM_ENTERSIZEMOVE:
        capture_pointer(hwnd, false);
        end_pan_drag(hwnd);
        g_hover_valid = false;
        g_mouse_in_client = false;
        break;
    case WM_MOUSELEAVE:
        g_mouse_in_client = false;
        g_tracking_leave = false;
        g_hover_valid = false;
        return 0;
    case WM_CAPTURECHANGED:
        g_pan_dragging = false;
        g_pan_button = 0;
        g_pan_moved = false;
        break;
    case WM_INPUT:
        raw_mouse(lp);
        break;
    case WM_MOUSEWHEEL: {
        if (g_world_active && !g_world_data.has_popup && !g_scripted) {
            const int delta = GET_WHEEL_DELTA_WPARAM(wp);
            if (delta != 0) g_world_view.adjust_zoom(delta > 0 ? 1.12f : (1.0f / 1.12f));
            return 0;
        }
        break;
    }
    case WM_SYSKEYDOWN:
        if (wp == VK_RETURN && (lp & (1 << 29))) {
            if (!(lp & (1 << 30))) toggle_fullscreen(hwnd);
            return 0;
        }
        if (wp == VK_F4 && (lp & (1 << 29))) {
            capture_pointer(hwnd, false);
            PostMessageW(hwnd, WM_CLOSE, 0, 0);
            return 0;
        }
        if (wp == VK_SPACE && (lp & (1 << 29))) {
            capture_pointer(hwnd, false);
            return DefWindowProcW(hwnd, msg, wp, lp);
        }
        [[fallthrough]];
    case WM_KEYDOWN: case WM_KEYUP: case WM_SYSKEYUP: {
        if (wp == VK_LWIN || wp == VK_RWIN) {
            capture_pointer(hwnd, false);
            return DefWindowProcW(hwnd, msg, wp, lp);
        }
        const bool down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
        if (wp == VK_CONTROL && ((lp >> 24) & 1)) {
            if (down && g_captured) capture_pointer(hwnd, false);
            return 0;
        }
        if (wp == VK_F12) { if (down) save_shot(); return 0; }
        if (wp == VK_F11) { if (down && !(lp & (1 << 30))) next_filter(hwnd); return 0; }
        if (wp == VK_F9) { if (down && !(lp & (1 << 30))) toggle_expanded_view(hwnd); return 0; }
        // In Expanded View (when no modal popup is open), WASD and arrow keys pan the ultrawide map smoothly.
        if (g_world_active && !g_world_data.has_popup && !g_scripted &&
            (wp == 'W' || wp == 'A' || wp == 'S' || wp == 'D' || wp == VK_UP || wp == VK_DOWN || wp == VK_LEFT || wp == VK_RIGHT)) {
            if (!down) sync_game_cam_to_view_center();
            return 0;
        }
        if (g_machine) g_machine->key_event(uint8_t((lp >> 16) & 0x7F), (lp >> 24) & 1, down);
        return 0;
    }
    case WM_SYSCHAR:
        if (wp == VK_SPACE) return DefWindowProcW(hwnd, msg, wp, lp);
        return 0;
    case WM_CHAR:
        return 0;
    case WM_MOUSEMOVE: {
        const POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
        g_client_mouse = p;
        g_mouse_in_client = true;
        if (!g_tracking_leave) {
            TRACKMOUSEEVENT tme{sizeof(tme), TME_LEAVE, hwnd, 0};
            if (TrackMouseEvent(&tme)) g_tracking_leave = true;
        }
        if (direct_pointer_mode() && g_presenter && g_machine) {
            if (should_hide_os_cursor(hwnd)) SetCursor(nullptr);
            const int w = int(g_presenter->width()), h = int(g_presenter->height());
            if (g_pan_dragging) {
                if (!g_pan_moved && (std::abs(p.x - g_pan_start.x) > 5 || std::abs(p.y - g_pan_start.y) > 5)) {
                    g_pan_moved = true;
                }
                if (g_pan_moved && g_world_active && !g_world_data.has_popup) {
                    g_world_view.pan_screen_pixels(float(p.x - g_pan_last.x), float(p.y - g_pan_last.y), w, h);
                }
                g_pan_last = p;
                return 0;
            }
            if (g_world_active) {
                const bool click_active = (wp & (MK_LBUTTON | MK_RBUTTON)) != 0 || g_lclick_pulse || g_rclick_pulse;
                const auto hit = g_world_view.hit_test(float(p.x), float(p.y), w, h,
                                                       g_world_view.game_cam_row(), g_world_view.game_cam_col(), &g_world_data);
                if (hit.kind == nax::WorldHit::Kind::tile) {
                    g_hover_valid = true;
                    g_hover_row = hit.tile_row;
                    g_hover_col = hit.tile_col;
                    if (!click_active) {
                        if (!hit.need_cam_shift) {
                            g_machine->mouse.x = hit.dos_x;
                            g_machine->mouse.y = hit.dos_y;
                            game::queue_world_input(true, hit.dos_x, hit.dos_y, false, 0, 0);
                        } else {
                            // Park Utopia's software cursor at (4, 155), strictly outside both Utopia's 8x8 map
                            // diamond and all docked HUD strips, so Utopia never falsely blinks a central tile.
                            game::queue_world_input(true, 4, 155, false, 0, 0);
                        }
                    }
                } else if (hit.kind == nax::WorldHit::Kind::hud) {
                    g_hover_valid = false;
                    if (!click_active) {
                        g_machine->mouse.x = hit.dos_x;
                        g_machine->mouse.y = hit.dos_y;
                        game::queue_world_input(true, hit.dos_x, hit.dos_y, false, 0, 0);
                    }
                } else {
                    g_hover_valid = false;
                }
            } else {
                g_hover_valid = false;
                map_client_to_dos_direct(p);
            }
            return 0;
        }
        mouse_to_game(lp);
        message_mouse(hwnd, lp);
        return 0;
    }
    case WM_MBUTTONDOWN: {
        if (direct_pointer_mode() && g_world_active && !g_world_data.has_popup) {
            g_pan_dragging = true;
            g_pan_button = WM_MBUTTONDOWN;
            g_pan_start = g_pan_last = POINT{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
            g_pan_moved = true;
            SetCapture(hwnd);
            return 0;
        }
        if (g_captured) capture_pointer(hwnd, false);
        return 0;
    }
    case WM_MBUTTONUP: {
        if (g_pan_dragging && g_pan_button == WM_MBUTTONDOWN) {
            end_pan_drag(hwnd);
            if (g_world_active) sync_game_cam_to_view_center();
            return 0;
        }
        return 0;
    }
    case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_RBUTTONDOWN: case WM_RBUTTONUP: {
        const bool button_down = msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN;
        // Always terminate an active right-button drag on WM_RBUTTONUP even if a popup opened mid-drag.
        if (msg == WM_RBUTTONUP && g_pan_dragging && g_pan_button == WM_RBUTTONDOWN) {
            const bool moved = g_pan_moved;
            end_pan_drag(hwnd);
            if (direct_pointer_mode() && g_presenter && g_machine && g_world_active) {
                if (moved) {
                    sync_game_cam_to_view_center();
                } else {
                    const POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
                    const int w = int(g_presenter->width()), h = int(g_presenter->height());
                    const auto hit = g_world_view.hit_test(float(p.x), float(p.y), w, h,
                                                           g_world_view.game_cam_row(), g_world_view.game_cam_col(), &g_world_data);
                    const int next_btns = int(g_machine->mouse.buttons.load()) | 2;
                    if (hit.kind == nax::WorldHit::Kind::tile && hit.need_cam_shift) {
                        g_machine->mouse.x = hit.dos_x;
                        g_machine->mouse.y = hit.dos_y;
                        g_world_view.note_click_shift(hit.shift_cam_row, hit.shift_cam_col);
                        game::queue_world_input(true, hit.dos_x, hit.dos_y, true, hit.shift_cam_row, hit.shift_cam_col, next_btns);
                    } else if (hit.kind != nax::WorldHit::Kind::none) {
                        g_machine->mouse.x = hit.dos_x;
                        g_machine->mouse.y = hit.dos_y;
                        game::queue_world_input(true, hit.dos_x, hit.dos_y, false, 0, 0, next_btns);
                    } else {
                        game::queue_world_input(false, 0, 0, false, 0, 0, next_btns);
                    }
                    g_rclick_pulse = true;
                    g_rclick_release_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
                }
            } else if (g_machine && !g_scripted) {
                game::release_world_buttons(*g_machine, 2);
            }
            return 0;
        }
        if (direct_pointer_mode() && g_presenter && g_machine) {
            const POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
            const int w = int(g_presenter->width()), h = int(g_presenter->height());
            if (g_world_active) {
                if (msg == WM_RBUTTONDOWN) {
                    if (g_world_data.has_popup) {
                        // While a modal popup is open, forward right-click directly instead of starting a map drag.
                        g_rclick_pulse = false;
                        const auto hit = g_world_view.hit_test(float(p.x), float(p.y), w, h,
                                                               g_world_view.game_cam_row(), g_world_view.game_cam_col(), &g_world_data);
                        const int next_btns = int(g_machine->mouse.buttons.load()) | 2;
                        if (hit.kind != nax::WorldHit::Kind::none) {
                            g_machine->mouse.x = hit.dos_x;
                            g_machine->mouse.y = hit.dos_y;
                            game::queue_world_input(true, hit.dos_x, hit.dos_y, false, 0, 0, next_btns);
                        } else {
                            game::queue_world_input(false, 0, 0, false, 0, 0, next_btns);
                        }
                        SetCapture(hwnd);
                        return 0;
                    }
                    g_pan_dragging = true;
                    g_pan_button = WM_RBUTTONDOWN;
                    g_pan_start = g_pan_last = p;
                    g_pan_moved = false;
                    SetCapture(hwnd);
                    return 0;
                }
                if (msg == WM_RBUTTONUP) {
                    if ((g_machine->mouse.buttons.load() & 2) == 0) {
                        // A fast right-click over a popup arrived before 0ffe:1085 consumed the queued press:
                        // hold briefly so dialog dismissal right-clicks are never dropped.
                        g_rclick_pulse = true;
                        g_rclick_release_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(80);
                    } else {
                        g_rclick_pulse = false;
                        game::release_world_buttons(*g_machine, 2);
                    }
                    if (!g_pan_dragging && !(wp & MK_LBUTTON) && GetCapture() == hwnd) ReleaseCapture();
                    return 0;
                }
                if (msg == WM_LBUTTONDOWN) {
                    const auto hit = g_world_view.hit_test(float(p.x), float(p.y), w, h,
                                                           g_world_view.game_cam_row(), g_world_view.game_cam_col(), &g_world_data);
                    const int next_btns = int(g_machine->mouse.buttons.load()) | 1;
                    if (hit.kind == nax::WorldHit::Kind::tile) {
                        g_hover_valid = true;
                        g_hover_row = hit.tile_row;
                        g_hover_col = hit.tile_col;
                        g_machine->mouse.x = hit.dos_x;
                        g_machine->mouse.y = hit.dos_y;
                        g_lclick_pulse = false;
                        if (hit.need_cam_shift) {
                            g_world_view.note_click_shift(hit.shift_cam_row, hit.shift_cam_col);
                            game::queue_world_input(true, hit.dos_x, hit.dos_y, true, hit.shift_cam_row, hit.shift_cam_col, next_btns);
                        } else {
                            game::queue_world_input(true, hit.dos_x, hit.dos_y, false, 0, 0, next_btns);
                        }
                        SetCapture(hwnd);
                    } else if (hit.kind == nax::WorldHit::Kind::hud) {
                        g_machine->mouse.x = hit.dos_x;
                        g_machine->mouse.y = hit.dos_y;
                        g_lclick_pulse = false;
                        game::queue_world_input(true, hit.dos_x, hit.dos_y, false, 0, 0, next_btns);
                        SetCapture(hwnd);
                    }
                    return 0;
                }
                if (msg == WM_LBUTTONUP) {
                    if ((g_machine->mouse.buttons.load() & 1) == 0) {
                        // A fast click arrived before the game thread consumed the queued button press:
                        // let it remain pressed briefly so Utopia never misses a quick click on a distant tile.
                        g_lclick_pulse = true;
                        g_lclick_release_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(80);
                    } else {
                        g_lclick_pulse = false;
                        game::release_world_buttons(*g_machine, 1);
                    }
                    if (!g_pan_dragging && GetCapture() == hwnd) ReleaseCapture();
                    return 0;
                }
            }
            // Expanded view is enabled, but a 4:3 menu or popup (such as SELECT A BUILDING.. or Options) is open:
            // map clicks directly through picture_rect() without capturing the mouse.
            map_client_to_dos_direct(p);
            const int btns = ((wp & MK_LBUTTON) ? 1 : 0) | ((wp & MK_RBUTTON) ? 2 : 0);
            if (!button_down) {
                g_lclick_pulse = false;
                g_rclick_pulse = false;
                game::release_world_buttons(*g_machine, (~btns) & 3);
            } else {
                g_machine->mouse.buttons = btns;
            }
            if (button_down) SetCapture(hwnd);
            else if (!(wp & (MK_LBUTTON | MK_RBUTTON)) && GetCapture() == hwnd) ReleaseCapture();
            return 0;
        }
        // Classic 4:3 mode: clicking inside the game picture captures the mouse without passing a stray click.
        if (button_down && !g_captured && GetForegroundWindow() == hwnd) {
            if (g_presenter) {
                const RECT pr = g_presenter->picture_rect();
                const POINT click_pt{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
                if (PtInRect(&pr, click_pt)) capture_pointer(hwnd, true);
            }
            return 0;
        }
        mouse_to_game(lp);
        if (g_machine && !g_scripted) {
            const int btns = ((wp & MK_LBUTTON) ? 1 : 0) | ((wp & MK_RBUTTON) ? 2 : 0);
            game::release_world_buttons(*g_machine, (~btns) & 3);
            g_machine->mouse.buttons = btns;
        }
        if (button_down) SetCapture(hwnd);
        else if (!(wp & (MK_LBUTTON | MK_RBUTTON)) && GetCapture() == hwnd) ReleaseCapture();
        return 0;
    }
    case WM_SETCURSOR:
        if (LOWORD(lp) == HTCLIENT && should_hide_os_cursor(hwnd)) {
            SetCursor(nullptr);
            return TRUE;
        }
        break;
    case WM_CLOSE:
        capture_pointer(hwnd, false);
        if (g_machine) g_machine->request_stop();
        return 0;
    case kGameFinished:
        capture_pointer(hwnd, false);
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        capture_pointer(hwnd, false);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

std::map<std::wstring, std::wstring> parse_options() {
    std::map<std::wstring, std::wstring> options;
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; i < argc; i++) {
        std::wstring key = argv[i];
        if (key.rfind(L"--", 0) != 0) continue;
        key = key.substr(2);
        if (i + 1 < argc && std::wstring(argv[i + 1]).rfind(L"--", 0) != 0) options[key] = argv[++i];
        else options[key] = L"1";
    }
    LocalFree(argv);
    return options;
}

std::filesystem::path find_data(const std::map<std::wstring, std::wstring>& options) {
    if (options.count(L"data")) return options.at(L"data");
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::filesystem::path dir = std::filesystem::path(exe).parent_path();
    for (const auto& candidate : {dir, dir / L"game", dir.parent_path() / L"gamedata" / game::data_folder_name(),
                                  dir.parent_path().parent_path() / L"gamedata" / game::data_folder_name()}) {
        std::error_code ec;
        if (std::filesystem::exists(candidate / game::required_file(), ec)) return candidate;
    }
    return {};
}

void run_script(const std::filesystem::path& path, nax::Machine* m) {
    std::ifstream in(path);
    std::string line;
    const auto start = std::chrono::steady_clock::now();
    int script_x = 0, script_y = 0;  // the script's own pointer, independent of the real one
    while (std::getline(in, line) && !m->stopped()) {
        std::istringstream words(line);
        long ms = 0;
        std::string what;
        if (!(words >> ms >> what) || line[0] == '#') continue;
        // Wait in short steps, so that closing the window ends the script at once.
        while (!m->stopped() && std::chrono::steady_clock::now() < start + std::chrono::milliseconds(ms))
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (m->stopped()) break;
        if (what == "key" || what == "down" || what == "up") {
            unsigned code = 0;
            words >> std::hex >> code;
            if (what != "up") m->key_event(uint8_t(code & 0x7F), code > 0xFF, true);
            if (what == "key") std::this_thread::sleep_for(std::chrono::milliseconds(80));
            if (what != "down") m->key_event(uint8_t(code & 0x7F), code > 0xFF, false);
        } else if (what == "move" || what == "click" || what == "rclick") {
            int x = 0, y = 0;
            words >> x >> y;
            m->mouse.dx += x - script_x;
            m->mouse.dy += y - script_y;
            script_x = x;
            script_y = y;
            m->mouse.x = x;
            m->mouse.y = y;
            if (what != "move") {
                std::this_thread::sleep_for(std::chrono::milliseconds(60));
                m->mouse.buttons = what == "click" ? 1 : 2;
                std::this_thread::sleep_for(std::chrono::milliseconds(120));
                m->mouse.buttons = 0;
            }
        } else if (what == "shot") {
            PostMessageW(g_hwnd, WM_KEYDOWN, VK_F12, 0);
        } else if (what == "quit") {
            m->request_stop();
        }
    }
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    const auto options = parse_options();
    g_scripted = options.count(L"script") > 0;
    const std::filesystem::path data = find_data(options);
    if (data.empty()) {
        std::wstring text = L"The game files were not found. Start the program with --data <folder>, where the folder holds ";
        text += game::required_file();
        text += L" and the other files of your copy of the game.";
        report_error(text, game::title());
        return 1;
    }
    // One copy of a game at a time: a second copy sharing the save folder could remove the first copy's
    // unfinished files at start, or replace a save the first copy is writing.
    const std::wstring mutex_name = std::wstring(L"Local\\NaxGameRestoration-") + game::data_folder_name();
    // If the guard cannot be created at all, the game still starts; only a running copy stops it.
    HANDLE instance_mutex = CreateMutexW(nullptr, TRUE, mutex_name.c_str());
    if (instance_mutex && GetLastError() == ERROR_ALREADY_EXISTS) {
        report_error(L"The game is already running.", game::title());
        return 1;
    }
    wchar_t local[MAX_PATH] = L".";
    GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
    const std::filesystem::path saves = std::filesystem::path(local) / L"Nax Game Restoration" / game::data_folder_name();
    g_shots = saves / L"shots";
    g_settings = saves / L"settings.ini";
    g_error_file = saves / L"last-error.txt";
    // Seed any portable hd\original and hd\upscaled PNGs placed next to the executable (or inside the game folder)
    // into %LOCALAPPDATA%\...\hd atomically without ever overwriting existing local files.
    {
        wchar_t exe_buf[MAX_PATH];
        GetModuleFileNameW(nullptr, exe_buf, MAX_PATH);
        const std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
        for (const auto& hd_src : {exe_dir / L"hd", data / L"hd"}) {
            std::error_code ec;
            if (!std::filesystem::is_directory(hd_src / L"upscaled", ec)) continue;
            for (const wchar_t* sub : {L"original", L"upscaled"}) {
                const auto src_sub = hd_src / sub;
                if (!std::filesystem::is_directory(src_sub, ec)) continue;
                const auto dst_sub = saves / L"hd" / sub;
                std::filesystem::create_directories(dst_sub, ec);
                for (std::filesystem::directory_iterator it(src_sub, ec), end; !ec && it != end; it.increment(ec)) {
                    std::error_code file_ec;
                    if (!it->is_regular_file(file_ec) || it->path().extension() != L".png") continue;
                    const auto dst = dst_sub / it->path().filename();
                    if (std::filesystem::exists(dst, file_ec)) continue;
                    const auto tmp = dst.wstring() + L".part";
                    if (std::filesystem::copy_file(it->path(), tmp, std::filesystem::copy_options::overwrite_existing, file_ec)) {
                        const DWORD attrs = GetFileAttributesW(tmp.c_str());
                        if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_READONLY))
                            SetFileAttributesW(tmp.c_str(), attrs & ~FILE_ATTRIBUTE_READONLY);
                        if (!MoveFileExW(tmp.c_str(), dst.c_str(), MOVEFILE_WRITE_THROUGH))
                            std::filesystem::remove(tmp, file_ec);
                    } else {
                        std::filesystem::remove(tmp, file_ec);
                    }
                }
            }
            break;
        }
    }
    {
        // The last filter (F11) and view mode (F9) are kept; --filter and --view override them for one start.
        std::error_code ec;
        std::filesystem::create_directories(saves, ec);
        g_expanded_view = !g_scripted;
        g_filter = g_scripted ? 0 : nax::Presenter::kAiFilter;
        wchar_t line[128] = L"";
        if (FILE* f = _wfopen(g_settings.c_str(), L"rN")) {
            while (std::fgetws(line, 128, f)) {
                std::wstring text = line;
                while (!text.empty() && (text.back() == L'\n' || text.back() == L'\r')) text.pop_back();
                if (text.rfind(L"filter=", 0) == 0 && filter_from_name(text.substr(7)) >= 0) g_filter = filter_from_name(text.substr(7));
                else if (text == L"view=expanded" && !g_scripted) g_expanded_view = true;
                else if (text == L"view=classic" && !g_scripted) g_expanded_view = false;
            }
            std::fclose(f);
        }
        if (options.count(L"filter") && filter_from_name(options.at(L"filter")) >= 0) g_filter = filter_from_name(options.at(L"filter"));
        if (options.count(L"view")) {
            if (options.at(L"view") == L"expanded") g_expanded_view = true;
            else if (options.at(L"view") == L"classic") g_expanded_view = false;
        }
    }

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.lpszClassName = L"NaxGameRestoration";
    RegisterClassExW(&wc);
    int init_w = 1280, init_h = 960;
    bool explicit_window = false;
    if (options.count(L"window")) {
        int ww = 0, wh = 0;
        if (std::swscanf(options.at(L"window").c_str(), L"%dx%d", &ww, &wh) == 2 && ww >= 320 && wh >= 200) {
            init_w = ww;
            init_h = wh;
            explicit_window = true;
        }
    }
    RECT work{};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    const int work_w = std::max<int>(640, work.right - work.left);
    const int work_h = std::max<int>(480, work.bottom - work.top);
    if (!explicit_window && !g_scripted && g_expanded_view) {
        init_w = std::clamp(work_w * 85 / 100, 1280, 2400);
        init_h = std::clamp(work_h * 85 / 100, 720, 1200);
    }
    RECT r{0, 0, init_w, init_h};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    const int win_outer_w = r.right - r.left, win_outer_h = r.bottom - r.top;
    const int win_x = work.left + std::max(0, (work_w - win_outer_w) / 2);
    const int win_y = work.top + std::max(0, (work_h - win_outer_h) / 2);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, game::title(), WS_OVERLAPPEDWINDOW, win_x, win_y, win_outer_w,
                             win_outer_h, nullptr, nullptr, instance, nullptr);
    nax::Presenter presenter;
    std::string error;
    if (!presenter.init(g_hwnd, &error)) {
        report_error(widen("Direct3D 12 could not start: " + error), L"Nax Game Restoration");
        return 1;
    }
    g_presenter = &presenter;
    {
        // The AI model lies next to the program, or in the models folder of the source tree during development.
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        const std::filesystem::path dir = std::filesystem::path(exe).parent_path();
        const wchar_t* model = L"realesr-animevideov3.bin";
        std::string ai_error = "The AI model file was not found.";
        bool loaded = false;
        for (const auto& candidate : {dir / L"models" / model, dir.parent_path().parent_path() / L"models" / model}) {
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec)) {
                loaded = presenter.enable_ai(candidate, &ai_error);
                break;
            }
        }
        if (!loaded) {
            g_ai_problem = ai_error;
            if (g_filter == nax::Presenter::kAiFilter) g_filter = 0;
        }
    }
    show_filter(g_hwnd);
    RAWINPUTDEVICE mouse_device{0x01, 0x02, 0, g_hwnd};
    RegisterRawInputDevices(&mouse_device, 1, sizeof(mouse_device));
    ShowWindow(g_hwnd, show);

    nax::Machine machine;
    machine.set_folders(data, saves);
    if (!g_ai_problem.empty()) machine.log("AI filter not available: %s", g_ai_problem.c_str());
    nax::Audio audio;
    std::string audio_error;
    // The chip's timers answer even without a sound device, so the game's AdLib detection still succeeds.
    machine.audio = &audio;
    // Every sprite the game draws is saved once in the "original" folder next to the saves; upscaled pictures
    // in the "upscaled" folder replace them in the AI filter.
    nax::HdSprites hd;
    hd.set_folder(saves / L"hd");
    machine.hd = &hd;
    presenter.set_hd(&hd);
    SpriteRedrawer redrawer;
    if (!options.count(L"no-redraw")) {
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        redrawer.init(std::filesystem::path(exe).parent_path(), saves / L"hd", machine);
    }
    if (options.count(L"record-audio")) {
        if (options.count(L"mute")) machine.log("--record-audio is ignored because --mute turns sound output off");
        else audio.record(options.at(L"record-audio"));
    }
    if (!options.count(L"mute") && !audio.start(&audio_error)) machine.log("No sound: %s", audio_error.c_str());
    game::register_modules(machine);
    g_machine = &machine;
    std::thread game_thread([&] {
        game::run(machine, options);
        PostMessageW(g_hwnd, kGameFinished, 0, 0);
    });
    std::thread script_thread;
    if (options.count(L"script")) script_thread = std::thread(run_script, std::filesystem::path(options.at(L"script")), &machine);

    const auto start = std::chrono::steady_clock::now();
    const long shot_every = options.count(L"shot-every") ? std::wcstol(options.at(L"shot-every").c_str(), nullptr, 10) : 0;
    const long quit_after = options.count(L"quit-after") ? std::wcstol(options.at(L"quit-after").c_str(), nullptr, 10) : 0;
    auto next_shot = start + std::chrono::milliseconds(shot_every);
    std::vector<uint32_t> frame(nax::Presenter::kWidth * nax::Presenter::kHeight, 0xFF000000u);
    std::vector<uint32_t> refs(nax::Presenter::kWidth * nax::Presenter::kHeight, nax::HdSprites::kNone);
    auto last_frame = start;
    MSG msg{};
    bool quit = false;
    auto update_world = [&] {
        const bool was_active = g_world_active;
        if (g_expanded_view && g_filter == nax::Presenter::kAiFilter) {
            g_world_active = game::build_world_render(machine, hd, g_world_view, g_world_fallback,
                                                      int(presenter.width()), int(presenter.height()), &g_world_data);
            if (g_world_active) {
                g_world_data.has_hover = g_hover_valid && !g_world_data.has_popup && !g_pan_dragging && !g_scripted;
                g_world_data.hover_row = g_hover_row;
                g_world_data.hover_col = g_hover_col;
                g_world_data.has_game_cursor = g_mouse_in_client && !g_scripted && !g_captured;
                g_world_data.cursor_x = float(g_client_mouse.x);
                g_world_data.cursor_y = float(g_client_mouse.y);
                if (g_world_data.has_popup && g_pan_dragging) end_pan_drag(g_hwnd);
            }
        } else {
            g_world_active = false;
        }
        if (!g_world_active) {
            hd.begin_world_frame();
            g_world_data.tiles.clear();
            g_world_data.has_popup = false;
            if (was_active) {
                end_pan_drag(g_hwnd);
                g_hover_valid = false;
            }
        }
        if (g_world_active != was_active && !g_scripted) {
            if (should_hide_os_cursor(g_hwnd)) SetCursor(nullptr);
            else if (g_mouse_in_client) SetCursor(LoadCursorW(nullptr, IDC_ARROW));
        }
    };
    while (!quit) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) quit = true;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (quit) break;
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::clamp(std::chrono::duration<float>(now - last_frame).count(), 0.0f, 0.1f);
        last_frame = now;
        if (g_rclick_pulse && now >= g_rclick_release_at) {
            g_rclick_pulse = false;
            game::release_world_buttons(machine, 2);
        }
        if (g_lclick_pulse && now >= g_lclick_release_at) {
            g_lclick_pulse = false;
            game::release_world_buttons(machine, 1);
        }
        if (g_world_active && !g_world_data.has_popup && !g_scripted && GetForegroundWindow() == g_hwnd) {
            float pan_x = 0.0f, pan_y = 0.0f;
            const float speed = 780.0f * dt;
            if ((GetAsyncKeyState('A') & 0x8000) || (GetAsyncKeyState(VK_LEFT) & 0x8000)) pan_x += speed;
            if ((GetAsyncKeyState('D') & 0x8000) || (GetAsyncKeyState(VK_RIGHT) & 0x8000)) pan_x -= speed;
            if ((GetAsyncKeyState('W') & 0x8000) || (GetAsyncKeyState(VK_UP) & 0x8000)) pan_y += speed;
            if ((GetAsyncKeyState('S') & 0x8000) || (GetAsyncKeyState(VK_DOWN) & 0x8000)) pan_y -= speed;
            if (pan_x != 0.0f || pan_y != 0.0f) {
                g_world_view.pan_screen_pixels(pan_x, pan_y, int(presenter.width()), int(presenter.height()));
            }
        }
        if (shot_every > 0 && now >= next_shot) {
            save_shot();
            next_shot += std::chrono::milliseconds(shot_every);
        }
        if (quit_after > 0 && now - start > std::chrono::milliseconds(quit_after)) machine.request_stop();
        flush_pending_motion();
        update_world();
        build_frame(machine, hd, frame, refs);
        redrawer.tick(hd);
        std::filesystem::path shot;
        if (g_shot_requested) {
            g_shot_requested = false;
            std::error_code ec;
            std::filesystem::create_directories(g_shots, ec);
            wchar_t name[32];
            swprintf(name, 32, L"shot-%04d.png", g_shot_number++);
            shot = g_shots / name;
        }
        presenter.draw(frame.data(), refs.data(), g_filter, shot.empty() ? nullptr : &shot,
                       [&] {
                           update_world();
                           build_frame(machine, hd, frame, refs);
                       },
                       g_world_active ? &g_world_data : nullptr);
        if (!shot.empty() && g_filter == nax::Presenter::kAiFilter) machine.log("AI filter: %.1f ms of GPU time per picture", presenter.ai_gpu_ms());
    }
    machine.request_stop();
    redrawer.stop();
    game_thread.join();
    machine.audio = nullptr;
    machine.hd = nullptr;
    audio.stop();
    if (script_thread.joinable()) script_thread.join();
    g_machine = nullptr;
    g_presenter = nullptr;
    if (!machine.failure().empty()) {
        report_error(widen(machine.failure()), L"Nax Game Restoration");
        return 2;
    }
    return 0;
}
