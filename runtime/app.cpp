// The Windows program: window, input, presentation and the thread the translated game runs on.
//
// Options: --data <folder> (the game files), --sound adlib|beeper|roland, --language en|fr|de|it, --skip-intro, --mute,
// --filter sharp|smooth|crt|ai,
// and for automated checks --script <file>, --shot-every <ms> and --quit-after <ms>.
#include <windows.h>
#include <shellapi.h>
#include <windowsx.h>

#include <chrono>
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
bool g_shot_requested = false;
std::filesystem::path g_settings;
std::filesystem::path g_error_file;
std::string g_ai_problem;  // why the AI filter is not available, written to the log once the machine exists

// Shows an error to the player. During a scripted check nobody watches the screen, so the message goes to
// last-error.txt in the save folder instead of a dialog that would wait for a click.
void report_error(const std::wstring& text, const wchar_t* caption) {
    if (g_scripted && !g_error_file.empty()) {
        if (FILE* f = _wfopen(g_error_file.c_str(), L"w")) {
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
    refs.assign(nax::Presenter::kWidth * nax::Presenter::kHeight, nax::HdSprites::kNone);
    out.assign(nax::Presenter::kWidth * nax::Presenter::kHeight, 0xFF000000u);
    if (m.video_mode != 0x13) return;
    uint32_t palette[256];
    for (int i = 0; i < 256; i++) {
        auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
        palette[i] = expand(m.dac[i][0]) | (expand(m.dac[i][1]) << 8) | (expand(m.dac[i][2]) << 16) | 0xFF000000u;
    }
    const uint8_t* vram = m.vram();
    for (size_t i = 0; i < out.size(); i++) out[i] = palette[vram[i]];
    hd.frame(m, refs.data());
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

// The next frame is saved as it appears on screen, with the current filter.
void save_shot() { g_shot_requested = true; }

void show_filter(HWND hwnd) {
    std::wstring title = game::title();
    title += L"  [";
    title += kFilterTitles[g_filter];
    title += L"]";
    SetWindowTextW(hwnd, title.c_str());
}

// Switches to the next filter and remembers the choice for the next start.
void next_filter(HWND hwnd) {
    g_filter = (g_filter + 1) % nax::Presenter::kFilterCount;
    // The AI filter is offered only when its model has loaded.
    if (g_filter == nax::Presenter::kAiFilter && !(g_presenter && g_presenter->ai_ready()))
        g_filter = (g_filter + 1) % nax::Presenter::kFilterCount;
    show_filter(hwnd);
    if (FILE* f = _wfopen(g_settings.c_str(), L"w")) {
        std::fwprintf(f, L"filter=%ls\n", kFilterNames[g_filter]);
        std::fclose(f);
    }
}

int filter_from_name(const std::wstring& name) {
    for (int i = 0; i < nax::Presenter::kFilterCount; i++)
        if (name == kFilterNames[i]) return i;
    return -1;
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
}

// While the window is active the pointer stays inside it, so that relative movement from raw input always
// reaches the game. Switching to another window releases it.
void capture_pointer(HWND hwnd, bool on) {
    if (g_scripted) on = false;
    const bool starting = on && !g_captured;
    g_captured = on;
    g_have_last_client = false;
    g_abs_baseline = false;
    g_pending_x = g_pending_y = 0;
    if (!on) { ClipCursor(nullptr); return; }
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
    case WM_ACTIVATE:
        capture_pointer(hwnd, LOWORD(wp) != WA_INACTIVE && !HIWORD(wp));
        break;  // the default handling also gives the window keyboard focus
    case WM_INPUT:
        raw_mouse(lp);
        break;
    case WM_SYSKEYDOWN:
        if (wp == VK_RETURN && (lp & (1 << 29))) {
            if (!(lp & (1 << 30))) toggle_fullscreen(hwnd);
            return 0;
        }
        [[fallthrough]];
    case WM_KEYDOWN: case WM_KEYUP: case WM_SYSKEYUP: {
        const bool down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
        if (wp == VK_F12) { if (down) save_shot(); return 0; }
        if (wp == VK_F11) { if (down && !(lp & (1 << 30))) next_filter(hwnd); return 0; }
        if (g_machine) g_machine->key_event(uint8_t((lp >> 16) & 0x7F), (lp >> 24) & 1, down);
        return 0;
    }
    case WM_SYSCHAR: case WM_CHAR:
        return 0;
    case WM_MOUSEMOVE:
        mouse_to_game(lp);
        message_mouse(hwnd, lp);
        return 0;
    case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_RBUTTONDOWN: case WM_RBUTTONUP: {
        // A click into the window captures the mouse even if the window was not activated before.
        if (!g_captured && GetForegroundWindow() == hwnd) capture_pointer(hwnd, true);
        mouse_to_game(lp);
        if (g_machine && !g_scripted) g_machine->mouse.buttons = ((wp & MK_LBUTTON) ? 1 : 0) | ((wp & MK_RBUTTON) ? 2 : 0);
        if (msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN) SetCapture(hwnd);
        else if (!(wp & (MK_LBUTTON | MK_RBUTTON))) ReleaseCapture();
        return 0;
    }
    case WM_SETCURSOR:
        if (LOWORD(lp) == HTCLIENT) { SetCursor(nullptr); return TRUE; }
        break;
    case WM_CLOSE:
        if (g_machine) g_machine->request_stop();
        return 0;
    case kGameFinished:
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
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
        MessageBoxW(nullptr, L"The game is already running.", game::title(), MB_ICONINFORMATION);
        return 1;
    }
    wchar_t local[MAX_PATH] = L".";
    GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
    const std::filesystem::path saves = std::filesystem::path(local) / L"Nax Game Restoration" / game::data_folder_name();
    g_shots = saves / L"shots";
    g_settings = saves / L"settings.ini";
    g_error_file = saves / L"last-error.txt";
    {
        // The last filter chosen with F11 is kept; --filter overrides it for one start.
        std::error_code ec;
        std::filesystem::create_directories(saves, ec);
        wchar_t line[64] = L"";
        if (FILE* f = _wfopen(g_settings.c_str(), L"r")) {
            if (std::fgetws(line, 64, f)) {
                std::wstring text = line;
                while (!text.empty() && (text.back() == L'\n' || text.back() == L'\r')) text.pop_back();
                if (text.rfind(L"filter=", 0) == 0 && filter_from_name(text.substr(7)) >= 0) g_filter = filter_from_name(text.substr(7));
            }
            std::fclose(f);
        }
        if (options.count(L"filter") && filter_from_name(options.at(L"filter")) >= 0) g_filter = filter_from_name(options.at(L"filter"));
    }

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.lpszClassName = L"NaxGameRestoration";
    RegisterClassExW(&wc);
    RECT r{0, 0, 1280, 960};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, game::title(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left,
                             r.bottom - r.top, nullptr, nullptr, instance, nullptr);
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
    std::vector<uint32_t> frame, refs;
    MSG msg{};
    bool quit = false;
    while (!quit) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) quit = true;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (quit) break;
        const auto now = std::chrono::steady_clock::now();
        if (shot_every > 0 && now >= next_shot) {
            save_shot();
            next_shot += std::chrono::milliseconds(shot_every);
        }
        if (quit_after > 0 && now - start > std::chrono::milliseconds(quit_after)) machine.request_stop();
        flush_pending_motion();
        build_frame(machine, hd, frame, refs);
        std::filesystem::path shot;
        if (g_shot_requested) {
            g_shot_requested = false;
            std::error_code ec;
            std::filesystem::create_directories(g_shots, ec);
            wchar_t name[32];
            swprintf(name, 32, L"shot-%04d.png", g_shot_number++);
            shot = g_shots / name;
        }
        presenter.draw(frame.data(), refs.data(), g_filter, shot.empty() ? nullptr : &shot);
        if (!shot.empty() && g_filter == nax::Presenter::kAiFilter) machine.log("AI filter: %.1f ms of GPU time per picture", presenter.ai_gpu_ms());
    }
    machine.request_stop();
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
