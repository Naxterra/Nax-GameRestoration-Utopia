// The Windows program: window, input, presentation and the thread the translated game runs on.
//
// Options: --data <folder> (the game files), --sound adlib|beeper|roland, --language en|fr|de|it, --skip-intro, --mute,
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
WINDOWPLACEMENT g_placement{sizeof(WINDOWPLACEMENT)};
std::filesystem::path g_shots;
int g_shot_number = 0;

const uint16_t kArrow[32] = {
    0x3FFF, 0x1FFF, 0x0FFF, 0x07FF, 0x03FF, 0x01FF, 0x00FF, 0x007F, 0x003F, 0x001F, 0x01FF, 0x10FF, 0x30FF, 0xF87F, 0xF87F, 0xFC7F,
    0x0000, 0x4000, 0x6000, 0x7000, 0x7800, 0x7C00, 0x7E00, 0x7F00, 0x7F80, 0x7C00, 0x6C00, 0x4600, 0x0600, 0x0300, 0x0300, 0x0000,
};

void build_frame(nax::Machine& m, std::vector<uint32_t>& out) {
    out.assign(nax::Presenter::kWidth * nax::Presenter::kHeight, 0xFF000000u);
    if (m.video_mode != 0x13) return;
    uint32_t palette[256];
    for (int i = 0; i < 256; i++) {
        auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
        palette[i] = expand(m.dac[i][0]) | (expand(m.dac[i][1]) << 8) | (expand(m.dac[i][2]) << 16) | 0xFF000000u;
    }
    const uint8_t* vram = m.vram();
    for (size_t i = 0; i < out.size(); i++) out[i] = palette[vram[i]];
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
            if (!screen) p = cursor ? 0xFFFFFFFFu : 0xFF000000u;
            else if (cursor) p ^= 0x00FFFFFFu;
        }
    }
}

void save_shot() {
    if (!g_machine || g_shots.empty()) return;
    std::vector<uint32_t> frame;
    build_frame(*g_machine, frame);
    std::error_code ec;
    std::filesystem::create_directories(g_shots, ec);
    wchar_t name[32];
    swprintf(name, 32, L"shot-%04d.png", g_shot_number++);
    nax::write_png(g_shots / name, frame.data(), nax::Presenter::kWidth, nax::Presenter::kHeight, 2);
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
    g_captured = on;
    if (!on) { ClipCursor(nullptr); return; }
    RECT r;
    GetClientRect(hwnd, &r);
    MapWindowPoints(hwnd, nullptr, reinterpret_cast<POINT*>(&r), 2);
    ClipCursor(&r);
}

void raw_mouse(LPARAM lp) {
    if (!g_captured || !g_presenter || !g_machine) return;
    RAWINPUT input;
    UINT size = sizeof(input);
    if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lp), RID_INPUT, &input, &size, sizeof(RAWINPUTHEADER)) == UINT(-1)) return;
    if (input.header.dwType != RIM_TYPEMOUSE || (input.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) return;
    const RECT r = g_presenter->picture_rect();
    g_rest_x += double(input.data.mouse.lLastX) * 320.0 / std::max<LONG>(1, r.right - r.left);
    g_rest_y += double(input.data.mouse.lLastY) * 200.0 / std::max<LONG>(1, r.bottom - r.top);
    const int mx = int(g_rest_x), my = int(g_rest_y);
    g_rest_x -= mx;
    g_rest_y -= my;
    g_machine->mouse.dx += mx;
    g_machine->mouse.dy += my;
}

void mouse_to_game(LPARAM lp) {
    if (!g_presenter || !g_machine || g_scripted) return;
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
        return 0;
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
        if (wp == VK_F12 && down) { save_shot(); return 0; }
        if (g_machine) g_machine->key_event(uint8_t((lp >> 16) & 0x7F), (lp >> 24) & 1, down);
        return 0;
    }
    case WM_SYSCHAR: case WM_CHAR:
        return 0;
    case WM_MOUSEMOVE:
        mouse_to_game(lp);
        return 0;
    case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_RBUTTONDOWN: case WM_RBUTTONUP: {
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
        MessageBoxW(nullptr, text.c_str(), game::title(), MB_ICONERROR);
        return 1;
    }
    // One copy of a game at a time: a second copy sharing the save folder could remove the first copy's
    // unfinished files at start, or replace a save the first copy is writing.
    const std::wstring mutex_name = std::wstring(L"Local\\NaxGameRestoration-") + game::data_folder_name();
    HANDLE instance_mutex = CreateMutexW(nullptr, TRUE, mutex_name.c_str());
    if (!instance_mutex || GetLastError() == ERROR_ALREADY_EXISTS) {
        MessageBoxW(nullptr, L"The game is already running.", game::title(), MB_ICONINFORMATION);
        return 1;
    }
    wchar_t local[MAX_PATH] = L".";
    GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
    const std::filesystem::path saves = std::filesystem::path(local) / L"Nax Game Restoration" / game::data_folder_name();
    g_shots = saves / L"shots";

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
        MessageBoxA(nullptr, ("Direct3D 12 could not start: " + error).c_str(), "Nax Game Restoration", MB_ICONERROR);
        return 1;
    }
    g_presenter = &presenter;
    RAWINPUTDEVICE mouse_device{0x01, 0x02, 0, g_hwnd};
    RegisterRawInputDevices(&mouse_device, 1, sizeof(mouse_device));
    ShowWindow(g_hwnd, show);

    nax::Machine machine;
    machine.set_folders(data, saves);
    nax::Audio audio;
    std::string audio_error;
    // The chip's timers answer even without a sound device, so the game's AdLib detection still succeeds.
    machine.audio = &audio;
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
    std::vector<uint32_t> frame;
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
        build_frame(machine, frame);
        presenter.draw(frame.data());
    }
    machine.request_stop();
    game_thread.join();
    machine.audio = nullptr;
    audio.stop();
    if (script_thread.joinable()) script_thread.join();
    g_machine = nullptr;
    g_presenter = nullptr;
    if (!machine.failure().empty()) {
        MessageBoxA(nullptr, machine.failure().c_str(), "Nax Game Restoration", MB_ICONERROR);
        return 2;
    }
    return 0;
}
