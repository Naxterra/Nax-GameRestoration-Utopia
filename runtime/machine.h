// The native runtime that translated DOS programs run on. It owns the 1 MB address space, the CPU registers
// and flags, the interrupt controller and timer, and the DOS, BIOS, mouse and VGA services. Translated code
// calls into it for memory, stack, port and interrupt operations; everything else is plain C++.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace nax {

class Audio;
class Machine;

union Reg16 {
    uint16_t x;
    struct {
        uint8_t l, h;
    };
};

struct Cpu {
    Reg16 a{}, b{}, c{}, d{};
    uint16_t si = 0, di = 0, bp = 0, sp = 0;
    uint16_t cs = 0, ds = 0, es = 0, ss = 0, ip = 0;
    bool cf = false, zf = false, sf = false, of = false, df = false, iflag = true, tf = false;
    // No translated instruction reads PF or AF. They are kept so that a pushed flags word comes back unchanged.
    bool pf = false, af = false;

    uint16_t flags() const {
        return (uint16_t)(0xF002 | (cf ? 0x001 : 0) | (pf ? 0x004 : 0) | (af ? 0x010 : 0) | (zf ? 0x040 : 0) |
                          (sf ? 0x080 : 0) | (tf ? 0x100 : 0) | (iflag ? 0x200 : 0) | (df ? 0x400 : 0) | (of ? 0x800 : 0));
    }
    void set_flags(uint16_t f) {
        cf = f & 0x001; pf = f & 0x004; af = f & 0x010; zf = f & 0x040; sf = f & 0x080;
        tf = f & 0x100; iflag = f & 0x200; df = f & 0x400; of = f & 0x800;
    }
};

// Describes one translated program or driver. The generator writes one of these per module.
struct ModuleInfo {
    const char* source;
    void (*run)(Machine&, uint16_t base);
    const uint16_t* segments;
    size_t segment_count;
    uint32_t crc;
    uint32_t size;
    bool flat;
    const uint8_t* signature;
    size_t signature_size;
    uint16_t cs, ip, ss, sp;
    uint32_t min_extra_paragraphs;
};

constexpr uint16_t kBiosSegment = 0xF000;
constexpr uint32_t kMemorySize = 0x110000;
constexpr uint16_t kFirstFreeSegment = 0x0800;
constexpr uint16_t kMemoryTop = 0xA000;

struct MouseState {
    std::atomic<int> x{0}, y{0};  // position inside the game picture, 0..319 and 0..199
    std::atomic<int> buttons{0};
    std::atomic<int> dx{0}, dy{0};  // movement in game pixels not yet read through INT 33h function 0Bh
};

class Machine {
public:
    Machine();
    ~Machine();

    Cpu c;
    std::vector<uint8_t> mem;

    // Memory and stack. Addresses wrap at 1 MB as on a PC with the A20 line disabled.
    uint8_t rd8(uint16_t s, uint16_t o) const { return mem[((uint32_t(s) << 4) + o) & 0xFFFFF]; }
    uint16_t rd16(uint16_t s, uint16_t o) const { return uint16_t(rd8(s, o) | (rd8(s, uint16_t(o + 1)) << 8)); }
    void wr8(uint16_t s, uint16_t o, uint8_t v) { mem[((uint32_t(s) << 4) + o) & 0xFFFFF] = v; }
    void wr16(uint16_t s, uint16_t o, uint16_t v) { wr8(s, o, uint8_t(v)); wr8(s, uint16_t(o + 1), uint8_t(v >> 8)); }
    void push(uint16_t v) { c.sp = uint16_t(c.sp - 2); wr16(c.ss, c.sp, v); }
    uint16_t pop() { const uint16_t v = rd16(c.ss, c.sp); c.sp = uint16_t(c.sp + 2); return v; }
    void iret() { c.ip = pop(); c.cs = pop(); c.set_flags(pop()); }

    // Called by translated code.
    void interrupt(uint8_t n);
    bool poll();
    bool service_pending();
    uint8_t in8(uint16_t port);
    void out8(uint16_t port, uint8_t v);
    uint16_t in16(uint16_t port) { return uint16_t(in8(port) | (in8(uint16_t(port + 1)) << 8)); }
    void out16(uint16_t port, uint16_t v) { out8(port, uint8_t(v)); out8(uint16_t(port + 1), uint8_t(v >> 8)); }
    void untranslated(uint16_t base);
    void halt_cpu();
    void divide_error();

    // Set up by the host before a program runs.
    void add_module(const ModuleInfo* info) { modules_.push_back(info); }
    void set_folders(const std::filesystem::path& data, const std::filesystem::path& saves);
    int run_program(const ModuleInfo* program, const std::string& tail);
    void request_stop() { stop_requested_ = true; }
    bool stopped() const { return stop_requested_; }
    std::string failure() const { return failure_; }
    void log(const char* fmt, ...);
    // Closes every open file. Writes are kept only when the program ended normally; after a stop or an error
    // the temporary files are deleted and the previous versions stay as they were.
    void finish_files(bool commit);
    // Runs one DOS or BIOS service directly, for the runtime tests.
    void service_for_test(uint8_t n) { native_interrupt(n); }

    // Sound output, or null when no sound device is available.
    Audio* audio = nullptr;

    // Shared with the window thread.
    uint8_t dac[256][3]{};  // VGA palette, 6 bits per component
    std::atomic<uint8_t> video_mode{3};
    MouseState mouse;
    std::atomic<int> cursor_visible{-1};  // mouse driver show counter; the cursor is drawn while it is zero
    uint16_t cursor_shape[32]{};          // screen mask then cursor mask, as set by INT 33h function 9
    int cursor_hot_x = 0, cursor_hot_y = 0;
    bool custom_cursor = false;
    void key_event(uint8_t scancode, bool extended, bool down);
    // Adds mouse movement in game pixels to the motion counters and to the position, which stays inside the
    // range the game set, as a DOS mouse driver keeps it.
    void mouse_motion(int dx, int dy);
    // Places the mouse at a position in game pixels, inside the game's range.
    void mouse_place(int x, int y);
    const uint8_t* vram() const { return &mem[0xA0000]; }

private:
    struct Instance {
        const ModuleInfo* info;
        uint16_t base;
    };
    struct Block {
        uint16_t seg;
        uint16_t paras;
    };
    struct File {
        FILE* f = nullptr;
        std::filesystem::path final_path;  // set when the file was opened for writing
        std::filesystem::path temp_path;
        std::string name;
    };

    const Instance* find_module(uint16_t cs);
    bool load_executable(const ModuleInfo* program, const std::string& tail);
    void fatal(const std::string& message);
    void bios_stub(uint16_t ip);
    void native_interrupt(uint8_t n);
    void int10();
    void int16();
    void int1a();
    void int21();
    void int33();
    void find_entry(const std::string& name, uintmax_t size);
    void keyboard_irq();
    void update_timers();
    bool irq_ready(int* which) const;
    void deliver_irq(int n);
    void eoi(int n);
    uint32_t far_vector(uint8_t n) const { return rd16(0, uint16_t(n * 4)) | (uint32_t(rd16(0, uint16_t(n * 4 + 2))) << 16); }
    bool default_vector(uint8_t n) const { return far_vector(n) == ((uint32_t(kBiosSegment) << 16) | n); }
    std::string read_string(uint16_t s, uint16_t o, size_t max = 128) const;
    bool resolve(const std::string& dos_name, bool for_write, std::filesystem::path* out);
    void dos_error(uint16_t code) { c.cf = true; c.a.x = code; }
    bool close_file(int handle, bool commit);
    uint16_t alloc(uint16_t paras, uint16_t* largest);
    bool resize(uint16_t seg, uint16_t paras, uint16_t* largest);
    bool free_block(uint16_t seg);

    std::vector<const ModuleInfo*> modules_;
    std::vector<Instance> instances_;
    std::vector<Block> blocks_;
    std::map<int, File> files_;
    std::filesystem::path data_dir_, save_dir_;
    std::string cwd_;  // current DOS directory relative to the game folder, with forward slashes
    uint16_t dta_seg_ = 0, dta_off_ = 0x80;
    std::vector<std::string> find_results_;
    size_t find_next_ = 0;
    std::atomic<bool> stop_requested_{false};
    bool running_ = false;
    int exit_code_ = 0;
    std::string failure_;
    FILE* log_ = nullptr;
    std::mutex log_mutex_;

    // Interrupt controller and timer.
    uint8_t irr_ = 0, isr_ = 0, imr_ = 0;
    int64_t qpc_freq_ = 1, next_irq0_ = 0, period0_ = 0;
    uint16_t pit_reload_[3] = {0, 0, 0};
    uint8_t pit_access_[3] = {3, 3, 3};
    bool pit_low_written_[3] = {false, false, false};
    int64_t pit_start_[3] = {0, 0, 0};
    uint8_t port61_ = 0;
    int poll_counter_ = 0;

    // Keyboard.
    std::mutex key_mutex_;
    std::deque<uint8_t> key_queue_;
    uint8_t kbd_data_ = 0;
    std::deque<uint16_t> bios_keys_;
    bool e0_prefix_ = false;
    uint8_t shift_state_ = 0;

    // VGA.
    uint8_t dac_write_index_ = 0, dac_read_index_ = 0, dac_component_ = 0, dac_read_component_ = 0;
    uint8_t vga_index_[0x20]{};
    std::map<uint16_t, bool> unknown_ports_;
    std::map<uint16_t, bool> unknown_services_;

    // Mouse driver.
    int mouse_min_x_ = 0, mouse_max_x_ = 639, mouse_min_y_ = 0, mouse_max_y_ = 199;
};

}  // namespace nax
