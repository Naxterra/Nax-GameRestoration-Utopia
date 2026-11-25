// Machine core: program loading, module dispatch, interrupts, the interrupt controller, the timer and port I/O.
#include "machine.h"

#include <windows.h>

#include "audio.h"

#include <cstdarg>
#include <cstring>

namespace nax {

namespace {

int64_t qpc_now() {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return v.QuadPart;
}

uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; i++) {
        crc ^= data[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1)));
    }
    return ~crc;
}

constexpr double kPitHz = 1193181.666;

}  // namespace

Machine::Machine() : mem(kMemorySize, 0), origin(kMemorySize, 0), written_by(kMemorySize, 0) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    qpc_freq_ = f.QuadPart;
    for (int n = 0; n < 256; n++) {
        wr16(0, uint16_t(n * 4), uint16_t(n));
        wr16(0, uint16_t(n * 4 + 2), kBiosSegment);
    }
    wr16(0x40, 0x10, 0x0021);     // equipment: colour 80x25, one floppy bit kept for old checks
    wr16(0x40, 0x13, 640);        // conventional memory in KB
    wr8(0x40, 0x49, 3);           // current video mode
    wr16(0x40, 0x4A, 80);         // text columns
    wr16(0x40, 0x63, 0x3D4);      // CRT controller port of a colour adapter; programs add 6 for the status port
    wr8(0x40, 0x84, 24);          // text rows minus one
    wr8(0x40, 0x87, 0x60);        // EGA/VGA information bytes of a VGA card with 256 KB
    wr8(0x40, 0x88, 0x09);
    wr8(0x40, 0x89, 0x11);
    wr8(kBiosSegment, 0xFFFE, 0xFC);  // machine model byte: AT class
    // Standard VGA palette for the first 16 colours; the game sets the rest itself.
    static const uint8_t ega[16][3] = {{0, 0, 0}, {0, 0, 42}, {0, 42, 0}, {0, 42, 42}, {42, 0, 0}, {42, 0, 42}, {42, 21, 0}, {42, 42, 42},
                                       {21, 21, 21}, {21, 21, 63}, {21, 63, 21}, {21, 63, 63}, {63, 21, 21}, {63, 21, 63}, {63, 63, 21}, {63, 63, 63}};
    std::memcpy(dac, ega, sizeof(ega));
    period0_ = int64_t(qpc_freq_ * 65536.0 / kPitHz);
}

Machine::~Machine() {
    for (auto& [handle, file] : files_) {
        if (file.f) std::fclose(file.f);
        if (!file.temp_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(file.temp_path, ec);
        }
    }
    if (log_) std::fclose(log_);
}

void Machine::set_folders(const std::filesystem::path& data, const std::filesystem::path& saves) {
    data_dir_ = data;
    save_dir_ = saves;
    std::error_code ec;
    std::filesystem::create_directories(save_dir_, ec);
    log_ = _wfopen((save_dir_ / L"runtime.log").c_str(), L"wN");
    std::filesystem::recursive_directory_iterator it(save_dir_ / L"files", ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        if (it->path().filename().wstring().find(L".nax-tmp") == std::wstring::npos) continue;
        std::error_code remove_ec;
        if (std::filesystem::remove(it->path(), remove_ec)) log("Removed an unfinished file left by an earlier session");
        else log("Could not remove an unfinished file left by an earlier session");
    }
}

void Machine::log(const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (!log_) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(log_, fmt, args);
    va_end(args);
    std::fputc('\n', log_);
    std::fflush(log_);
}

void Machine::fatal(const std::string& message) {
    if (failure_.empty()) failure_ = message;
    log("FATAL: %s", message.c_str());
    running_ = false;
    stop_requested_ = true;
}

// Program loading ------------------------------------------------------------------------------------------

namespace {

bool unpack_exepack(std::vector<uint8_t>& body, uint16_t& cs, uint16_t& ip, uint16_t& ss, uint16_t& sp,
                    std::vector<std::pair<uint16_t, uint16_t>>& relocs, uint32_t& extra, std::string& error) {
    const size_t stub = size_t(cs) * 16;
    if (stub + 18 > body.size()) { error = "EXEPACK header outside the program"; return false; }
    auto word = [&](size_t at) { return uint16_t(body[at] | (body[at + 1] << 8)); };
    const uint16_t real_ip = word(stub), real_cs = word(stub + 2), exepack_size = word(stub + 6);
    const uint16_t real_sp = word(stub + 8), real_ss = word(stub + 10), dest_len = word(stub + 12);
    const uint16_t skip_len = ip == 0x10 ? 1 : word(stub + 14);
    const size_t compressed = stub - size_t(skip_len - 1) * 16;
    const size_t dest = size_t(dest_len) * 16;
    if (compressed > body.size()) { error = "EXEPACK compressed length is impossible"; return false; }
    std::vector<uint8_t> buf(std::max(dest, compressed), 0);
    std::memcpy(buf.data(), body.data(), compressed);
    size_t src = compressed;
    for (int i = 0; i < 16 && src > 0 && buf[src - 1] == 0xFF; i++) src--;
    size_t dst = dest;
    for (;;) {
        if (src < 3) { error = "EXEPACK stream ends early"; return false; }
        const uint8_t command = buf[src - 1];
        const size_t length = buf[src - 3] | (buf[src - 2] << 8);
        src -= 3;
        if ((command & 0xFE) == 0xB0) {
            if (src < 1) { error = "EXEPACK fill without a byte"; return false; }
            const uint8_t fill = buf[--src];
            if (dst < length || dst - length < src) { error = "EXEPACK fill overruns its input"; return false; }
            std::memset(&buf[dst - length], fill, length);
            dst -= length;
        } else if ((command & 0xFE) == 0xB2) {
            if (src < length || dst < src) { error = "EXEPACK copy overruns its input"; return false; }
            for (size_t i = 0; i < length; i++) buf[--dst] = buf[--src];
        } else {
            error = "unknown EXEPACK command";
            return false;
        }
        if (command & 1) break;
    }
    const char marker[] = "Packed file is corrupt";
    const size_t end = std::min(body.size(), stub + exepack_size);
    size_t pos = 0;
    for (size_t i = stub; i + sizeof(marker) - 1 <= end; i++) {
        if (std::memcmp(&body[i], marker, sizeof(marker) - 1) == 0) { pos = i + sizeof(marker) - 1; break; }
    }
    if (pos == 0) { error = "EXEPACK relocation table not found"; return false; }
    relocs.clear();
    for (int group = 0; group < 16; group++) {
        if (pos + 2 > end) { error = "EXEPACK relocation table overruns"; return false; }
        const uint16_t count = word(pos);
        pos += 2;
        if (pos + size_t(count) * 2 > end) { error = "EXEPACK relocation table overruns"; return false; }
        for (int k = 0; k < count; k++, pos += 2) relocs.emplace_back(uint16_t(group * 0x1000), word(pos));
    }
    const uint32_t packed_paras = uint32_t((body.size() + 15) / 16) + extra;
    extra = packed_paras > dest_len ? packed_paras - dest_len : 0;
    buf.resize(dest);
    body.swap(buf);
    cs = real_cs; ip = real_ip; ss = real_ss; sp = real_sp;
    return true;
}

}  // namespace

bool Machine::load_executable(const ModuleInfo* program, const std::string& tail) {
    std::filesystem::path path;
    if (!resolve(program->source, false, &path)) {
        fatal(std::string("The game file ") + program->source + " was not found in the game folder.");
        return false;
    }
    FILE* f = _wfopen(path.c_str(), L"rbN");
    if (!f) { fatal(std::string("Cannot open ") + program->source); return false; }
    std::vector<uint8_t> raw;
    uint8_t chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) raw.insert(raw.end(), chunk, chunk + n);
    std::fclose(f);
    if (crc32(raw.data(), raw.size()) != program->crc) {
        fatal(std::string(program->source) + " differs from the copy this build was translated from. Rebuild the game from this copy.");
        return false;
    }
    auto word = [&](size_t at) { return uint16_t(raw[at] | (raw[at + 1] << 8)); };
    const uint16_t last = word(2), pages = word(4), nrel = word(6), hdr = word(8), min_alloc = word(10);
    uint16_t ss = word(14), sp = word(16), ip = word(20), cs = word(22);
    const uint16_t rel_off = word(24);
    const size_t size = size_t(pages) * 512 - (last ? 512 - last : 0);
    std::vector<uint8_t> body(raw.begin() + hdr * 16, raw.begin() + std::min(size, raw.size()));
    std::vector<std::pair<uint16_t, uint16_t>> relocs;
    for (int i = 0; i < nrel; i++) relocs.emplace_back(word(rel_off + 4 * i + 2), word(rel_off + 4 * i));
    uint32_t extra = min_alloc;
    if ((ip == 0x10 || ip == 0x12) && size_t(cs) * 16 + ip <= body.size() && body[cs * 16 + ip - 2] == 'R' && body[cs * 16 + ip - 1] == 'B') {
        std::string error;
        if (!unpack_exepack(body, cs, ip, ss, sp, relocs, extra, error)) { fatal(error); return false; }
    }

    blocks_.clear();
    const uint16_t env = kFirstFreeSegment;
    const uint16_t psp = uint16_t(env + 0x10);
    const uint16_t load = uint16_t(psp + 0x10);
    const uint32_t need = uint32_t((body.size() + 15) / 16) + extra;
    if (load + need > kMemoryTop) { fatal("The program needs more conventional memory than 640 KB provides."); return false; }
    blocks_.push_back({env, 0x10});
    blocks_.push_back({psp, uint16_t(kMemoryTop - psp)});
    std::memset(&mem[uint32_t(env) << 4], 0, (uint32_t(kMemoryTop) - env) << 4);
    const char environment[] = "COMSPEC=C:\\COMMAND.COM\0PATH=C:\\\0\0\x01\0C:\\GAME\\PROGRAM.EXE";
    std::memcpy(&mem[uint32_t(env) << 4], environment, sizeof(environment));
    wr16(psp, 0x00, 0x20CD);
    wr16(psp, 0x02, kMemoryTop);
    wr16(psp, 0x2C, env);
    wr8(psp, 0x80, uint8_t(tail.size()));
    for (size_t i = 0; i < tail.size(); i++) wr8(psp, uint16_t(0x81 + i), uint8_t(tail[i]));
    wr8(psp, uint16_t(0x81 + tail.size()), 0x0D);
    std::memcpy(&mem[uint32_t(load) << 4], body.data(), body.size());
    for (auto [seg, off] : relocs) wr16(uint16_t(load + seg), off, uint16_t(rd16(uint16_t(load + seg), off) + load));

    c = Cpu{};
    c.cs = uint16_t(load + cs);
    c.ip = ip;
    c.ss = uint16_t(load + ss);
    c.sp = sp;
    c.ds = c.es = psp;
    instances_.clear();
    instances_.push_back({program, load});
    dta_seg_ = psp;
    dta_off_ = 0x80;
    cwd_.clear();
    log("Loaded %s at segment %04x, entry %04x:%04x, %u bytes, %zu relocations", program->source, load, c.cs, c.ip,
        unsigned(body.size()), relocs.size());
    return true;
}

int Machine::run_program(const ModuleInfo* program, const std::string& tail) {
    if (!load_executable(program, tail)) return -1;
    running_ = true;
    exit_code_ = 0;
    while (running_ && !stop_requested_) {
        if (c.cs == kBiosSegment) {
            // A stub that waits stays where it is, so interrupts have to be delivered here as well.
            if (service_pending()) break;
            if (c.cs == kBiosSegment) bios_stub(c.ip);
            continue;
        }
        const Instance* inst = find_module(c.cs);
        if (!inst) {
            untranslated(0);
            break;
        }
        inst->info->run(*this, inst->base);
    }
    finish_files(failure_.empty() && !stop_requested_);
    return failure_.empty() ? exit_code_ : -1;
}

const Machine::Instance* Machine::find_module(uint16_t cs) {
    auto has_segment = [](const ModuleInfo* info, uint16_t rel) {
        for (size_t i = 0; i < info->segment_count; i++)
            if (info->segments[i] == rel) return true;
        return false;
    };
    auto signature_matches = [&](const ModuleInfo* info, uint16_t base) {
        const uint32_t at = uint32_t(base) << 4;
        return at + info->signature_size <= kMemorySize && std::memcmp(&mem[at], info->signature, info->signature_size) == 0;
    };
    for (auto& inst : instances_) {
        if (!has_segment(inst.info, uint16_t(cs - inst.base))) continue;
        if (inst.info->flat && !signature_matches(inst.info, inst.base)) continue;
        return &inst;
    }
    // A driver the program has just loaded into memory is recognised by its first bytes.
    for (const ModuleInfo* info : modules_) {
        if (!info->flat || !info->signature_size) continue;
        for (size_t i = 0; i < info->segment_count; i++) {
            const uint16_t base = uint16_t(cs - info->segments[i]);
            if (signature_matches(info, base)) {
                instances_.push_back({info, base});
                log("Recognised driver %s loaded at segment %04x", info->source, base);
                return &instances_.back();
            }
        }
    }
    return nullptr;
}

void Machine::untranslated(uint16_t base) {
    char text[256];
    std::snprintf(text, sizeof(text),
                  "The game reached code at %04X:%04X (offset %04X:%04X from its load segment) that was not translated. "
                  "Add this address to the seed list and rebuild.",
                  c.cs, c.ip, uint16_t(c.cs - (base ? base : (instances_.empty() ? 0 : instances_[0].base))), c.ip);
    fatal(text);
}

void Machine::halt_cpu() {
    if (!c.iflag) { fatal("The game halted the processor with interrupts disabled."); return; }
    Sleep(1);
}

void Machine::divide_error() { fatal("The game divided by zero."); }

// Interrupts -----------------------------------------------------------------------------------------------

void Machine::interrupt(uint8_t n) {
    if (default_vector(n) && n != 0x08 && n != 0x09) {
        native_interrupt(n);
        return;
    }
    push(c.flags());
    push(c.cs);
    push(c.ip);
    c.iflag = false;
    c.tf = false;
    const uint32_t v = far_vector(n);
    c.ip = uint16_t(v);
    c.cs = uint16_t(v >> 16);
}

void Machine::bios_stub(uint16_t ip) {
    // Entered through an interrupt frame (a hardware interrupt, or a chained call to an old vector).
    switch (ip) {
    case 0x08: {
        const uint32_t ticks = (rd16(0x40, 0x6C) | (uint32_t(rd16(0x40, 0x6E)) << 16)) + 1;
        if (ticks >= 0x1800B0) {
            wr16(0x40, 0x6C, 0);
            wr16(0x40, 0x6E, 0);
            wr8(0x40, 0x70, 1);
        } else {
            wr16(0x40, 0x6C, uint16_t(ticks));
            wr16(0x40, 0x6E, uint16_t(ticks >> 16));
        }
        if (!default_vector(0x1C)) {
            // Call the user tick handler, then send the end of interrupt and return, as the PC BIOS does.
            push(c.flags());
            push(kBiosSegment);
            push(0x108);
            c.iflag = false;
            const uint32_t v = far_vector(0x1C);
            c.ip = uint16_t(v);
            c.cs = uint16_t(v >> 16);
            return;
        }
        eoi(0);
        iret();
        return;
    }
    case 0x108:
        eoi(0);
        iret();
        return;
    case 0x09:
        keyboard_irq();
        eoi(1);
        iret();
        return;
    default:
        if (ip == 0x16 && (c.a.h == 0x00 || c.a.h == 0x10) && bios_keys_.empty()) {
            // A chained keyboard read waits at the stub instead of rewinding the caller's instruction. Like the
            // BIOS it enables interrupts while it waits, so the keystroke can arrive; the iret restores the flags.
            c.iflag = true;
            Sleep(1);
            return;
        }
        if (ip < 0x100) {
            // BIOS and DOS services return their results in CF and ZF, so those survive the return like "retf 2".
            native_interrupt(uint8_t(ip));
            const bool cf = c.cf, zf = c.zf;
            if (!running_) return;
            iret();
            c.cf = cf;
            c.zf = zf;
            return;
        }
        fatal("The game jumped into an unknown BIOS address.");
    }
}

void Machine::native_interrupt(uint8_t n) {
    switch (n) {
    case 0x10: int10(); break;
    case 0x11: c.a.x = rd16(0x40, 0x10); break;
    case 0x12: c.a.x = rd16(0x40, 0x13); break;
    case 0x15: c.cf = true; c.a.h = 0x86; break;
    case 0x16: int16(); break;
    case 0x1A: int1a(); break;
    case 0x21: int21(); break;
    case 0x33: int33(); break;
    case 0x2F: c.a.l = 0; break;
    case 0x1C: case 0x23: case 0x24: case 0x28: break;
    default:
        if (n >= 0x60 && n <= 0x67) break;  // unused user vectors: the drivers install their own when loaded
        if (!unknown_services_[n]) {
            unknown_services_[n] = true;
            log("Unhandled interrupt %02x (ax=%04x)", n, c.a.x);
        }
    }
}

// Interrupt controller and timer ---------------------------------------------------------------------------

void Machine::update_timers() {
    const int64_t now = qpc_now();
    if (next_irq0_ == 0) next_irq0_ = now + period0_;
    if (now >= next_irq0_) {
        irr_ |= 1;
        next_irq0_ += period0_;
        if (now - next_irq0_ > period0_ * 8) next_irq0_ = now + period0_;
    }
    if (!(irr_ & 2) && !(isr_ & 2)) {
        std::lock_guard<std::mutex> lock(key_mutex_);
        if (!key_queue_.empty()) {
            kbd_data_ = key_queue_.front();
            key_queue_.pop_front();
            irr_ |= 2;
        }
    }
}

bool Machine::irq_ready(int* which) const {
    const uint8_t pending = uint8_t(irr_ & ~imr_);
    for (int n = 0; n < 8; n++) {
        if (isr_ & (1 << n)) return false;  // an equal or higher priority interrupt is still being serviced
        if (pending & (1 << n)) { *which = n; return true; }
    }
    return false;
}

void Machine::deliver_irq(int n) {
    irr_ &= uint8_t(~(1 << n));
    isr_ |= uint8_t(1 << n);
    push(c.flags());
    push(c.cs);
    push(c.ip);
    c.iflag = false;
    c.tf = false;
    const uint32_t v = far_vector(uint8_t(8 + n));
    c.ip = uint16_t(v);
    c.cs = uint16_t(v >> 16);
}

void Machine::eoi(int n) {
    if (n < 0) {
        for (int k = 0; k < 8; k++)
            if (isr_ & (1 << k)) { isr_ &= uint8_t(~(1 << k)); return; }
        return;
    }
    isr_ &= uint8_t(~(1 << n));
}

bool Machine::poll() {
    if (stop_requested_ || !running_) return true;
    if (++poll_counter_ & 15) return false;
    update_timers();
    int n;
    return c.iflag && irq_ready(&n);
}

bool Machine::service_pending() {
    if (stop_requested_ || !running_) return true;
    update_timers();
    int n;
    if (c.iflag && irq_ready(&n)) deliver_irq(n);
    return false;
}

// Keyboard -------------------------------------------------------------------------------------------------

void Machine::key_event(uint8_t scancode, bool extended, bool down) {
    std::lock_guard<std::mutex> lock(key_mutex_);
    if (key_queue_.size() > 64) return;
    if (extended) key_queue_.push_back(0xE0);
    key_queue_.push_back(uint8_t(scancode | (down ? 0 : 0x80)));
}

namespace {
// Scancode to ASCII for the BIOS key buffer, without and with shift.
const char kAscii[2][0x3A] = {
    {0, 27, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 8, 9, 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 13, 0,
     'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', '`', 0, '\\', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' '},
    {0, 27, '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', 8, 9, 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', 13, 0,
     'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"', '~', 0, '|', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?', 0, '*', 0, ' '},
};
}  // namespace

void Machine::keyboard_irq() {
    const uint8_t code = kbd_data_;
    if (code == 0xE0) { e0_prefix_ = true; return; }
    const bool release = code & 0x80;
    const uint8_t key = code & 0x7F;
    if (key == 0x2A || key == 0x36) {
        shift_state_ = release ? uint8_t(shift_state_ & ~(key == 0x2A ? 2 : 1)) : uint8_t(shift_state_ | (key == 0x2A ? 2 : 1));
    } else if (key == 0x1D) {
        shift_state_ = release ? uint8_t(shift_state_ & ~4) : uint8_t(shift_state_ | 4);
    } else if (key == 0x38) {
        shift_state_ = release ? uint8_t(shift_state_ & ~8) : uint8_t(shift_state_ | 8);
    } else if (!release && bios_keys_.size() < 16) {
        const bool shift = shift_state_ & 3;
        const uint8_t ascii = (!e0_prefix_ && key < 0x3A) ? uint8_t(kAscii[shift ? 1 : 0][key]) : 0;
        bios_keys_.push_back(uint16_t((key << 8) | ascii));
    }
    wr8(0x40, 0x17, shift_state_);
    e0_prefix_ = false;
}

// Ports ----------------------------------------------------------------------------------------------------

uint8_t Machine::in8(uint16_t port) {
    switch (port) {
    case 0x20: return isr_;
    case 0x21: return imr_;
    case 0x40: case 0x41: case 0x42: {
        const int ch = port - 0x40;
        const uint32_t reload = pit_reload_[ch] ? pit_reload_[ch] : 0x10000;
        const int64_t elapsed = int64_t(double(qpc_now() - pit_start_[ch]) * kPitHz / double(qpc_freq_));
        const uint16_t count = uint16_t(reload - uint32_t(elapsed % reload));
        pit_low_written_[ch] = !pit_low_written_[ch];
        return pit_low_written_[ch] ? uint8_t(count) : uint8_t(count >> 8);
    }
    case 0x60: return kbd_data_;
    case 0x61:
        port61_ ^= 0x10;  // refresh toggle, which some delay loops wait for
        return port61_;
    case 0x3C7: return 3;
    case 0x3C9: {
        const uint8_t v = dac[dac_read_index_][dac_read_component_];
        if (++dac_read_component_ == 3) { dac_read_component_ = 0; dac_read_index_++; }
        return v;
    }
    case 0x3DA: case 0x3BA: {
        // Vertical retrace for about 1.3 ms of every 70 Hz frame.
        const int64_t frame = qpc_freq_ / 70;
        const int64_t phase = qpc_now() % frame;
        const bool retrace = phase < frame / 11;
        const bool display_off = retrace || (phase / (frame / 400)) % 2 == 0;
        return uint8_t((retrace ? 0x08 : 0) | (display_off ? 0x01 : 0));
    }
    case 0x3C4: case 0x3CE: case 0x3D4: return vga_index_[port & 0x1F];
    case 0x388: return audio ? audio->opl_status() : 0xFF;
    case 0x389: return 0xFF;
    default:
        if (!unknown_ports_[port]) { unknown_ports_[port] = true; log("Read from unhandled port %04x", port); }
        return 0xFF;
    }
}

void Machine::out8(uint16_t port, uint8_t v) {
    switch (port) {
    case 0x20:
        if (v == 0x20) eoi(-1);
        else if ((v & 0xF8) == 0x60) eoi(v & 7);
        return;
    case 0x21: imr_ = v; return;
    case 0x43: {
        const int ch = v >> 6;
        if (ch < 3) {
            pit_access_[ch] = uint8_t((v >> 4) & 3);
            pit_low_written_[ch] = false;
        }
        return;
    }
    case 0x40: case 0x41: case 0x42: {
        const int ch = port - 0x40;
        if (pit_access_[ch] == 1) pit_reload_[ch] = v;
        else if (pit_access_[ch] == 2) pit_reload_[ch] = uint16_t(v << 8);
        else if (!pit_low_written_[ch]) { pit_reload_[ch] = uint16_t((pit_reload_[ch] & 0xFF00) | v); pit_low_written_[ch] = true; return; }
        else { pit_reload_[ch] = uint16_t((pit_reload_[ch] & 0x00FF) | (v << 8)); pit_low_written_[ch] = false; }
        pit_start_[ch] = qpc_now();
        if (ch == 2 && audio) audio->speaker(port61_ & 1, port61_ & 2, pit_reload_[2]);
        if (ch == 0) {
            const uint32_t reload = pit_reload_[0] ? pit_reload_[0] : 0x10000;
            period0_ = int64_t(double(qpc_freq_) * reload / kPitHz);
            next_irq0_ = qpc_now() + period0_;
            log("Timer set to %.1f Hz", kPitHz / reload);
        }
        return;
    }
    case 0x61:
        port61_ = uint8_t((port61_ & 0x10) | (v & 0xEF));
        if (audio) audio->speaker(port61_ & 1, port61_ & 2, pit_reload_[2]);
        return;
    case 0x3C8: dac_write_index_ = v; dac_component_ = 0; return;
    case 0x3C7: dac_read_index_ = v; dac_read_component_ = 0; return;
    case 0x3C9:
        dac[dac_write_index_][dac_component_] = uint8_t(v & 0x3F);
        if (++dac_component_ == 3) { dac_component_ = 0; dac_write_index_++; }
        return;
    case 0x3C4: case 0x3CE: case 0x3D4: vga_index_[port & 0x1F] = v; return;
    case 0x3C5: case 0x3CF: case 0x3D5: case 0x3C0: case 0x3C2: return;
    case 0x388: if (audio) audio->opl_address(v); return;
    case 0x389: if (audio) audio->opl_data(v); return;
    default:
        if (!unknown_ports_[port]) { unknown_ports_[port] = true; log("Write %02x to unhandled port %04x", v, port); }
    }
}

}  // namespace nax
