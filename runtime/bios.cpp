// BIOS video (INT 10h), keyboard (INT 16h), clock (INT 1Ah) and mouse driver (INT 33h) services.
#include <windows.h>

#include <cstring>

#include "machine.h"

namespace nax {

void Machine::int10() {
    switch (c.a.h) {
    case 0x00: {
        const uint8_t mode = c.a.l & 0x7F;
        video_mode = mode;
        wr8(0x40, 0x49, mode);
        if (mode == 0x13 && !(c.a.l & 0x80)) std::memset(&mem[0xA0000], 0, 64000);
        log("Video mode %02x", mode);
        return;
    }
    case 0x0F:
        c.a.l = video_mode;
        c.a.h = video_mode == 0x13 ? 40 : 80;
        c.b.h = 0;
        return;
    case 0x10:
        switch (c.a.l) {
        case 0x10:
            dac[c.b.x & 0xFF][0] = c.d.h & 0x3F;
            dac[c.b.x & 0xFF][1] = c.c.h & 0x3F;
            dac[c.b.x & 0xFF][2] = c.c.l & 0x3F;
            return;
        case 0x12:
            for (uint16_t i = 0; i < c.c.x && c.b.x + i < 256; i++)
                for (int k = 0; k < 3; k++) dac[c.b.x + i][k] = rd8(c.es, uint16_t(c.d.x + i * 3 + k)) & 0x3F;
            return;
        case 0x15:
            c.d.h = dac[c.b.l][0];
            c.c.h = dac[c.b.l][1];
            c.c.l = dac[c.b.l][2];
            return;
        case 0x17:
            for (uint16_t i = 0; i < c.c.x && c.b.x + i < 256; i++)
                for (int k = 0; k < 3; k++) wr8(c.es, uint16_t(c.d.x + i * 3 + k), dac[c.b.x + i][k]);
            return;
        default:
            return;  // EGA palette registers have no effect in the 256-colour mode
        }
    case 0x1A:
        if (c.a.l == 0) { c.a.l = 0x1A; c.b.x = 0x0008; }
        return;
    case 0x12:
        if (c.b.l == 0x10) { c.b.h = 0; c.b.l = 3; c.c.x = 0; }
        return;
    case 0x01: case 0x02: case 0x05: case 0x06: case 0x07: case 0x0B: case 0x09: case 0x0A: return;
    case 0x03: c.c.x = 0x0607; c.d.x = 0; return;
    case 0x08: c.a.x = 0x0720; return;
    case 0x0E: log("BIOS output: %c", c.a.l); return;
    default:
        if (!unknown_services_[0x1000 | c.a.h]) {
            unknown_services_[0x1000 | c.a.h] = true;
            log("Unhandled video function %02x (al=%02x)", c.a.h, c.a.l);
        }
    }
}

void Machine::int16() {
    switch (c.a.h) {
    case 0x00: case 0x10:
        if (bios_keys_.empty()) {
            // Wait by running the INT 16h instruction again, so that timer and keyboard interrupts still arrive.
            c.ip = uint16_t(c.ip - 2);
            Sleep(1);
            return;
        }
        c.a.x = bios_keys_.front();
        bios_keys_.pop_front();
        return;
    case 0x01: case 0x11:
        c.zf = bios_keys_.empty();
        if (!c.zf) c.a.x = bios_keys_.front();
        return;
    case 0x02: case 0x12:
        c.a.l = shift_state_;
        return;
    default:
        return;
    }
}

void Machine::int1a() {
    if (c.a.h == 0x00) {
        c.d.x = rd16(0x40, 0x6C);
        c.c.x = rd16(0x40, 0x6E);
        c.a.l = rd8(0x40, 0x70);
        wr8(0x40, 0x70, 0);
        return;
    }
    SYSTEMTIME st;
    GetLocalTime(&st);
    auto bcd = [](int v) { return uint8_t(((v / 10) << 4) | (v % 10)); };
    if (c.a.h == 0x02) {
        c.c.h = bcd(st.wHour);
        c.c.l = bcd(st.wMinute);
        c.d.h = bcd(st.wSecond);
        c.d.l = 0;
    } else if (c.a.h == 0x04) {
        c.c.h = bcd(st.wYear / 100);
        c.c.l = bcd(st.wYear % 100);
        c.d.h = bcd(st.wMonth);
        c.d.l = bcd(st.wDay);
    }
}

void Machine::mouse_place(int x, int y) {
    const int div = video_mode == 0x13 ? 2 : 1;  // the 256-colour mode counts 640 virtual columns
    mouse.x = std::clamp(x, std::max(0, mouse_min_x_ / div), std::min(319, mouse_max_x_ / div));
    mouse.y = std::clamp(y, std::max(0, mouse_min_y_), std::min(199, mouse_max_y_));
}

void Machine::mouse_motion(int dx, int dy) {
    mouse.dx += dx;
    mouse.dy += dy;
    mouse_place(mouse.x + dx, mouse.y + dy);
}

void Machine::int33() {
    auto virtual_x = [&] {
        const int x = mouse.x * (video_mode == 0x13 ? 2 : 1);
        return std::clamp(x, mouse_min_x_, mouse_max_x_);
    };
    auto virtual_y = [&] { return std::clamp(int(mouse.y), mouse_min_y_, mouse_max_y_); };
    switch (c.a.x) {
    case 0x00: case 0x21:
        c.a.x = 0xFFFF;
        c.b.x = 2;
        cursor_visible = -1;
        mouse.dx = 0;
        mouse.dy = 0;
        mouse_min_x_ = 0; mouse_max_x_ = 639; mouse_min_y_ = 0; mouse_max_y_ = 199;
        return;
    case 0x01:
        if (cursor_visible < 0) cursor_visible++;
        return;
    case 0x02:
        cursor_visible--;
        return;
    case 0x03:
        c.b.x = uint16_t(mouse.buttons.load());
        c.c.x = uint16_t(virtual_x());
        c.d.x = uint16_t(virtual_y());
        return;
    case 0x05: case 0x06:
        c.a.x = uint16_t(mouse.buttons.load());
        c.b.x = 0;
        c.c.x = uint16_t(virtual_x());
        c.d.x = uint16_t(virtual_y());
        return;
    case 0x07:
        mouse_min_x_ = std::min<int>(int16_t(c.c.x), int16_t(c.d.x));
        mouse_max_x_ = std::max<int>(int16_t(c.c.x), int16_t(c.d.x));
        return;
    case 0x08:
        mouse_min_y_ = std::min<int>(int16_t(c.c.x), int16_t(c.d.x));
        mouse_max_y_ = std::max<int>(int16_t(c.c.x), int16_t(c.d.x));
        return;
    case 0x09:
        cursor_hot_x = int16_t(c.b.x);
        cursor_hot_y = int16_t(c.c.x);
        for (int i = 0; i < 32; i++) cursor_shape[i] = rd16(c.es, uint16_t(c.d.x + i * 2));
        custom_cursor = true;
        return;
    case 0x0B:
        // Motion counters in mickeys. Four mickeys per game pixel keep the cursor of a game that divides by
        // four, as Utopia does, in step with the hand on the mouse.
        c.c.x = uint16_t(mouse.dx.exchange(0) * 4);
        c.d.x = uint16_t(mouse.dy.exchange(0) * 4);
        return;
    case 0x24:
        c.b.x = 0x0626;
        c.c.x = 0x0400;
        return;
    case 0x04: case 0x0A: case 0x0F: case 0x13: case 0x1A: case 0x1D:
        return;
    default:
        if (!unknown_services_[0x3300 | (c.a.x & 0xFF)]) {
            unknown_services_[0x3300 | (c.a.x & 0xFF)] = true;
            log("Unhandled mouse function %04x", c.a.x);
        }
    }
}

}  // namespace nax
