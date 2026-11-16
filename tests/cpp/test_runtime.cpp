// Tests for the runtime's DOS file services, sound timing and sprite tracking. Each test drives INT 21h directly on a fresh machine whose game
// folder and save folder are temporary directories, and prints one line per test with PASS or FAIL.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/audio.h"
#include "runtime/hd_sprites.h"
#include "runtime/machine.h"

namespace fs = std::filesystem;

namespace {

struct Fixture {
    fs::path root, data, saves;
    nax::Machine m;

    Fixture() {
        root = fs::temp_directory_path() / (L"nax-runtime-test-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(counter()++));
        data = root / L"game";
        saves = root / L"saves";
        fs::create_directories(data);
        m.set_folders(data, saves);
    }
    ~Fixture() {
        m.finish_files(false);
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    static int& counter() {
        static int n = 0;
        return n;
    }

    void put_string(uint16_t seg, uint16_t off, const std::string& s) {
        for (size_t i = 0; i <= s.size(); i++) m.wr8(seg, uint16_t(off + i), i < s.size() ? uint8_t(s[i]) : 0);
    }
    // Returns the handle, or -1 with the DOS error left in AX.
    int open(const std::string& name, uint8_t function, uint8_t mode) {
        put_string(0x2000, 0, name);
        m.c.ds = 0x2000;
        m.c.d.x = 0;
        m.c.a.h = function;
        m.c.a.l = mode;
        m.c.c.x = 0;
        m.service_for_test(0x21);
        return m.c.cf ? -1 : m.c.a.x;
    }
    bool write(int handle, const std::string& text) {
        for (size_t i = 0; i < text.size(); i++) m.wr8(0x3000, uint16_t(i), uint8_t(text[i]));
        m.c.ds = 0x3000;
        m.c.d.x = 0;
        m.c.b.x = uint16_t(handle);
        m.c.c.x = uint16_t(text.size());
        m.c.a.h = 0x40;
        m.service_for_test(0x21);
        return !m.c.cf && m.c.a.x == text.size();
    }
    bool close(int handle) {
        m.c.b.x = uint16_t(handle);
        m.c.a.h = 0x3E;
        m.service_for_test(0x21);
        return !m.c.cf;
    }
};

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

bool no_temporary_files(const fs::path& dir) {
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.path().wstring().find(L".nax-tmp") != std::wstring::npos) return false;
    return true;
}

// A 32x2 sprite at linear 30000h whose bottom row starts with the colours 1 and 2, drawn with colour base 40h
// so that its bottom row starts at offset 3205 of a buffer at 60000h, and that buffer copied to the screen.
constexpr uint32_t kSpriteData = 0x30000, kBuffer = 0x60000, kDest = kBuffer + 10 * 320 + 5;
void draw_test_sprite(nax::Machine& m, nax::HdSprites& hd) {
    m.mem[kSpriteData] = 0x12;
    hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
    for (uint32_t i = 0; i < 64000; i++) {
        m.mem[0xA0000 + i] = m.mem[kBuffer + i];
        m.origin[0xA0000 + i] = kBuffer + i + 1;
    }
    m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
    m.mem[0xA0000 + 10 * 320 + 6] = 0x42;
}

struct Test {
    const char* name;
    std::function<bool()> run;
};

const std::vector<Test> kTests = {
    {"completed_save_replaces_file_on_close", [] {
         Fixture f;
         write_file(f.saves / L"files" / L"save1.dat", "old");
         const int h = f.open("SAVE1.DAT", 0x3C, 0);
         return h >= 0 && f.write(h, "new game") && f.close(h) && read_file(f.saves / L"files" / L"save1.dat") == "new game" &&
                no_temporary_files(f.saves);
     }},
    {"save_interrupted_by_stop_keeps_previous_version", [] {
         Fixture f;
         write_file(f.saves / L"files" / L"save1.dat", "old");
         const int h = f.open("SAVE1.DAT", 0x3C, 0);
         const bool wrote = h >= 0 && f.write(h, "half");
         f.m.finish_files(false);  // the window was closed before the game closed the file
         return wrote && read_file(f.saves / L"files" / L"save1.dat") == "old" && no_temporary_files(f.saves);
     }},
    {"two_write_handles_to_one_file_keep_separate_temporaries", [] {
         Fixture f;
         const int a = f.open("SAVE2.DAT", 0x3C, 0);
         const bool first = a >= 0 && f.write(a, "first");
         const int b = f.open("SAVE2.DAT", 0x3C, 0);
         const bool second = b >= 0 && f.write(b, "second");
         const bool closed = f.close(a) && f.close(b);
         return first && second && closed && read_file(f.saves / L"files" / L"save2.dat") == "second" && no_temporary_files(f.saves);
     }},
    {"writing_to_game_file_leaves_original_unchanged", [] {
         Fixture f;
         write_file(f.data / L"score.dat", "original");
         const int h = f.open("SCORE.DAT", 0x3D, 2);
         const bool ok = h >= 0 && f.write(h, "CHANGED") && f.close(h);
         // As on DOS, writing at the start of the existing file overwrites only the first bytes of the copy.
         return ok && read_file(f.data / L"score.dat") == "original" && read_file(f.saves / L"files" / L"score.dat") == "CHANGEDl";
     }},
    {"names_with_colons_devices_or_parent_folders_are_refused", [] {
         Fixture f;
         const char* bad[] = {"DIR\\E:X", "SAVE.DAT:STREAM", "CON", "con.txt", "CON ", "CON.", "CON .DAT", "LPT1", "..\\OUTSIDE.DAT", "SCENARIO\\..\\..\\X"};
         for (const char* name : bad)
             if (f.open(name, 0x3C, 0) >= 0) return false;
         return no_temporary_files(f.saves) && !fs::exists(f.root / L"outside.dat");
     }},
    {"search_and_directory_change_stay_inside_the_game_folder", [] {
         Fixture f;
         auto call = [&](uint8_t function, const std::string& name) {
             f.put_string(0x2000, 0, name);
             f.m.c.ds = 0x2000;
             f.m.c.d.x = 0;
             f.m.c.c.x = 0;
             f.m.c.a.h = function;
             f.m.service_for_test(0x21);
             return !f.m.c.cf;
         };
         // A parent folder or a colon inside the path is refused by the safety rule (DOS error 3, path not found),
         // not merely because a folder is missing (which would give error 18, no more files).
         const char* refused[] = {"..\\*.*", "SUB\\..\\..\\*.*", "A\\E:X\\*.*", "A\\CON\\*.*"};
         for (const char* p : refused)
             if (call(0x4E, p) || f.m.c.a.x != 3) return false;
         // A leading backslash, with or without a drive letter, and a network path all stay inside the game folder.
         write_file(f.data / L"x" / L"inside.dat", "game");
         if (!call(0x4E, "\\X\\*.*") || !call(0x4E, "C:\\X\\*.*")) return false;
         if (call(0x4E, "\\\\SRV\\SHARE\\*.*")) return false;
         fs::create_directories(f.data / L"a");
         return !call(0x3B, "A\\E:X") && !call(0x3B, "CON");
     }},
    {"name_with_trailing_space_opens_the_file_as_on_dos", [] {
         Fixture f;
         write_file(f.data / L"option.vgg", "picture");
         const int h = f.open("option.vgg ", 0x3D, 0);  // Utopia passes this exact name
         return h >= 0 && f.close(h);
     }},
    {"unfinished_file_from_crashed_session_is_removed_at_start", [] {
         Fixture f;
         write_file(f.saves / L"files" / L"save1.dat.nax-tmp5", "from a crash");
         nax::Machine next;
         next.set_folders(f.data, f.saves);
         return no_temporary_files(f.saves);
     }},
    {"adlib_detection_answers_without_sound_device", [] {
         nax::Audio audio;  // never started, as with --mute or without an output device
         audio.opl_address(0x04);
         audio.opl_data(0x60);
         audio.opl_data(0x80);
         const bool reset = (audio.opl_status() & 0xE0) == 0;
         audio.opl_address(0x02);
         audio.opl_data(0xFF);
         audio.opl_address(0x04);
         audio.opl_data(0x21);
         uint8_t status = 0;
         for (int i = 0; i < 100; i++) status = audio.opl_status();
         return reset && (status & 0xE0) == 0xC0;
     }},
    {"audio_queue_overflow_keeps_the_complete_chip_state", [] {
         nax::Audio audio;
         audio.accept_events_for_test();
         audio.opl_address(0xB0);
         audio.opl_data(0x31);  // key on, which must survive the overflow
         for (size_t i = 0; i < nax::Audio::kMaxEvents + 10; i++) {
             audio.opl_address(0xA0);
             audio.opl_data(uint8_t(i));
         }
         const size_t queued = audio.queued_events();
         uint8_t key = 0, fnum = 0;
         const bool restored = audio.queued_register(0xB0, &key) && audio.queued_register(0xA0, &fnum);
         return queued < nax::Audio::kMaxEvents && queued >= 256 && restored && key == 0x31 &&
                fnum == uint8_t(nax::Audio::kMaxEvents + 9);
     }},
    {"audio_queue_filled_at_once_keeps_normal_tempo", [] {
         nax::Audio audio;
         const int64_t base = audio.render_time();
         const int64_t span = audio.next_until(base + audio.latency(), 512) - base;
         // After the four queued buffers were rendered in one go, the timeline is four buffers ahead of real
         // time. That is the intended offset, so the next buffer still covers exactly one buffer length.
         return audio.next_until(base + audio.latency() - span * 4, 512) == base + span;
     }},
    {"audio_clock_running_ahead_is_slowed_gently_without_jumping", [] {
         nax::Audio audio;
         const int64_t base = audio.render_time();
         const int64_t span = audio.next_until(base + audio.latency(), 512) - base;
         // The sound card has drifted eight buffers ahead of the system clock, beyond the queue offset.
         const int64_t until = audio.next_until(base + audio.latency() - span * 8, 512);
         return until < base + span && until >= base + span - span / 4;
     }},
    {"audio_clock_running_behind_is_pulled_forward_gently", [] {
         nax::Audio audio;
         const int64_t base = audio.render_time();
         const int64_t span = audio.next_until(base + audio.latency(), 512) - base;
         // The sound card has fallen two buffers behind the system clock. There is no dead band for a lag.
         const int64_t until = audio.next_until(base + audio.latency() + span * 3, 512);
         return until > base + span && until <= base + span + span / 4;
     }},
    {"audio_catches_up_after_a_stall_instead_of_squeezing_it_into_one_buffer", [] {
         nax::Audio audio;
         const int64_t base = audio.render_time();
         const int64_t span = audio.next_until(base + audio.latency(), 512) - base;
         const int64_t stalled = base + audio.latency() + span * 1000;  // ten seconds later
         float out[512];
         audio.render(out, 512, audio.next_until(stalled, 512));
         return span > 0 && audio.render_time() == stalled - audio.latency();
     }},
    {"copied_byte_keeps_the_address_it_was_computed_at", [] {
         nax::Machine m;
         m.wr8(0x1000, 0x0010, 7);
         m.wr8o(0xA000, 0x0000, m.rd8(0x1000, 0x0010), m.org(0x1000, 0x0010));
         m.wr8o(0xA000, 0x0001, 9, 0);
         return m.origin[0xA0000] == 0x10010 + 1 && m.origin[0xA0001] == 0xA0001 + 1;
     }},
    {"sprite_pixel_on_screen_points_into_its_upscaled_picture", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);
         hd.placed(0, 100, 200);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         const uint32_t at = 10 * 320 + 5;
         return hd.sprite_count() == 1 && refs[at] == (100u | (201u << 16)) && refs[at + 1] == (101u | (201u << 16)) &&
                refs[at + 2] == nax::HdSprites::kNone;
     }},
    {"sprite_pixel_drawn_over_or_without_upscaled_picture_keeps_the_original", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         const bool none_before = refs[10 * 320 + 5] == nax::HdSprites::kNone;
         hd.placed(0, 0, 0);
         m.mem[0xA0000 + 10 * 320 + 5] = 0x07;  // something else was drawn there afterwards
         hd.frame(m, refs.data());
         return none_before && refs[10 * 320 + 5] == nax::HdSprites::kNone && refs[10 * 320 + 6] != nax::HdSprites::kNone;
     }},
    {"picture_waits_for_a_steady_palette_and_shows_again_where_the_game_restores_it", [] {
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_screen_source(kBuffer);
         for (uint32_t i = 0; i < 320 * 20; i++) m.mem[kBuffer + i] = 0x07;  // a picture of 20 lines
         hd.image(m, kBuffer, 320, 20, 320, kBuffer, 320);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());  // sees the palette for the first time, so the picture waits
         const bool waited = hd.sprite_count() == 0;
         Sleep(600);
         hd.frame(m, refs.data());  // the palette has stayed the same, so the picture is taken over
         hd.placed(0, 0, 0);
         m.mem[kSpriteData] = 0x12;
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);  // drawn over line 10 at x 5
         hd.placed(1, 400, 0);
         const uint32_t at = 10 * 320 + 5;
         m.mem[0xA0000 + at] = 0x41;  // the sprite on screen
         hd.frame(m, refs.data());
         const bool sprite = refs[at] == (400u | (1u << 16));
         m.mem[0xA0000 + at] = 0x07;  // the game restored the picture
         hd.frame(m, refs.data());
         return waited && sprite && refs[at] == (5u | (10u << 16));
     }},
    {"sprite_drawn_twice_is_saved_once_with_transparency", [] {
         Fixture f;
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         draw_test_sprite(f.m, hd);
         draw_test_sprite(f.m, hd);
         std::vector<fs::path> files;
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"original")) files.push_back(e.path());
         if (files.size() != 1 || hd.sprite_count() != 1) return false;
         std::ifstream in(files[0], std::ios::binary);
         std::string head(26, char(0));
         in.read(head.data(), 26);
         return uint8_t(head[25]) == 6;  // colour type RGBA
     }},
};

}  // namespace

int main() {
    int failed = 0;
    for (const auto& t : kTests) {
        const bool ok = t.run();
        std::printf("tests/cpp/test_runtime.cpp::%s %s\n", t.name, ok ? "PASS" : "FAIL");
        failed += ok ? 0 : 1;
    }
    return failed ? 1 : 0;
}
