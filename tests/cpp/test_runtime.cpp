// Tests for the runtime's DOS file services, sound timing and sprite tracking. Each test drives INT 21h directly on a fresh machine whose game
// folder and save folder are temporary directories, and prints one line per test with PASS or FAIL.
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/ai_upscaler.h"
#include "runtime/audio.h"
#include "runtime/hd_sprites.h"
#include "runtime/machine.h"
#include "runtime/world_view.h"

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

// The file name earlier versions gave the test sprite: its pixels, its width and all 15 colours of its bank.
std::wstring legacy_name(nax::Machine& m) {
    m.mem[kSpriteData] = 0x12;
    uint64_t legacy = 0xcbf29ce484222325ull;
    auto mix = [&legacy](uint8_t v) { legacy = (legacy ^ v) * 0x100000001b3ull; };
    for (uint32_t i = 0; i < 32; i++) mix(m.mem[kSpriteData + i]);
    mix(32);
    for (int c = 1; c < 16; c++)
        for (int k = 0; k < 3; k++) mix(m.dac[0x40 + c][k]);
    wchar_t name[32];
    swprintf(name, 32, L"%016llx.png", static_cast<unsigned long long>(legacy));
    return name;
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
         return hd.sprite_count() == 1 && refs[at] == nax::HdSprites::encode(100, 201, 1) && refs[at + 1] == nax::HdSprites::encode(101, 201, 1) &&
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
         const bool sprite = refs[at] == nax::HdSprites::encode(400, 1, 1);
         m.mem[0xA0000 + at] = 0x07;  // the game restored the picture
         hd.frame(m, refs.data());
         return waited && sprite && refs[at] == nax::HdSprites::encode(5, 10, 0);
     }},
    {"upscaled_sprite_gives_way_to_original_pixels_while_its_colours_fade", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);
         hd.placed(0, 100, 200);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         const bool before = refs[10 * 320 + 5] != nax::HdSprites::kNone;
         m.dac[0x41][0] ^= 0x10;  // the game changes one of the sprite's colours
         hd.frame(m, refs.data());
         return before && refs[10 * 320 + 5] == nax::HdSprites::kNone;
     }},
    {"upscaled_sprite_stays_while_a_colour_of_its_bank_that_it_does_not_use_blinks", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);  // uses only the colours 41h and 42h of the bank 40h
         hd.placed(0, 100, 200);
         std::vector<uint32_t> refs(64000);
         m.dac[0x45][0] ^= 0x10;  // a lamp of the same bank blinks
         hd.frame(m, refs.data());
         return refs[10 * 320 + 5] != nax::HdSprites::kNone;
     }},
    {"sprite_drawn_again_after_a_colour_it_does_not_use_changed_keeps_its_name_and_picture", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);
         hd.placed(0, 100, 200);
         std::vector<uint32_t> refs(64000);
         m.dac[0x45][0] ^= 0x10;  // a lamp of the same bank has changed and stays so
         hd.frame(m, refs.data());  // notices the change
         draw_test_sprite(m, hd);   // the game draws the sprite again at once
         hd.frame(m, refs.data());
         return hd.sprite_count() == 1 && refs[10 * 320 + 5] == nax::HdSprites::encode(100, 201, 1 + 1);
     }},
    {"picture_written_under_the_earlier_name_after_the_rename_is_still_taken_over", [] {
         Fixture f;
         const std::wstring name = legacy_name(f.m);
         write_file(f.root / L"hd" / L"original" / name, "old original");
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         draw_test_sprite(f.m, hd);  // renames the original; no picture yet
         const bool missing = hd.missing_count() == 1;
         write_file(f.root / L"hd" / L"upscaled" / name, "late picture");  // a redraw that started earlier finishes
         hd.take_pending();
         std::vector<fs::path> upscaled;
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"upscaled")) upscaled.push_back(e.path());
         return missing && hd.missing_count() == 0 && upscaled.size() == 1 && upscaled[0].filename() != name &&
                read_file(upscaled[0]) == "late picture";
     }},
    {"layout_pairs_follow_a_sprite_to_its_new_name", [] {
         Fixture f;
         const std::wstring name = legacy_name(f.m);
         write_file(f.root / L"hd" / L"original" / name, "old original");
         std::string old_key;
         old_key.reserve(16);
         for (size_t i = 0; i < 16 && i < name.size(); i++) old_key.push_back(char(name[i]));
         write_file(f.root / L"hd" / L"layout.txt", "0000000000000001 " + old_key + " 16 8\n0000000000000001 0000000000000002 -16 8\n");
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         draw_test_sprite(f.m, hd);
         std::string new_key;
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"original")) new_key = e.path().stem().string();
         return new_key != old_key &&
                read_file(f.root / L"hd" / L"layout.txt") == "0000000000000001 " + new_key + " 16 8\n0000000000000001 0000000000000002 -16 8\n";
     }},
    {"pictures_saved_under_the_earlier_name_are_taken_over", [] {
         Fixture f;
         const std::wstring name = legacy_name(f.m);
         write_file(f.root / L"hd" / L"original" / name, "old original");
         write_file(f.root / L"hd" / L"upscaled" / name, "old picture");
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         draw_test_sprite(f.m, hd);
         std::vector<fs::path> originals, upscaled;
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"original")) originals.push_back(e.path());
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"upscaled")) upscaled.push_back(e.path());
         return originals.size() == 1 && upscaled.size() == 1 && originals[0].filename() != name &&
                read_file(upscaled[0]) == "old picture" && upscaled[0].filename() == originals[0].filename() && hd.take_pending().empty() &&
                hd.missing_count() == 0;
     }},
    {"emptied_atlas_asks_again_only_for_pictures_that_show_on_screen", [] {
         Fixture f;
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         draw_test_sprite(f.m, hd);
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"original"))
             fs::copy_file(e.path(), f.root / L"hd" / L"upscaled" / e.path().filename());
         hd.placed(0, 1, 1);
         const bool none_after_placing = hd.take_pending().empty();
         hd.atlas_cleared();
         std::vector<uint32_t> refs(64000);
         const uint32_t at = 0xA0000 + 10 * 320 + 5;
         f.m.mem[at] = f.m.mem[at + 1] = 0x07;  // the sprite is not on screen
         hd.frame(f.m, refs.data());
         const bool none_while_hidden = hd.take_pending().empty();
         f.m.mem[at] = 0x41;  // the sprite shows again
         hd.frame(f.m, refs.data());
         return none_after_placing && none_while_hidden && hd.take_pending().size() == 1;
     }},
    {"picture_without_room_in_full_atlas_is_asked_for_again_when_it_shows", [] {
         Fixture f;
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         draw_test_sprite(f.m, hd);
         for (const auto& e : fs::directory_iterator(f.root / L"hd" / L"original"))
             fs::copy_file(e.path(), f.root / L"hd" / L"upscaled" / e.path().filename());
         std::vector<uint32_t> refs(64000);
         hd.take_pending();  // finds the upscaled picture on disk
         hd.frame(f.m, refs.data());
         const bool asked = hd.take_pending().size() == 1;
         hd.no_room(0);  // the atlas was full
         const bool waits = hd.take_pending().empty();
         hd.frame(f.m, refs.data());
         return asked && waits && hd.take_pending().size() == 1;
     }},
    {"waiting_picture_covered_by_a_newer_one_is_dropped_and_nine_others_all_wait", [] {
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_screen_source(kBuffer);
         for (int c = 0; c < 16; c++) m.dac[0x20 + c][0] = uint8_t(c + 1);
         auto picture = [&](uint32_t line, uint32_t lines, uint8_t colour) {
             for (uint32_t i = 0; i < 320 * lines; i++) m.mem[kBuffer + line * 320 + i] = colour;
             hd.image(m, kBuffer + line * 320, 320, int(lines), 320, kBuffer + line * 320, 320);
         };
         picture(0, 10, 0x20);   // covered completely by the next picture
         picture(0, 20, 0x21);
         for (uint32_t k = 0; k < 9; k++) picture(20 + k * 10, 10, uint8_t(0x22 + k));
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(600);
         hd.frame(m, refs.data());
         return hd.sprite_count() == 10;
     }},
    {"full_atlas_is_emptied_once_and_not_again_while_the_same_crowded_screen_stays", [] {
         nax::AtlasClearing clearing(1000);
         const bool first = clearing.overflow(0, 400);         // the screen has moved on: emptied
         const bool soon = clearing.overflow(5000, 400);       // full again at once, so emptying did not help
         const bool later = clearing.overflow(20000, 400);     // the same screen ten seconds later
         const bool much_later = clearing.overflow(50000, 400);
         return first && !soon && !later && !much_later;
     }},
    {"full_atlas_is_not_emptied_when_the_screen_alone_needs_more_than_half_of_it", [] {
         nax::AtlasClearing clearing(1000);
         return !clearing.overflow(0, 501) && clearing.overflow(1, 500);
     }},
    {"full_atlas_is_emptied_again_a_minute_after_emptying_did_not_help_although_it_kept_running_full", [] {
         nax::AtlasClearing clearing(1000);
         const bool first = clearing.overflow(0, 400);
         bool held = true;
         for (int64_t t = 5000; t <= 64000; t += 1000) held = held && !clearing.overflow(t, 400);  // every frame
         return first && held && clearing.overflow(66000, 400);
     }},
    {"screen_that_changes_soon_after_emptying_is_not_held_back_by_it", [] {
         nax::AtlasClearing clearing(1000);
         const bool first = clearing.overflow(0, 400);
         const bool soon = clearing.overflow(5000, 100);  // another screen that needs much less
         return first && !soon && clearing.overflow(11000, 100);
     }},
    {"second_crowded_screen_within_the_same_minute_is_held_back_too", [] {
         nax::AtlasClearing clearing(1000);
         const bool a = clearing.overflow(0, 400) && !clearing.overflow(5000, 400);       // screen A did not help
         const bool b = clearing.overflow(20000, 200) && !clearing.overflow(25000, 200);  // screen B did not help
         return a && b && !clearing.overflow(36000, 200) && !clearing.overflow(36000, 400);
     }},
    {"clearly_larger_screen_is_not_held_back_by_a_smaller_one", [] {
         nax::AtlasClearing clearing(1000);
         const bool held = clearing.overflow(0, 200) && !clearing.overflow(5000, 200);
         return held && !clearing.overflow(20000, 200) && clearing.overflow(20000, 450);
     }},
    {"room_needed_on_screen_counts_each_shown_sprite_once_with_its_gap", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);  // a 32x2 sprite, two of whose pixels show
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         const bool none_without_picture = hd.on_screen_cells() == 0;
         hd.placed(0, 1, 1);
         hd.frame(m, refs.data());
         return none_without_picture && hd.on_screen_cells() == 34 * 4;
     }},
    {"moved_pointer_reruns_the_ai_filter_only_around_it_with_each_earlier_layer_one_pixel_wider", [] {
         using U = nax::AiUpscaler;
         const U::Rect pointer{100, 50, 112, 59};  // a 12x9 pointer
         const U::Rect out = U::region(pointer, 8, 8), last = U::region(pointer, 8, 7), first = U::region(pointer, 8, 0);
         const bool result = out.x0 == 92 && out.y0 == 42 && out.x1 == 120 && out.y1 == 67;
         const bool layers = last.x0 == 92 && first.x0 == 85 && first.x1 == 127 && U::region(pointer, 8, 1).x0 == 86;
         return result && layers;
     }},
    {"ai_filter_region_near_the_edge_stays_inside_the_picture", [] {
         using U = nax::AiUpscaler;
         const U::Rect r = U::region({0, 195, 5, 200}, 8, 0);
         return r.x0 == 0 && r.y0 == 180 && r.x1 == 20 && r.y1 == 200;
     }},
    {"tiles_drawn_side_by_side_in_three_frames_are_recorded_once_with_where_the_second_lies", [] {
         nax::NeighbourLog log;
         const uint32_t a = 0x10000 + 100 * 320 + 50;  // the bottom row of tile 5
         const uint32_t b = a + 8 * 320 + 16;          // tile 3, half a tile to the right and 8 lines lower
         std::vector<std::vector<nax::NeighbourLog::Pair>> found;
         for (uint32_t frame = 1; frame <= 5; frame++) {
             auto pairs = log.drawn(5, 32, 15, a, frame);
             for (const auto& p : log.drawn(3, 32, 15, b, frame)) pairs.push_back(p);
             found.push_back(pairs);
         }
         const auto& third = found[2];
         return found[0].empty() && found[1].empty() && third.size() == 1 && third[0].a == 3 && third[0].b == 5 &&
                third[0].dx == -16 && third[0].dy == -8 && found[3].empty() && found[4].empty();
     }},
    {"tiles_that_met_only_while_the_screen_scrolled_are_not_recorded", [] {
         nax::NeighbourLog log;
         const uint32_t a = 0x10000 + 100 * 320 + 50;
         bool none = log.drawn(5, 32, 15, a, 1).empty() && log.drawn(3, 32, 15, a + 8 * 320 + 16, 1).empty();
         for (uint32_t frame = 2; frame <= 6; frame++) {  // tile 3 has moved far away since
             none = none && log.drawn(5, 32, 15, a, frame).empty();
             none = none && log.drawn(3, 32, 15, a + 60 * 320 + 100, frame).empty();
         }
         return none;
     }},
    {"tiles_that_only_meet_at_a_corner_are_not_neighbours", [] {
         nax::NeighbourLog log;
         const uint32_t a = 0x10000 + 100 * 320 + 50;
         bool none = true;
         for (uint32_t frame = 1; frame <= 5; frame++) {
             none = none && log.drawn(5, 32, 15, a, frame).empty();
             none = none && log.drawn(3, 32, 15, a + 15 * 320, frame).empty();  // directly below
             none = none && log.drawn(4, 32, 15, a + 32, frame).empty();        // directly to the right
         }
         return none;
     }},
    {"neighbours_are_written_to_the_layout_file_once_across_runs", [] {
         Fixture f;
         auto run = [&f] {
             nax::HdSprites hd;
             hd.set_folder(f.root / L"hd");
             std::vector<uint32_t> refs(64000);
             for (int frame = 0; frame < 4; frame++) {
                 f.m.mem[kSpriteData] = 0x12;
                 hd.draw(f.m, kSpriteData, 32, 2, kDest, 0x40);
                 f.m.mem[kSpriteData] = 0x34;
                 hd.draw(f.m, kSpriteData, 32, 2, kDest + 8 * 320 + 16, 0x40);  // another tile, the next one on the grid
                 hd.frame(f.m, refs.data());
             }
         };
         run();
         const std::string first = read_file(f.root / L"hd" / L"layout.txt");
         run();
         const std::string second = read_file(f.root / L"hd" / L"layout.txt");
         return std::count(first.begin(), first.end(), '\n') == 1 && second == first &&
                (first.ends_with(" 16 8\n") || first.ends_with(" -16 -8\n"));
     }},
    {"unit_standing_on_a_tile_and_tiles_off_the_grid_are_not_neighbours", [] {
         nax::NeighbourLog log;
         const uint32_t a = 0x10000 + 100 * 320 + 50;
         bool none = true;
         for (uint32_t frame = 1; frame <= 5; frame++) {
             none = none && log.drawn(5, 32, 15, a, frame).empty();
             none = none && log.drawn(3, 16, 10, a - 2 * 320 + 8, frame).empty();     // a small unit on the tile
             none = none && log.drawn(4, 32, 15, a + 4 * 320 + 8, frame).empty();     // a tile a quarter step away
         }
         return none;
     }},
    {"layout_line_cut_short_by_a_crash_does_not_hide_the_lines_after_it", [] {
         Fixture f;
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         // Pick the keys the two test tiles get, by recording them once in a separate folder.
         Fixture g;
         {
             nax::HdSprites probe;
             probe.set_folder(g.root / L"hd");
             std::vector<uint32_t> refs(64000);
             for (int frame = 0; frame < 4; frame++) {
                 g.m.mem[kSpriteData] = 0x12;
                 probe.draw(g.m, kSpriteData, 32, 2, kDest, 0x40);
                 g.m.mem[kSpriteData] = 0x34;
                 probe.draw(g.m, kSpriteData, 32, 2, kDest + 8 * 320 + 16, 0x40);
                 probe.frame(g.m, refs.data());
             }
         }
         const std::string pair = read_file(g.root / L"hd" / L"layout.txt");
         write_file(f.root / L"hd" / L"layout.txt", "00000000000000aa 00000000000000bb 16\n" + pair);
         std::vector<uint32_t> refs(64000);
         for (int frame = 0; frame < 4; frame++) {
             f.m.mem[kSpriteData] = 0x12;
             hd.draw(f.m, kSpriteData, 32, 2, kDest, 0x40);
             f.m.mem[kSpriteData] = 0x34;
             hd.draw(f.m, kSpriteData, 32, 2, kDest + 8 * 320 + 16, 0x40);
             hd.frame(f.m, refs.data());
         }
         const std::string after = read_file(f.root / L"hd" / L"layout.txt");
         return !pair.empty() && after == "00000000000000aa 00000000000000bb 16\n" + pair;  // not written a second time
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
    {"top_down_sprite_marks_rows_downward_and_skips_neighbour_log", [] {
         Fixture f;
         nax::HdSprites hd;
         hd.set_folder(f.root / L"hd");
         hd.set_screen_source(kBuffer);
         f.m.mem[kSpriteData] = 0x12;      // row 0: colours 1, 2
         f.m.mem[kSpriteData + 8] = 0x34;  // row 1 (stride 8 bytes for 16px width): colours 3, 4
         const uint32_t dest = kBuffer + 20 * 320 + 10;
         std::vector<uint32_t> refs(64000);
         for (int frame = 0; frame < 4; frame++) {
             hd.draw(f.m, kSpriteData, 16, 2, dest, 0x10, false, false);
             hd.draw(f.m, kSpriteData, 16, 2, dest + 4 * 320 + 8, 0x10, false, false);
             hd.frame(f.m, refs.data());
         }
         hd.placed(0, 50, 60);
         f.m.mem[0xA0000 + 20 * 320 + 10] = 0x11;
         f.m.mem[0xA0000 + 21 * 320 + 10] = 0x13;
         hd.frame(f.m, refs.data());
         const bool row0 = (refs[20 * 320 + 10] & 0xFFFFF) == (nax::HdSprites::encode(50, 60, 0) & 0xFFFFF);
         const bool row1 = (refs[21 * 320 + 10] & 0xFFFFF) == (nax::HdSprites::encode(50, 61, 0) & 0xFFFFF);
         return row0 && row1 && !fs::exists(f.root / L"hd" / L"layout.txt");
     }},
    {"direct_vram_picture_and_button_show_until_buffer_is_presented", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x07][0] = 40;
         m.dac[0x11][1] = 50;
         m.dac[0x22][2] = 60;
         hd.set_screen_source(kBuffer);
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[kBuffer + i] = 0x07;
         hd.image(m, kBuffer, 320, 10, 320, kBuffer, 320);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(600);
         hd.frame(m, refs.data());
         hd.placed(0, 10, 10);  // buffer picture at atlas (10, 10)
         // Now unpack a menu screen directly to 0xA0000 and draw a button directly to 0xA0000.
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[0xA0000 + i] = 0x22;
         hd.set_screen_source(0xA0000);
         hd.image(m, 0xA0000, 320, 10, 320, 0xA0000, 320);
         Sleep(600);
         hd.frame(m, refs.data());
         hd.placed(1, 200, 10);  // VRAM picture at atlas (200, 10)
         m.mem[kSpriteData] = 0x10;
         hd.draw(m, kSpriteData, 16, 2, 0xA0000 + 2 * 320 + 4, 0x10, false, false);
         hd.placed(2, 400, 20);  // VRAM button at atlas (400, 20)
         m.mem[0xA0000 + 2 * 320 + 4] = 0x11;
         hd.frame(m, refs.data());
         const bool vram_pic = refs[0] == nax::HdSprites::encode(200, 10, 0);
         const bool vram_btn = (refs[2 * 320 + 4] & 0xFFFFF) == (nax::HdSprites::encode(400, 20, 0) & 0xFFFFF);
         // Presenting the offscreen buffer clears VRAM marks and restores the buffer picture.
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[0xA0000 + i] = 0x07;
         hd.present_buffer(kBuffer);
         hd.frame(m, refs.data());
         return vram_pic && vram_btn && refs[0] == nax::HdSprites::encode(10, 10, 0) &&
                refs[2 * 320 + 4] == nax::HdSprites::encode(14, 12, 0);
     }},
    {"presenting_buffer_drops_pending_vram_picture_before_it_settles", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x22][0] = 50;
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[0xA0000 + i] = 0x22;
         hd.set_screen_source(0xA0000);
         hd.image(m, 0xA0000, 320, 10, 320, 0xA0000, 320);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         // Close the menu before the 500 ms settle timer finishes.
         hd.present_buffer(kBuffer);
         Sleep(600);
         hd.frame(m, refs.data());
         return hd.sprite_count() == 0;
     }},
    {"panel_fill_hides_covered_sprites_and_backdrop_until_backdrop_is_restored", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x07][0] = 40;
         hd.set_screen_source(kBuffer);
         for (uint32_t i = 0; i < 320 * 20; i++) {
             m.mem[kBuffer + i] = 0x07;
             m.mem[0xA0000 + i] = 0x07;
         }
         hd.image(m, kBuffer, 320, 20, 320, kBuffer, 320);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(600);
         hd.frame(m, refs.data());
         hd.placed(0, 10, 10);
         m.mem[kSpriteData] = 0x10;
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.placed(1, 100, 100);
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         hd.frame(m, refs.data());
         const bool before = refs[10 * 320 + 5] != nax::HdSprites::kNone && refs[10 * 320 + 10] != nax::HdSprites::kNone;
         // A popup panel fills over x=0..15, y=8..15; even with identical colour indices, nothing bleeds through.
         hd.clear_rect(kBuffer + 8 * 320, 16, 8);
         hd.frame(m, refs.data());
         const bool hidden = refs[10 * 320 + 5] == nax::HdSprites::kNone && refs[10 * 320 + 10] == nax::HdSprites::kNone &&
                             refs[10 * 320 + 20] != nax::HdSprites::kNone;
         // Restoring the saved map/backdrop buffer unmasks both sprites and backdrop wherever their colours match.
         hd.restore_rect(kBuffer, 256, 20);
         hd.frame(m, refs.data());
         const bool sprite_restored = (refs[10 * 320 + 5] & 0xFFFFF) == (nax::HdSprites::encode(100, 101, 0) & 0xFFFFF);
         m.mem[0xA0000 + 10 * 320 + 5] = 0x07;
         hd.frame(m, refs.data());
         const bool backdrop_restored = refs[10 * 320 + 5] == nax::HdSprites::encode(15, 20, 0) &&
                                        refs[10 * 320 + 10] == nax::HdSprites::encode(20, 20, 0);
         return before && hidden && sprite_restored && backdrop_restored;
     }},
    {"vram_panel_fill_blocks_offscreen_buffer_bleed_until_buffer_is_presented", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x07][0] = 40;
         hd.set_screen_source(kBuffer);
         for (uint32_t i = 0; i < 320 * 20; i++) {
             m.mem[kBuffer + i] = 0x07;
             m.mem[0xA0000 + i] = 0x07;
         }
         hd.image(m, kBuffer, 320, 20, 320, kBuffer, 320);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(600);
         hd.frame(m, refs.data());
         hd.placed(0, 10, 10);
         // A popup drawn directly into VRAM (0xA0000) covers x=0..15, y=8..15 while screen_source is still kBuffer.
         hd.clear_rect(0xA0000 + 8 * 320, 16, 8);
         hd.frame(m, refs.data());
         const bool blocked = refs[10 * 320 + 5] == nax::HdSprites::kNone && refs[10 * 320 + 20] != nax::HdSprites::kNone;
         // Presenting kBuffer back to VRAM clears the VRAM popup mask and restores the HD backdrop.
         hd.present_buffer(kBuffer);
         hd.frame(m, refs.data());
         return blocked && refs[10 * 320 + 5] == nax::HdSprites::encode(15, 20, 0);
     }},
    {"frame_uses_supplied_vram_snapshot_consistently", [] {
         nax::Machine m;
         nax::HdSprites hd;
         draw_test_sprite(m, hd);
         hd.placed(0, 100, 200);
         std::vector<uint8_t> snapshot(64000, 0);
         std::memcpy(snapshot.data(), m.vram(), 64000);
         // Mutate live VRAM after the snapshot (simulating the game thread writing the next frame).
         m.mem[0xA0000 + 10 * 320 + 5] = 0x99;
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data(), snapshot.data());
         return refs[10 * 320 + 5] == nax::HdSprites::encode(100, 201, 1);
     }},
    {"text_clear_syncs_hd_backdrop_so_stipple_bands_do_not_bleed_through", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x07][0] = 20;
         m.dac[0x08][0] = 30;
         m.dac[0x0E][0] = 60;
         hd.set_screen_source(0xA0000);
         for (uint32_t i = 0; i < 320 * 20; i++) m.mem[0xA0000 + i] = 0x07;
         hd.image(m, 0xA0000, 320, 20, 320, 0xA0000, 320);
         // A background-clear stipple write before settle updates the pending picture's expected pixels.
         m.mem[0xA0000 + 4 * 320 + 8] = 0x08;
         hd.sync_image_rect(m, 0xA0000 + 4 * 320 + 8, 8, 8);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(600);
         hd.frame(m, refs.data());
         hd.placed(0, 10, 10);
         hd.frame(m, refs.data());
         const bool synced_before_settle = refs[4 * 320 + 8] == nax::HdSprites::encode(18, 14, 0);
         // Text drawn over the cleared background still renders as crisp non-HD pixels.
         m.mem[0xA0000 + 4 * 320 + 9] = 0x0E;
         // A second clear after settle keeps the HD backdrop intact too.
         m.mem[0xA0000 + 12 * 320 + 8] = 0x08;
         hd.sync_image_rect(m, 0xA0000 + 12 * 320 + 8, 8, 6);
         hd.frame(m, refs.data());
         const bool text_on_top = refs[4 * 320 + 9] == nax::HdSprites::kNone;
         const bool synced_after_settle = refs[12 * 320 + 8] == nax::HdSprites::encode(18, 22, 0);
         return synced_before_settle && text_on_top && synced_after_settle;
     }},
    {"vram_text_clear_keeps_offscreen_buffer_backdrop_intact_after_present_and_preserves_pre_settle_panel_mask", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x07][0] = 20;
         m.dac[0x08][0] = 30;
         hd.set_screen_source(kBuffer);
         for (uint32_t i = 0; i < 320 * 20; i++) {
             m.mem[kBuffer + i] = 0x07;
             m.mem[0xA0000 + i] = 0x07;
         }
         hd.image(m, kBuffer, 320, 20, 320, kBuffer, 320);
         // A popup panel drawn into the offscreen buffer before settle must stay masked after take_over().
         hd.clear_rect(kBuffer + 2 * 320 + 2, 8, 8);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(600);
         hd.frame(m, refs.data());
         hd.placed(0, 10, 10);
         hd.frame(m, refs.data());
         const bool pre_settle_masked = refs[2 * 320 + 2] == nax::HdSprites::kNone &&
                                        refs[12 * 320 + 8] == nax::HdSprites::encode(18, 22, 0);
         // A direct-VRAM text clear writes stipple 0x08 only to VRAM, never corrupting kBuffer's expected 0x07.
         m.mem[0xA0000 + 12 * 320 + 8] = 0x08;
         hd.sync_image_rect(m, 0xA0000 + 12 * 320 + 8, 8, 8);
         hd.frame(m, refs.data());
         const bool vram_overlay_ok = refs[12 * 320 + 8] == nax::HdSprites::encode(18, 22, 0);
         // Presenting kBuffer back to VRAM restores original pixel 0x07 and shows the offscreen HD backdrop again.
         m.mem[0xA0000 + 12 * 320 + 8] = 0x07;
         hd.present_buffer(kBuffer);
         hd.frame(m, refs.data());
         const bool buffer_preserved = refs[12 * 320 + 8] == nax::HdSprites::encode(18, 22, 0);
         // Clearing a VRAM popup interior with sync_image_rect() must erase any prior direct-VRAM sprite marks.
         hd.clear_rect(0xA0000 + 10 * 320 + 4, 16, 8);
         m.dac[0x41][0] = 30;
         m.mem[kSpriteData] = 0x10;
         hd.draw(m, kSpriteData, 32, 2, 0xA0000 + 10 * 320 + 5, 0x40);
         hd.placed(1, 200, 200);
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         hd.sync_image_rect(m, 0xA0000 + 10 * 320 + 4, 8, 8);
         hd.frame(m, refs.data());
         const bool popup_sprite_erased = refs[10 * 320 + 5] == nax::HdSprites::kNone;
         // sync_image_rect when screen_source is 0 must not index low memory.
         hd.present_buffer(0);
         hd.sync_image_rect(m, 0xA0000 + 12 * 320 + 8, 8, 8);
         return pre_settle_masked && vram_overlay_ok && buffer_preserved && popup_sprite_erased;
     }},
    {"ambient_nighttime_dimming_keeps_hd_assets_and_matches_daytime_sprites_on_disk", [] {
         Fixture f;
         const fs::path folder = f.root / L"hd";
         // First session (daytime): save a sprite and place an upscaled PNG for it.
         uint64_t daytime_key = 0;
         {
             nax::Machine m;
             nax::HdSprites hd;
             hd.set_folder(folder);
             m.dac[0x41][0] = 34; m.dac[0x41][1] = 34; m.dac[0x41][2] = 61;
             m.dac[0x42][0] = 41; m.dac[0x42][1] = 42; m.dac[0x42][2] = 61;
             m.mem[kSpriteData] = 0x12;
             hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
             for (const auto& entry : fs::directory_iterator(folder / L"original")) {
                 unsigned long long k = 0;
                 if (swscanf(entry.path().stem().wstring().c_str(), L"%llx", &k) == 1) daytime_key = k;
                 fs::copy_file(entry.path(), folder / L"upscaled" / entry.path().filename());
             }
         }
         // Second session (nighttime save loaded directly): dimmed by (0.61, 0.61, 0.72).
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_folder(folder);
         hd.set_screen_source(kBuffer);
         m.dac[0x41][0] = 21; m.dac[0x41][1] = 21; m.dac[0x41][2] = 44;
         m.dac[0x42][0] = 25; m.dac[0x42][1] = 26; m.dac[0x42][2] = 44;
         m.mem[kSpriteData] = 0x12;
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         m.mem[0xA0000 + 10 * 320 + 6] = 0x42;
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         const auto pending = hd.take_pending();
         const bool matched_disk = daytime_key != 0 && pending.size() == 1 && hd.sprite_count() == 1;
         if (!pending.empty()) hd.placed(pending[0].slot, 100, 200);
         hd.frame(m, refs.data());
         float tint[3] = {1, 1, 1};
         hd.ambient_tint(tint);
         const bool rendered_dim = refs[10 * 320 + 5] == nax::HdSprites::encode(100, 201, 1, true) &&
                                   tint[0] > 0.55f && tint[0] < 0.68f && tint[2] > 0.66f && tint[2] < 0.78f;
         // Returning to daylight restores full (1, 1, 1) tint and keeps the HD sprite (undimmed bit).
         m.dac[0x41][0] = 34; m.dac[0x41][1] = 34; m.dac[0x41][2] = 61;
         m.dac[0x42][0] = 41; m.dac[0x42][1] = 42; m.dac[0x42][2] = 61;
         hd.frame(m, refs.data());
         hd.ambient_tint(tint);
         const bool daylight_back = refs[10 * 320 + 5] == nax::HdSprites::encode(100, 201, 1, false) &&
                                    tint[0] > 0.99f && tint[1] > 0.99f && tint[2] > 0.99f;
         // A newly discovered daytime sprite in the same session must also match when night falls later.
         m.mem[kSpriteData + 64] = 0x34;
         m.dac[0x43][0] = 40; m.dac[0x43][1] = 30; m.dac[0x43][2] = 20;
         m.dac[0x44][0] = 50; m.dac[0x44][1] = 40; m.dac[0x44][2] = 30;
         hd.draw(m, kSpriteData + 64, 32, 2, kDest + 64, 0x40);
         const size_t before_night_count = hd.sprite_count();
         m.dac[0x43][0] = 24; m.dac[0x43][1] = 18; m.dac[0x43][2] = 14;
         m.dac[0x44][0] = 30; m.dac[0x44][1] = 24; m.dac[0x44][2] = 21;
         hd.draw(m, kSpriteData + 64, 32, 2, kDest + 64, 0x40);
         const bool same_session_reused = hd.sprite_count() == before_night_count;
         // Cross-planet palette (Mars reddish-brown instead of Ice blue) must NOT match the Ice daytime sprite.
         m.dac[0x41][0] = 25; m.dac[0x41][1] = 8; m.dac[0x41][2] = 0;
         m.dac[0x42][0] = 35; m.dac[0x42][1] = 12; m.dac[0x42][2] = 0;
         hd.frame(m, refs.data());
         const bool mars_rejected = refs[10 * 320 + 5] == nax::HdSprites::kNone;
         return matched_disk && rendered_dim && daylight_back && same_session_reused && mars_rejected;
     }},
    {"single_blinking_colour_and_two_channel_hue_edit_do_not_trigger_ambient_dimming", [] {
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_screen_source(kBuffer);
         // 5-colour sprite: dimming only 1 of 5 colours (a blinking indicator) must NOT darken the frame.
         for (uint8_t c = 1; c <= 5; c++) {
             m.dac[0x40 + c][0] = uint8_t(10 * c);
             m.dac[0x40 + c][1] = uint8_t(10 * c);
             m.dac[0x40 + c][2] = uint8_t(10 * c);
         }
         m.mem[kSpriteData] = 0x12;
         m.mem[kSpriteData + 1] = 0x34;
         m.mem[kSpriteData + 2] = 0x50;
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.placed(0, 10, 10);
         m.dac[0x45][0] = 25; m.dac[0x45][1] = 25; m.dac[0x45][2] = 25;  // only 1 of 5 colours dimmed
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         float tint[3] = {1, 1, 1};
         hd.ambient_tint(tint);
         const bool single_blink_rejected = refs[10 * 320 + 5] == nax::HdSprites::kNone && tint[0] > 0.99f;
         // 2-channel sprite (B=0): dimming only Red while Green stays ~1.0 must NOT count inactive B as dimmed.
         m.dac[0x51][0] = 40; m.dac[0x51][1] = 40; m.dac[0x51][2] = 0;
         m.dac[0x52][0] = 20; m.dac[0x52][1] = 20; m.dac[0x52][2] = 0;
         m.mem[kSpriteData + 32] = 0x12;
         m.mem[0xA0000 + 12 * 320 + 5] = 0x51;
         hd.draw(m, kSpriteData + 32, 32, 2, kBuffer + 12 * 320 + 5, 0x50);
         hd.placed(1, 50, 10);
         m.dac[0x51][0] = 28; m.dac[0x51][1] = 39; m.dac[0x51][2] = 0;
         m.dac[0x52][0] = 14; m.dac[0x52][1] = 19; m.dac[0x52][2] = 0;
         hd.frame(m, refs.data());
         const bool two_channel_edit_rejected = refs[12 * 320 + 5] == nax::HdSprites::kNone;
         return single_blink_rejected && two_channel_edit_rejected;
     }},
    {"near_exact_fade_rounding_and_landing_pad_emissive_match_immediately_without_settle_delay", [] {
         Fixture f;
         const fs::path folder = f.root / L"hd";
         // Daytime session saves a 5-colour landing-pad-like sprite and marks it upscaled.
         {
             nax::Machine m;
             nax::HdSprites hd;
             hd.set_folder(folder);
             m.dac[0x41][0] = 14; m.dac[0x41][1] = 14; m.dac[0x41][2] = 14;
             m.dac[0x42][0] = 42; m.dac[0x42][1] = 0;  m.dac[0x42][2] = 0;
             m.dac[0x43][0] = 21; m.dac[0x43][1] = 21; m.dac[0x43][2] = 21;
             m.dac[0x44][0] = 58; m.dac[0x44][1] = 58; m.dac[0x44][2] = 0;  // emissive yellow
             m.dac[0x45][0] = 42; m.dac[0x45][1] = 23; m.dac[0x45][2] = 0;
             m.mem[kSpriteData] = 0x12;
             m.mem[kSpriteData + 1] = 0x34;
             m.mem[kSpriteData + 2] = 0x50;
             hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
             for (const auto& entry : fs::directory_iterator(folder / L"original"))
                 fs::copy_file(entry.path(), folder / L"upscaled" / entry.path().filename());
         }
         // New session: palette just changed (unsettled!) and landed on +-1 fade rounding of daytime.
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_folder(folder);
         hd.set_screen_source(kBuffer);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());  // records initial DAC
         m.dac[0x41][0] = 15; m.dac[0x41][1] = 15; m.dac[0x41][2] = 15;
         m.dac[0x42][0] = 43; m.dac[0x42][1] = 0;  m.dac[0x42][2] = 0;
         m.dac[0x43][0] = 22; m.dac[0x43][1] = 22; m.dac[0x43][2] = 22;
         m.dac[0x44][0] = 59; m.dac[0x44][1] = 59; m.dac[0x44][2] = 0;
         m.dac[0x45][0] = 43; m.dac[0x45][1] = 23; m.dac[0x45][2] = 0;
         hd.frame(m, refs.data());  // marks palette unsettled right now
         m.mem[kSpriteData] = 0x12;
         m.mem[kSpriteData + 1] = 0x34;
         m.mem[kSpriteData + 2] = 0x50;
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         // Even before settle finishes, the already-upscaled sprite matches via +-2 near-exact immediately!
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.frame(m, refs.data());
         const auto pending = hd.take_pending();
         const bool instant_match = pending.size() == 1 && hd.sprite_count() == 1;
         if (!pending.empty()) hd.placed(pending[0].slot, 10, 20);
         hd.frame(m, refs.data());
         const bool near_exact_undimmed = refs[10 * 320 + 5] == nax::HdSprites::encode(10, 21, 1, false);
         // Now dim to nighttime with yellow emissive drifting +2 (58 -> 60): must still stay HD and dimmed!
         m.dac[0x41][0] = 8;  m.dac[0x41][1] = 8;  m.dac[0x41][2] = 10;
         m.dac[0x42][0] = 27; m.dac[0x42][1] = 0;  m.dac[0x42][2] = 0;
         m.dac[0x43][0] = 13; m.dac[0x43][1] = 13; m.dac[0x43][2] = 16;
         m.dac[0x44][0] = 60; m.dac[0x44][1] = 60; m.dac[0x44][2] = 0;
         m.dac[0x45][0] = 27; m.dac[0x45][1] = 15; m.dac[0x45][2] = 0;
         hd.frame(m, refs.data());
         const bool pad_night_ok = refs[10 * 320 + 5] == nax::HdSprites::encode(10, 21, 1, true);
         // Even if a stale un-upscaled nighttime PNG exists on disk, a new session must still pick the upscaled daytime PNG!
         {
             nax::Machine m2;
             nax::HdSprites hd2;
             // Write a stale un-upscaled nighttime PNG into folder/original/ (slightly shifted by +1 DAC).
             m2.dac[0x41][0] = 9;  m2.dac[0x41][1] = 9;  m2.dac[0x41][2] = 11;
             m2.dac[0x42][0] = 28; m2.dac[0x42][1] = 0;  m2.dac[0x42][2] = 0;
             m2.dac[0x43][0] = 14; m2.dac[0x43][1] = 14; m2.dac[0x43][2] = 17;
             m2.dac[0x44][0] = 60; m2.dac[0x44][1] = 60; m2.dac[0x44][2] = 0;
             m2.dac[0x45][0] = 28; m2.dac[0x45][1] = 16; m2.dac[0x45][2] = 0;
             m2.mem[kSpriteData] = 0x12;
             m2.mem[kSpriteData + 1] = 0x34;
             m2.mem[kSpriteData + 2] = 0x50;
             // Save stale PNG directly into original/ without disk_shapes_ matching.
             const fs::path stale_dir = f.root / L"stale_tmp";
             hd2.set_folder(stale_dir);
             hd2.draw(m2, kSpriteData, 32, 2, kDest, 0x40);
             for (const auto& entry : fs::directory_iterator(stale_dir / L"original"))
                 fs::copy_file(entry.path(), folder / L"original" / entry.path().filename(), fs::copy_options::overwrite_existing);
         }
         nax::Machine m3;
         nax::HdSprites hd3;
         hd3.set_folder(folder);
         hd3.set_screen_source(kBuffer);
         m3.dac[0x41][0] = 8;  m3.dac[0x41][1] = 8;  m3.dac[0x41][2] = 10;
         m3.dac[0x42][0] = 27; m3.dac[0x42][1] = 0;  m3.dac[0x42][2] = 0;
         m3.dac[0x43][0] = 13; m3.dac[0x43][1] = 13; m3.dac[0x43][2] = 16;
         m3.dac[0x44][0] = 60; m3.dac[0x44][1] = 60; m3.dac[0x44][2] = 0;
         m3.dac[0x45][0] = 27; m3.dac[0x45][1] = 15; m3.dac[0x45][2] = 0;
         m3.mem[kSpriteData] = 0x12;
         m3.mem[kSpriteData + 1] = 0x34;
         m3.mem[kSpriteData + 2] = 0x50;
         m3.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         hd3.draw(m3, kSpriteData, 32, 2, kDest, 0x40);
         hd3.frame(m3, refs.data());
         const bool prefers_upscaled_over_stale_night = hd3.take_pending().size() == 1;
         return instant_match && near_exact_undimmed && pad_night_ok && prefers_upscaled_over_stale_night;
     }},
    {"double_buffered_front_marks_and_begin_present_prevent_mid_draw_flicker_and_tearing", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x41][0] = 50; m.dac[0x41][1] = 30; m.dac[0x41][2] = 10;
         m.dac[0x42][0] = 10; m.dac[0x42][1] = 40; m.dac[0x42][2] = 60;
         hd.set_screen_source(kBuffer);
         // Frame 1: draw building (0x41) into offscreen buffer, copy to VRAM, and present_buffer().
         m.mem[kSpriteData] = 0x10;
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.placed(0, 100, 100);
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         hd.present_buffer(kBuffer);
         // Now simulate game thread starting Frame 2: clears map sprites and draws ground tile (0x42) over the
         // building's position in kBuffer BEFORE copying kBuffer to VRAM (0xA0000 still holds building 0x41).
         hd.restore_backdrop_rect(kBuffer, 256, 159);
         m.mem[kSpriteData + 64] = 0x20;
         hd.draw(m, kSpriteData + 64, 32, 2, kDest, 0x40);
         hd.placed(1, 200, 200);
         std::vector<uint8_t> snap(64000, 0);
         std::vector<uint32_t> refs(64000);
         hd.snapshot_and_frame(m, snap.data(), refs.data());
         // Front marks must still match VRAM building (atlas 100, 101), NOT flicker to kNone!
         const bool stable_mid_draw = (refs[10 * 320 + 5] & 0xFFFFF) == (nax::HdSprites::encode(100, 101, 0) & 0xFFFFF) &&
                                      snap[10 * 320 + 5] == 0x41;
         // During begin_present() .. present_buffer(), partial VRAM writes must not tear against front marks.
         hd.begin_present(m);
         m.mem[0xA0000 + 10 * 320 + 5] = 0x42;  // mid-copy half-written pixel
         hd.snapshot_and_frame(m, snap.data(), refs.data());
         const bool stable_during_copy = (refs[10 * 320 + 5] & 0xFFFFF) == (nax::HdSprites::encode(100, 101, 0) & 0xFFFFF) &&
                                         snap[10 * 320 + 5] == 0x41;
         // After present_buffer() finishes, front marks atomically switch to Frame 2's ground tile (200, 201).
         hd.present_buffer(kBuffer);
         hd.snapshot_and_frame(m, snap.data(), refs.data());
         const bool switched_cleanly = (refs[10 * 320 + 5] & 0xFFFFF) == (nax::HdSprites::encode(200, 201, 0) & 0xFFFFF) &&
                                       snap[10 * 320 + 5] == 0x42;
         return stable_mid_draw && stable_during_copy && switched_cleanly;
     }},
    {"upscaled_backdrop_takes_over_immediately_without_waiting_for_settle_or_fade_end", [] {
         Fixture f;
         const fs::path folder = f.root / L"hd";
         // Session 1: settle and save a 320x200 daytime screen, and place an upscaled PNG for it.
         {
             nax::Machine m;
             nax::HdSprites hd;
             hd.set_folder(folder);
             m.dac[0x07][0] = 40; m.dac[0x07][1] = 40; m.dac[0x07][2] = 48;
             m.dac[0x08][0] = 20; m.dac[0x08][1] = 24; m.dac[0x08][2] = 32;
             for (uint32_t i = 0; i < 64000; i++) m.mem[0xA0000 + i] = (i & 1) ? 0x07 : 0x08;
             hd.set_screen_source(0xA0000);
             hd.image(m, 0xA0000, 320, 200, 320, 0xA0000, 320);
             std::vector<uint32_t> refs(64000);
             hd.frame(m, refs.data());
             Sleep(200);
             hd.frame(m, refs.data());
             for (const auto& entry : fs::directory_iterator(folder / L"original"))
                 fs::copy_file(entry.path(), folder / L"upscaled" / entry.path().filename());
         }
         // Session 2: screen unpacks mid-fade at 50% brightness with unsettled DAC; must match immediately via disk_images_!
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_folder(folder);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());  // initial black DAC
         m.dac[0x07][0] = 20; m.dac[0x07][1] = 20; m.dac[0x07][2] = 24;
         m.dac[0x08][0] = 10; m.dac[0x08][1] = 12; m.dac[0x08][2] = 16;
         for (uint32_t i = 0; i < 64000; i++) m.mem[0xA0000 + i] = (i & 1) ? 0x07 : 0x08;
         hd.set_screen_source(0xA0000);
         hd.image(m, 0xA0000, 320, 200, 320, 0xA0000, 320);
         hd.frame(m, refs.data());  // no Sleep() — immediate takeover on first visible fade step!
         const auto pending = hd.take_pending();
         if (pending.size() != 1) return false;
         hd.placed(pending[0].slot, 10, 10);
         hd.frame(m, refs.data());
         const bool hd_during_fade = refs[0] == nax::HdSprites::encode(10, 10, 0, true);
         // Full daylight returns cleanly without dimming bit.
         m.dac[0x07][0] = 40; m.dac[0x07][1] = 40; m.dac[0x07][2] = 48;
         m.dac[0x08][0] = 20; m.dac[0x08][1] = 24; m.dac[0x08][2] = 32;
         hd.frame(m, refs.data());
         const bool hd_at_daylight = refs[0] == nax::HdSprites::encode(10, 10, 0, false);
         return hd_during_fade && hd_at_daylight;
     }},
    {"present_buffer_zero_clears_direct_vram_marks_and_pending_vram_pictures", [] {
         nax::Machine m;
         nax::HdSprites hd;
         m.dac[0x22][0] = 50;
         m.dac[0x11][1] = 50;
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[0xA0000 + i] = 0x22;
         hd.set_screen_source(0xA0000);
         hd.image(m, 0xA0000, 320, 10, 320, 0xA0000, 320);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(200);
         hd.frame(m, refs.data());
         hd.placed(0, 200, 10);
         // Queue a second unsettled VRAM picture too (simulating intro exiting mid-fade).
         m.dac[0x23][2] = 50;
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[0xA0000 + i] = 0x23;
         hd.image(m, 0xA0000, 320, 10, 320, 0xA0000, 320);
         hd.frame(m, refs.data());
         // Exiting the intro calls present_buffer(0), which must wipe all VRAM marks and pending VRAM pictures.
         hd.present_buffer(0);
         Sleep(200);
         for (uint32_t i = 0; i < 320 * 10; i++) m.mem[0xA0000 + i] = 0x22;
         hd.frame(m, refs.data());
         return refs[0] == nax::HdSprites::kNone && hd.sprite_count() == 1;
     }},
    {"missing_single_channel_sprite_rematches_when_ambient_tint_shifts_to_its_dim_level", [] {
         Fixture f;
         const fs::path folder = f.root / L"hd";
         // Daytime session: save a 1-channel red sprite (requires hint to match when dimmed) and mark it upscaled.
         {
             nax::Machine m;
             nax::HdSprites hd;
             hd.set_folder(folder);
             m.dac[0x41][0] = 60; m.dac[0x41][1] = 0; m.dac[0x41][2] = 0;
             m.dac[0x42][0] = 40; m.dac[0x42][1] = 0; m.dac[0x42][2] = 0;
             m.mem[kSpriteData] = 0x12;
             hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
             for (const auto& entry : fs::directory_iterator(folder / L"original"))
                 fs::copy_file(entry.path(), folder / L"upscaled" / entry.path().filename());
         }
         // New session: establish an ambient tint of ~0.85 first (via a multi-channel placed sprite).
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_folder(folder);
         hd.set_screen_source(kBuffer);
         m.dac[0x51][0] = 60; m.dac[0x51][1] = 60; m.dac[0x51][2] = 60;
         m.dac[0x52][0] = 40; m.dac[0x52][1] = 40; m.dac[0x52][2] = 40;
         m.mem[kSpriteData + 64] = 0x12;
         m.mem[0xA0000 + 12 * 320 + 5] = 0x51;
         hd.draw(m, kSpriteData + 64, 32, 2, kBuffer + 12 * 320 + 5, 0x50);
         hd.placed(0, 10, 10);
         // Dim the multi-channel sprite to 0.85 while the red sprite is dimmed to 0.50 (> 0.22 away from hint 0.85).
         m.dac[0x51][0] = 51; m.dac[0x51][1] = 51; m.dac[0x51][2] = 51;
         m.dac[0x52][0] = 34; m.dac[0x52][1] = 34; m.dac[0x52][2] = 34;
         m.dac[0x41][0] = 30; m.dac[0x41][1] = 0;  m.dac[0x41][2] = 0;
         m.dac[0x42][0] = 20; m.dac[0x42][1] = 0;  m.dac[0x42][2] = 0;
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(200);
         hd.frame(m, refs.data());  // ambient_tint_ is now ~0.85
         m.mem[kSpriteData] = 0x12;
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         // Draw at hint 0.85: creates a missing slot (mismatch > 0.22) and caches miss_hint = 0.85.
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.frame(m, refs.data());
         const bool missed_at_wrong_hint = hd.take_pending().empty();
         // Now shift ambient_tint_ to 0.50 (matching the red sprite!): cached miss must invalidate and match!
         m.dac[0x51][0] = 30; m.dac[0x51][1] = 30; m.dac[0x51][2] = 30;
         m.dac[0x52][0] = 20; m.dac[0x52][1] = 20; m.dac[0x52][2] = 20;
         hd.frame(m, refs.data());  // ambient_tint_ updates to ~0.50
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         hd.frame(m, refs.data());
         const bool matched_after_shift = hd.take_pending().size() == 1;
         return missed_at_wrong_hint && matched_after_shift;
     }},
    {"mild_uniform_fade_near_daylight_stays_hd_and_buffer_sync_updates_deferred_backdrop", [] {
         Fixture f;
         const fs::path folder = f.root / L"hd";
         // Daytime session: save a 5-colour building sprite (including bright yellow 60,60,0) and mark it upscaled.
         {
             nax::Machine m;
             nax::HdSprites hd;
             hd.set_folder(folder);
             m.dac[0x41][0] = 20; m.dac[0x41][1] = 20; m.dac[0x41][2] = 24;
             m.dac[0x42][0] = 30; m.dac[0x42][1] = 30; m.dac[0x42][2] = 36;
             m.dac[0x43][0] = 38; m.dac[0x43][1] = 0;  m.dac[0x43][2] = 0;
             m.dac[0x44][0] = 48; m.dac[0x44][1] = 36; m.dac[0x44][2] = 0;
             m.dac[0x45][0] = 60; m.dac[0x45][1] = 60; m.dac[0x45][2] = 0;
             m.mem[kSpriteData] = 0x12;
             m.mem[kSpriteData + 1] = 0x34;
             m.mem[kSpriteData + 2] = 0x50;
             hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
             for (const auto& entry : fs::directory_iterator(folder / L"original"))
                 fs::copy_file(entry.path(), folder / L"upscaled" / entry.path().filename());
         }
         // New session: 94% uniform fade (mid-tones shift by 1..2 DAC units, bright yellow shifts by 4 DAC units).
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_folder(folder);
         hd.set_screen_source(kBuffer);
         m.dac[0x41][0] = 19; m.dac[0x41][1] = 19; m.dac[0x41][2] = 23;
         m.dac[0x42][0] = 28; m.dac[0x42][1] = 28; m.dac[0x42][2] = 34;
         m.dac[0x43][0] = 36; m.dac[0x43][1] = 0;  m.dac[0x43][2] = 0;
         m.dac[0x44][0] = 45; m.dac[0x44][1] = 34; m.dac[0x44][2] = 0;
         m.dac[0x45][0] = 56; m.dac[0x45][1] = 56; m.dac[0x45][2] = 0;
         m.mem[kSpriteData] = 0x12;
         m.mem[kSpriteData + 1] = 0x34;
         m.mem[kSpriteData + 2] = 0x50;
         m.mem[0xA0000 + 10 * 320 + 5] = 0x41;
         hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         const auto pending = hd.take_pending();
         const bool matched_mild_fade = pending.size() == 1;
         if (!pending.empty()) hd.placed(pending[0].slot, 40, 40);
         hd.frame(m, refs.data());
         const bool rendered_mild_fade = refs[10 * 320 + 5] == nax::HdSprites::encode(40, 41, 1, true);
         // A single mildly blinking bright indicator (5..6 DAC units) with all 4 other colours unchanged must NOT
         // trigger ambient dimming on a 5-colour sprite.
         m.dac[0x41][0] = 20; m.dac[0x41][1] = 20; m.dac[0x41][2] = 24;
         m.dac[0x42][0] = 30; m.dac[0x42][1] = 30; m.dac[0x42][2] = 36;
         m.dac[0x43][0] = 38; m.dac[0x43][1] = 0;  m.dac[0x43][2] = 0;
         m.dac[0x44][0] = 48; m.dac[0x44][1] = 36; m.dac[0x44][2] = 0;
         m.dac[0x45][0] = 54; m.dac[0x45][1] = 54; m.dac[0x45][2] = 0;  // only 5th colour blinked by -6
         hd.frame(m, refs.data());
         const bool mild_single_blink_rejected = refs[10 * 320 + 5] == nax::HdSprites::kNone;
         // Offscreen buffer sync_image_rect updates deferred expected colours before settle so bottom bar stays HD.
         m.dac[0x10][0] = 12; m.dac[0x10][1] = 12; m.dac[0x10][2] = 12;
         m.dac[0x11][0] = 20; m.dac[0x11][1] = 20; m.dac[0x11][2] = 20;
         for (uint32_t i = 0; i < 320 * 20; i++) m.mem[kBuffer + 172 * 320 + i] = 0x10;
         hd.image(m, kBuffer + 172 * 320, 320, 20, 320, kBuffer + 172 * 320, 320);
         // Overwrite an 8x8 strip inside the unsettled offscreen buffer with a shifted dither index (0x11) and sync.
         for (int y = 0; y < 8; y++)
             for (int x = 0; x < 8; x++) {
                 m.mem[kBuffer + (172 + y) * 320 + 8 + x] = 0x11;
                 m.mem[0xA0000 + (172 + y) * 320 + 8 + x] = 0x11;
             }
         hd.sync_image_rect(m, kBuffer + 172 * 320 + 8, 8, 8);
         hd.present_buffer(kBuffer);
         hd.frame(m, refs.data());
         Sleep(200);
         hd.frame(m, refs.data());
         hd.placed(uint32_t(hd.sprite_count() - 1), 100, 100);
         hd.frame(m, refs.data());
         const bool synced_buffer_hd = refs[172 * 320 + 8] == nax::HdSprites::encode(108, 100, 0, false);
         return matched_mild_fade && rendered_mild_fade && mild_single_blink_rejected && synced_buffer_hd;
     }},
    {"world_view_isometric_projection_and_click_anywhere_hit_test", [] {
         nax::WorldView view;
         view.sync_game_camera(30, 40);
         const int W = 2560, H = 1080;
         nax::WorldRenderData data;
         data.win_w = W;
         data.win_h = H;
         data.center_row = view.center_row();
         data.center_col = view.center_col();
         view.compute_projection(W, H, &data.tile_scale_x, &data.tile_scale_y, &data.origin_x, &data.origin_y);
         // Round-trip tile centre -> screen pixel -> screen_to_map must recover the exact tile coordinates.
         float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
         nax::WorldView::project_tile(data, 33.0f, 44.0f, 32, 15, 0, 0, &x0, &y0, &x1, &y1);
         const float cx = (x0 + x1) * 0.5f, cy = (y0 + y1) * 0.5f;
         float fr = 0, fc = 0;
         view.screen_to_map(cx, cy, W, H, &fr, &fc);
         const bool roundtrip_ok = std::abs(fr - 33.0f) < 0.02f && std::abs(fc - 44.0f) < 0.02f;
         // Ultrawide viewport must span significantly more than the original 8x8 diamond.
         int r0 = 0, r1 = 0, c0 = 0, c1 = 0;
         view.visible_bounds(W, H, &r0, &r1, &c0, &c1);
         const bool wide_bounds = (r1 - r0 + 1) >= 20 && (c1 - c0 + 1) >= 20;
         // Clicking a tile inside Utopia's 8x8 core (33, 44) must not request a camera shift.
         const auto inside_hit = view.hit_test(cx, cy, W, H, 30, 40);
         int exp_dx = 0, exp_dy = 0;
         nax::WorldView::game_tile_center_dos(3, 4, &exp_dx, &exp_dy);
         const bool inside_ok = inside_hit.kind == nax::WorldHit::Kind::tile && !inside_hit.need_cam_shift &&
                                inside_hit.tile_row == 33 && inside_hit.tile_col == 44 &&
                                inside_hit.dos_x == exp_dx && inside_hit.dos_y == exp_dy;
         // Clicking a tile outside the 8x8 core (24, 48) requests a logic shift; note_click_shift keeps visual camera steady.
         nax::WorldView::project_tile(data, 24.0f, 48.0f, 32, 15, 0, 0, &x0, &y0, &x1, &y1);
         const auto outside_hit = view.hit_test((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, W, H, 30, 40);
         const float before_r = view.center_row(), before_c = view.center_col();
         view.note_click_shift(outside_hit.shift_cam_row, outside_hit.shift_cam_col);
         view.sync_game_camera(outside_hit.shift_cam_row, outside_hit.shift_cam_col);
         const bool outside_ok = outside_hit.kind == nax::WorldHit::Kind::tile && outside_hit.need_cam_shift &&
                                 outside_hit.tile_row == 24 && outside_hit.tile_col == 48 &&
                                 outside_hit.tile_row >= outside_hit.shift_cam_row &&
                                 outside_hit.tile_row < outside_hit.shift_cam_row + 8 &&
                                 outside_hit.tile_col >= outside_hit.shift_cam_col &&
                                 outside_hit.tile_col < outside_hit.shift_cam_col + 8 &&
                                 std::abs(view.center_row() - before_r) < 1e-4f &&
                                 std::abs(view.center_col() - before_c) < 1e-4f;
         // Clicking inside the docked top HUD strip maps directly to the top status bar in 320x200 DOS coordinates.
         const auto strips = nax::WorldView::compute_hud_strips(W, H);
         const bool strip_flags_ok = strips.size() == 5 && strips[0].key_out_backdrop && strips[2].diamond_clip &&
                                     strips[3].src_x == 268 && strips[3].src_y == 59 && strips[3].src_w == 48 && strips[3].src_h == 100;
         const float top_cx = float(strips[0].dst.left + strips[0].dst.right) * 0.5f;
         const float top_cy = float(strips[0].dst.top + strips[0].dst.bottom) * 0.5f;
         const auto hud_hit = view.hit_test(top_cx, top_cy, W, H, 30, 40);
         const bool hud_ok = strip_flags_ok && hud_hit.kind == nax::WorldHit::Kind::hud &&
                             hud_hit.dos_x >= 150 && hud_hit.dos_x <= 170 && hud_hit.dos_y < 14;
         // Compass diamond clip: centre hits HUD, while transparent outer corner passes through to the visible map tile.
         const float comp_cx = float(strips[2].dst.left + strips[2].dst.right) * 0.5f;
         const float comp_cy = float(strips[2].dst.top + strips[2].dst.bottom) * 0.5f;
         const auto comp_center_hit = view.hit_test(comp_cx, comp_cy, W, H, 30, 40);
         const auto comp_corner_hit = view.hit_test(float(strips[2].dst.left + 2), float(strips[2].dst.top + 2), W, H, 30, 40);
         const bool compass_clip_ok = comp_center_hit.kind == nax::WorldHit::Kind::hud &&
                                      comp_corner_hit.kind == nax::WorldHit::Kind::tile;
         // Top status bar floats as a compact centred pill so open map tiles to its left remain clickable.
         const auto side_map_hit = view.hit_test(8.0f, top_cy, W, H, 30, 40);
         const bool bar_blocked = side_map_hit.kind == nax::WorldHit::Kind::tile;
         // Floating popup hit-testing maps clicks inside popup_dst 1:1 to popup DOS coordinates and blocks clicks outside.
         nax::WorldRenderData pop_world = data;
         pop_world.has_popup = true;
         pop_world.popup_src_x = 0;
         pop_world.popup_src_y = 20;
         pop_world.popup_src_w = 224;
         pop_world.popup_src_h = 170;
         pop_world.popup_dst = nax::WorldView::compute_popup_rect(W, H, 224, 170);
         const float pop_cx = float(pop_world.popup_dst.left + pop_world.popup_dst.right) * 0.5f;
         const float pop_cy = float(pop_world.popup_dst.top + pop_world.popup_dst.bottom) * 0.5f;
         const auto pop_in = view.hit_test(pop_cx, pop_cy, W, H, 30, 40, &pop_world);
         const auto pop_out = view.hit_test(float(pop_world.popup_dst.left - 20), pop_cy, W, H, 30, 40, &pop_world);
         const bool popup_ok = pop_in.kind == nax::WorldHit::Kind::hud &&
                               pop_in.dos_x >= 108 && pop_in.dos_x <= 116 && pop_in.dos_y >= 100 && pop_in.dos_y <= 110 &&
                               pop_out.kind == nax::WorldHit::Kind::none;
         // Scroll-arrow pad (strips[4]) must never overlap the centred bottom message bar (strips[1]) on narrower windows.
         auto rects_overlap = [](const RECT& a, const RECT& b) {
             return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
         };
         bool no_hud_overlap = true;
         for (const auto [nw, nh] : std::array<std::pair<int, int>, 4>{{{1280, 720}, {1920, 1200}, {1280, 960}, {2560, 1080}}}) {
             const auto ns = nax::WorldView::compute_hud_strips(nw, nh);
             if (ns.size() < 5 || rects_overlap(ns[1].dst, ns[4].dst) || rects_overlap(ns[3].dst, ns[4].dst))
                 no_hud_overlap = false;
         }
         // L-shaped advisor portrait popup crops the covered top button row off strips[3] and ignores clicks in the L-cutout.
         nax::WorldRenderData adv_world = data;
         adv_world.has_popup = true;
         adv_world.popup_has_holes = true;
         adv_world.popup_src_x = 0;
         adv_world.popup_src_y = 0;
         adv_world.popup_src_w = 320;
         adv_world.popup_src_h = 200;
         adv_world.popup_dst = nax::WorldView::compute_popup_rect(W, H, 320, 200, true, true);
         nax::HudStrip cropped_bar = strips[3];
         const bool cropped_ok = nax::WorldView::crop_sidebar_for_popup(cropped_bar, &adv_world) &&
                                 cropped_bar.src_y == 79 && cropped_bar.src_h == 80 &&
                                 cropped_bar.dst.top > strips[3].dst.top;
         const float hole_px = float(adv_world.popup_dst.left) + float(adv_world.popup_dst.right - adv_world.popup_dst.left) * 0.80f;
         const float hole_py = float(adv_world.popup_dst.top) + float(adv_world.popup_dst.bottom - adv_world.popup_dst.top) * 0.70f;
         const auto hole_hit = view.hit_test(hole_px, hole_py, W, H, 30, 40, &adv_world);
         const bool lshape_hit_ok = cropped_ok && hole_hit.kind == nax::WorldHit::Kind::none;
         // Empty bottom message box auto-hides so clicks pass through to colony tiles; active 1-line messages fit tightly.
         nax::WorldRenderData empty_msg_world = data;
         empty_msg_world.has_bottom_message = false;
         const float bot_cx = float(strips[1].dst.left + strips[1].dst.right) * 0.5f;
         const float bot_cy = float(strips[1].dst.top + strips[1].dst.bottom) * 0.5f;
         const auto bot_hidden_hit = view.hit_test(bot_cx, bot_cy, W, H, 30, 40, &empty_msg_world);
         nax::WorldRenderData one_line_world = data;
         one_line_world.has_bottom_message = true;
         one_line_world.bottom_msg_src_h = 13;
         nax::HudStrip fitted_bot = strips[1];
         const bool bottom_msg_ok = bot_hidden_hit.kind == nax::WorldHit::Kind::tile &&
                                    nax::WorldView::fit_bottom_strip_for_message(fitted_bot, &one_line_world) &&
                                    fitted_bot.src_y == 170 && fitted_bot.src_h == 13 && fitted_bot.key_out_backdrop &&
                                    fitted_bot.dst.top > strips[1].dst.top && fitted_bot.dst.bottom == strips[1].dst.bottom;
         // Zoom clamps to [kMinZoom, kMaxZoom], and camera bounds enforce Utopia's 75x100 map (max cam row 67, col 92).
         view.adjust_zoom(100.0f);
         const bool max_zoom_ok = std::abs(view.zoom() - nax::WorldView::kMaxZoom) < 1e-4f;
         view.adjust_zoom(0.01f);
         const bool min_zoom_ok = std::abs(view.zoom() - nax::WorldView::kMinZoom) < 1e-4f;
         view.sync_game_camera(80, 99);
         const bool recenter_ok = view.game_cam_row() == nax::WorldView::kMaxGameCamRow &&
                                  view.game_cam_col() == nax::WorldView::kMaxGameCamCol &&
                                  std::abs(view.center_row() - 70.5f) < 1e-4f && std::abs(view.center_col() - 95.5f) < 1e-4f;
         int br0 = 0, br1 = 0, bc0 = 0, bc1 = 0;
         view.visible_bounds(W, H, &br0, &br1, &bc0, &bc1);
         const bool bounds_clamped = br1 < nax::WorldView::kMapRows && bc1 < nax::WorldView::kMapCols;
         return roundtrip_ok && wide_bounds && inside_ok && outside_ok && hud_ok && compass_clip_ok && bar_blocked &&
                popup_ok && no_hud_overlap && lshape_hit_ok && bottom_msg_ok && max_zoom_ok && min_zoom_ok && recenter_ok &&
                bounds_clamped;
     }},
    {"world_fallback_atlas_packs_deduplicates_and_wraps_cleanly", [] {
         nax::WorldFallbackAtlas atlas;
         std::vector<uint32_t> px1(32 * 15, 0xFF112233u);
         std::vector<uint32_t> px2(32 * 24, 0xFF445566u);
         const uint64_t v0 = atlas.version();
         const uint32_t id1 = atlas.get_or_add(101, 32, 15, px1.data());
         const uint32_t id1_again = atlas.get_or_add(101, 32, 15, px1.data());
         const uint32_t id2 = atlas.get_or_add(202, 32, 24, px2.data());
         const auto* e1 = atlas.entry(id1);
         const auto* e2 = atlas.entry(id2);
         const bool packed_ok = id1 == 1 && id1_again == 1 && id2 == 2 && e1 && e2 &&
                                e1->w == 32 && e1->h == 15 && e2->w == 32 && e2->h == 24 &&
                                atlas.version() == v0 + 2 &&
                                atlas.pixels()[size_t(e1->y) * nax::WorldFallbackAtlas::kSize + e1->x] == 0xFF112233u &&
                                atlas.pixels()[size_t(e2->y) * nax::WorldFallbackAtlas::kSize + e2->x] == 0xFF445566u;
         // Fill the 1024x1024 atlas until it wraps and increments clear_count(), ensuring the newly added sprite stays valid.
         std::vector<uint32_t> big(256 * 256, 0xFF778899u);
         const uint64_t clears_before = atlas.clear_count();
         uint32_t last_id = 0;
         for (uint64_t k = 1000; k < 1020; k++) last_id = atlas.get_or_add(k, 256, 256, big.data());
         const auto* wrapped = atlas.entry(last_id);
         const bool wrapped_ok = atlas.clear_count() > clears_before && wrapped != nullptr && wrapped->w == 256 && wrapped->h == 256 &&
                                 atlas.pixels()[size_t(wrapped->y) * nax::WorldFallbackAtlas::kSize + wrapped->x] == 0xFF778899u;
         atlas.clear();
         const bool cleared_ok = atlas.entry(1) == nullptr && atlas.clear_count() > clears_before + 1;
         return packed_ok && wrapped_ok && cleared_ok;
     }},
    {"world_sprite_resolution_and_offscreen_map_visibility_gate", [] {
         Fixture f;
         const fs::path folder = f.root / L"hd";
         // Session 1: save one tile sprite and mark it upscaled.
         {
             nax::Machine m;
             nax::HdSprites hd;
             hd.set_folder(folder);
             m.dac[0x41][0] = 30; m.dac[0x41][1] = 40; m.dac[0x41][2] = 55;
             m.dac[0x42][0] = 45; m.dac[0x42][1] = 50; m.dac[0x42][2] = 60;
             m.mem[kSpriteData] = 0x12;
             hd.draw(m, kSpriteData, 32, 2, kDest, 0x40);
             for (const auto& entry : fs::directory_iterator(folder / L"original"))
                 fs::copy_file(entry.path(), folder / L"upscaled" / entry.path().filename());
         }
         // Session 2: resolve_world_sprite must match the upscaled sprite and queue upload without creating new PNGs.
         nax::Machine m;
         nax::HdSprites hd;
         hd.set_folder(folder);
         m.dac[0x41][0] = 30; m.dac[0x41][1] = 40; m.dac[0x41][2] = 55;
         m.dac[0x42][0] = 45; m.dac[0x42][1] = 50; m.dac[0x42][2] = 60;
         m.mem[kSpriteData] = 0x12;
         m.mem[kSpriteData + 64] = 0x21;  // unknown sprite not on disk
         hd.begin_world_frame();
         const auto info_before = hd.resolve_world_sprite(m, kSpriteData, 32, 2, 0x40, true);
         const auto unknown_info = hd.resolve_world_sprite(m, kSpriteData + 64, 32, 2, 0x40, true);
         const auto pending = hd.take_pending();
         size_t orig_count = 0;
         for (const auto& entry : fs::directory_iterator(folder / L"original")) { (void)entry; orig_count++; }
         if (!pending.empty()) hd.placed(pending[0].slot, 80, 90);
         hd.begin_world_frame();
         const auto info_after = hd.resolve_world_sprite(m, kSpriteData, 32, 2, 0x40, true);
         const bool world_cells_ok = hd.on_screen_cells() >= uint64_t((32 + 2) * (2 + 2));
         const bool resolve_ok = !info_before.placed && !unknown_info.placed && pending.size() == 1 && orig_count == 1 &&
                                 info_after.placed && info_after.atlas_x == 80 && info_after.atlas_y == 90 && !info_after.dimmed &&
                                 world_cells_ok;
         // is_showing_offscreen_map must be true on clean offscreen presentation, and false when a popup or VRAM screen is active.
         hd.set_screen_source(kBuffer);
         hd.present_buffer(kBuffer);
         const bool clean_map = hd.is_showing_offscreen_map();
         // Direct-VRAM text clear on the bottom message bar (sync_image_rect) must NOT trigger a fake popup.
         hd.set_screen_source(kBuffer);
         for (uint32_t i = 0; i < 64000; i++) {
             m.mem[kBuffer + i] = 0x07;
             m.mem[0xA0000 + i] = 0x07;
         }
         hd.image(m, kBuffer, 320, 200, 320, kBuffer, 320);
         hd.present_buffer(kBuffer);
         std::vector<uint32_t> refs(64000);
         hd.frame(m, refs.data());
         Sleep(200);
         hd.frame(m, refs.data());
         m.mem[0xA0000 + 176 * 320 + 16] = 0x08;
         hd.sync_image_rect(m, 0xA0000 + 176 * 320 + 16, 8, 8);
         int tx0 = 0, ty0 = 0, tx1 = 0, ty1 = 0;
         const bool text_clear_not_popup = hd.is_showing_offscreen_map(&tx0, &ty0, &tx1, &ty1) && tx1 == 0 && ty1 == 0;
         hd.clear_rect(0xA0000 + 40 * 320 + 60, 100, 80);  // direct-VRAM popup over map
         const bool vram_popup_blocks = !hd.is_showing_offscreen_map();
         hd.present_buffer(kBuffer);
         const bool back_to_clean = hd.is_showing_offscreen_map();
         hd.clear_rect(kBuffer + 40 * 320 + 60, 100, 80);  // offscreen popup over map
         hd.present_buffer(kBuffer);
         const bool offscreen_popup_blocks = !hd.is_showing_offscreen_map();
         int px0 = 0, py0 = 0, px1 = 0, py1 = 0;
         const bool popup_bounds_ok = hd.is_showing_offscreen_map(&px0, &py0, &px1, &py1) &&
                                      px0 == 60 && py0 == 40 && px1 == 160 && py1 == 120;
         // Tall in-map popup (256x176 such as SELECT A BUILDING..) must still float cleanly instead of snapping to 4:3.
         hd.clear_rect(kBuffer + 16 * 320 + 0, 256, 176);
         hd.present_buffer(kBuffer);
         const bool tall_popup_floats = hd.is_showing_offscreen_map(&px0, &py0, &px1, &py1) &&
                                        px0 == 0 && py0 == 16 && px1 == 256 && py1 == 192;
         // L-shaped advisor/intelligence popup (left column 0..160, 0..199 + top-right portrait 216..319, 0..76 +
         // direct-VRAM sprite at x=180, y=50) floats with popup_has_holes=true, marks unmasked gaps with kPopupHole,
         // preserves visible HD sidebar buttons (268..315, 79..158), and keeps direct-VRAM sprites covered.
         hd.restore_rect(kBuffer, 320, 200);
         hd.present_buffer(kBuffer);
         hd.clear_rect(0xA0000 + 0 * 320 + 0, 160, 200);
         hd.clear_rect(0xA0000 + 0 * 320 + 216, 104, 76);
         hd.draw(m, kSpriteData, 32, 2, 0xA0000 + 50 * 320 + 180, 0x40);  // unmasked direct-VRAM sprite inside x < 214
         bool has_holes = false;
         const bool lshape_ok = hd.is_showing_offscreen_map(&px0, &py0, &px1, &py1, &has_holes) &&
                                has_holes && px0 == 0 && py0 == 0 && px1 == 320 && py1 == 200;
         std::fill(refs.begin(), refs.end(), nax::HdSprites::kNone);
         refs[140 * 320 + 280] = 0x00100042u;  // simulated HD sidebar button ref inside 268..315, 79..158
         hd.mark_popup_holes(refs.data(), px0, py0, px1, py1);
         const bool holes_marked = refs[20 * 320 + 20] == nax::HdSprites::kNone &&
                                   refs[20 * 320 + 240] == nax::HdSprites::kNone &&
                                   refs[50 * 320 + 180] == nax::HdSprites::kNone &&
                                   refs[140 * 320 + 240] == nax::HdSprites::kPopupHole &&
                                   refs[140 * 320 + 280] == 0x00100042u;
         // Full-screen 320x200 overlay (extending across both top-right and bottom-right HUD) falls back to classic 4:3.
         hd.clear_rect(kBuffer, 320, 200);
         hd.present_buffer(kBuffer);
         const bool full_overlay_blocks = !hd.is_showing_offscreen_map(&px0, &py0, &px1, &py1);
         // Resetting world frame clears stale world_on_screen_cells_ when leaving Expanded View.
         hd.begin_world_frame();
         const bool world_cells_reset = hd.on_screen_cells() == 0;
         hd.set_screen_source(0xA0000);  // full-screen VRAM sub-screen (Options / Advisors / Planetary Map)
         const bool vram_screen_blocks = !hd.is_showing_offscreen_map(&px0, &py0, &px1, &py1);
         return resolve_ok && clean_map && text_clear_not_popup && vram_popup_blocks && back_to_clean &&
                offscreen_popup_blocks && popup_bounds_ok && tall_popup_floats && lshape_ok && holes_marked &&
                full_overlay_blocks && world_cells_reset && vram_screen_blocks;
     }},
    {"save_handle_not_inherited_by_child_process_and_commits_cleanly", [] {
         Fixture f;
         const int h = f.open("save1.dat", 0x3C, 0);
         if (h < 0 || !f.write(h, "payload-123")) return false;
         // Spawn a short-lived child process with bInheritHandles=TRUE; because DOS handles use _O_NOINHERIT ("N"),
         // the child must NOT inherit the open .nax-tmp handle and closing the save must commit cleanly.
         wchar_t cmd[] = L"cmd.exe /c ping -n 2 127.0.0.1 >nul";
         STARTUPINFOW si{sizeof(si)};
         PROCESS_INFORMATION pi{};
         const bool spawned = CreateProcessW(nullptr, cmd, nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi) != 0;
         const bool closed = f.close(h);
         if (spawned) {
             TerminateProcess(pi.hProcess, 0);
             WaitForSingleObject(pi.hProcess, 3000);
             CloseHandle(pi.hThread);
             CloseHandle(pi.hProcess);
         }
         return spawned && closed && read_file(f.saves / L"files" / L"save1.dat") == "payload-123" && no_temporary_files(f.saves);
     }},
    {"readonly_original_game_file_opens_for_write_via_copy_on_write", [] {
         Fixture f;
         const fs::path orig = f.data / L"config.dat";
         write_file(orig, "original");
         SetFileAttributesW(orig.c_str(), FILE_ATTRIBUTE_READONLY);
         const int h = f.open("config.dat", 0x3D, 2);
         const bool wrote = h >= 0 && f.write(h, "modified");
         const bool closed = h >= 0 && f.close(h);
         SetFileAttributesW(orig.c_str(), FILE_ATTRIBUTE_NORMAL);
         return wrote && closed && read_file(orig) == "original" &&
                read_file(f.saves / L"files" / L"config.dat") == "modified" && no_temporary_files(f.saves);
     }},
    {"failed_rooted_chdir_keeps_previous_cwd_and_findfirst_reports_subdir_size", [] {
         Fixture f;
         const std::string body(1234, 'x');
         write_file(f.data / L"scenario" / L"level1.map", body);
         // 1. FindFirst on a subdirectory pattern while cwd is root must report the real 1234-byte size in the DTA.
         f.put_string(0x2000, 0, "scenario\\*.map");
         f.m.c.ds = 0x2000;
         f.m.c.d.x = 0;
         f.m.c.a.h = 0x4E;
         f.m.c.c.x = 0;
         f.m.service_for_test(0x21);
         const bool found = !f.m.c.cf;
         f.m.c.a.h = 0x2F;
         f.m.service_for_test(0x21);
         const uint32_t dta_lin = nax::Machine::linear(f.m.c.es, f.m.c.b.x);
         const uint32_t reported_size = uint32_t(f.m.mem[dta_lin + 26]) | (uint32_t(f.m.mem[dta_lin + 27]) << 8) |
                                        (uint32_t(f.m.mem[dta_lin + 28]) << 16) | (uint32_t(f.m.mem[dta_lin + 29]) << 24);
         // 2. Change into scenario, then attempt a failing rooted CHDIR; cwd must remain "scenario" so opening level1.map works.
         f.put_string(0x2000, 0, "scenario");
         f.m.c.ds = 0x2000;
         f.m.c.d.x = 0;
         f.m.c.a.h = 0x3B;
         f.m.service_for_test(0x21);
         const bool cd_ok = !f.m.c.cf;
         f.put_string(0x2000, 0, "\\nonexistent");
         f.m.c.ds = 0x2000;
         f.m.c.d.x = 0;
         f.m.c.a.h = 0x3B;
         f.m.service_for_test(0x21);
         const bool bad_cd_failed = f.m.c.cf;
         const int h = f.open("level1.map", 0x3D, 0);
         const bool still_in_subdir = h >= 0 && f.close(h);
         return found && reported_size == 1234 && cd_ok && bad_cd_failed && still_in_subdir;
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
