// Tests for the runtime's DOS file services, sound timing and sprite tracking. Each test drives INT 21h directly on a fresh machine whose game
// folder and save folder are temporary directories, and prints one line per test with PASS or FAIL.
#include <windows.h>

#include <algorithm>
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
         const std::string old_key(name.begin(), name.begin() + 16);
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
         return none_without_picture && hd.on_screen_cells() == 33 * 3;
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
