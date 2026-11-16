// Utopia: The Creation of a Nation (Gremlin Graphics, 1992).
//
// The original launcher UTOPIA.EXE asks for graphics, sound and language and then starts I.MDG (the intro)
// and U.MDG (the game) with a six-character command tail: graphics letter, sound letter, language letter,
// "M", the video mode to restore on exit, and "P" for the game or a space for the intro. This file takes the
// place of the launcher. The translated modules are generated from the user's own copy at build time.
#include <string>

#include "runtime/game.h"
#include "runtime/hd_sprites.h"

const nax::ModuleInfo* nax_module_i();
const nax::ModuleInfo* nax_module_u();
const nax::ModuleInfo* nax_module_beeper();
const nax::ModuleInfo* nax_module_adlib();
const nax::ModuleInfo* nax_module_mt32();

// The hooks listed in hooks-u.txt. Each runs where one of the game's three map sprite routines has set up a
// sprite: DS:SI points to its 4-bit pixels, DS:BX to its entry in the sprite table (whose first byte is the
// colour base), and DI is where its bottom row starts. The sprites are 32 pixels wide.
namespace {
void report_sprite(nax::Machine& m, uint16_t rows, uint16_t target_segment) {
    if (!m.hd) return;
    m.hd->draw(m, nax::Machine::linear(m.c.ds, m.c.si), 32, rows, nax::Machine::linear(target_segment, m.c.di), m.rd8(m.c.ds, m.c.b.x));
}
}  // namespace

void utopia_sprite_rows_in_cx(nax::Machine& m) { report_sprite(m, m.c.c.x, m.c.es); }
void utopia_sprite_rows_in_cx_es_in_ax(nax::Machine& m) { report_sprite(m, m.c.c.x, m.c.a.x); }
void utopia_sprite_rows_in_bp(nax::Machine& m) { report_sprite(m, m.c.bp, m.c.es); }

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
    if (!options.count(L"skip-intro")) {
        m.run_program(nax_module_i(), tail);
        if (m.stopped()) return -1;
    }
    tail[5] = 'P';
    return m.run_program(nax_module_u(), tail);
}

}  // namespace game
