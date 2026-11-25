// What each game build provides to the shared Windows program.
#pragma once

#include <map>
#include <string>

#include "machine.h"
#include "world_view.h"

namespace game {

const wchar_t* title();
const wchar_t* data_folder_name();  // also names the save folder
const wchar_t* required_file();     // a file that identifies the game folder
void register_modules(nax::Machine& m);
int run(nax::Machine& m, const std::map<std::wstring, std::wstring>& options);

// Optional expanded native-resolution world viewport hooks (implemented by games with a live tile map).
bool build_world_render(const nax::Machine& m, nax::HdSprites& hd, nax::WorldView& view, nax::WorldFallbackAtlas& fallback,
                        int win_w, int win_h, nax::WorldRenderData* out);
void queue_world_input(bool set_cursor, int cursor_x, int cursor_y, bool set_cam, int cam_row, int cam_col, int set_buttons = -1);
void release_world_buttons(nax::Machine& m, int clear_bits);
// Restores the clean saved pixels under Utopia's VRAM software cursor inside a temporary VRAM snapshot so
// docked HUD strips and floating popups do not show a duplicate lagging cursor beneath our native cursor.
bool erase_vram_cursor(const nax::Machine& m, uint8_t* vram_copy);

}  // namespace game
