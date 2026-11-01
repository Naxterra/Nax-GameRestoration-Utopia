// What each game build provides to the shared Windows program.
#pragma once

#include <map>
#include <string>

#include "machine.h"

namespace game {

const wchar_t* title();
const wchar_t* data_folder_name();  // also names the save folder
const wchar_t* required_file();     // a file that identifies the game folder
void register_modules(nax::Machine& m);
int run(nax::Machine& m, const std::map<std::wstring, std::wstring>& options);

}  // namespace game
