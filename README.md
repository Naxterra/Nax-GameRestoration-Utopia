# Nax Game Restoration

Nax Game Restoration makes classic DOS games run natively on Windows 11 without an emulator. A translator turns the 16-bit x86 code of a legally owned copy into C++ ahead of time, and a native runtime replaces DOS and the PC hardware with Win32 services, a Direct3D 12 presenter and XAudio2 sound.

The repository contains only the translator and the runtime. It contains no game code, graphics, sound or data. Every build reads the original files from a copy that the user owns, and the translated program stays on that user's machine.

## Status

| Game | State |
|---|---|
| Utopia: The Creation of a Nation (1992) | Playable. The intro, the options screen and the first scenario run natively with mouse control, AdLib music, sound effects and PC speaker sound. |
| Dune II: The Building of a Dynasty (1992) | Not started. |

Some parts of Utopia have not been played through yet. If the game reaches code that was not translated, it stops with a message that names the address, and that address is then added to the game's seed list.

## Requirements

- Windows 11 (Windows 10 with Direct3D 12 should also work).
- Visual Studio Build Tools with the C++ workload, installed with `winget install Microsoft.VisualStudio.BuildTools`.
- Python 3 with the Capstone disassembler, installed with `python -m pip install capstone`.
- Your own copy of the game.

## Building

Copy the files of your copy of Utopia (the folder that contains `u.mdg`, `i.mdg` and the `scenario` folder) into `gamedata\utopia`, then run the build script from PowerShell 7:

```powershell
.\build.ps1 -Game utopia
```

The script translates the original program files into `generated\utopia` and compiles `build\utopia\Utopia.exe`. A different game folder can be given with `-GameFolder <folder>`. With `-Tests` the script first builds and runs the runtime tests, which need no game files.

## Playing

Start `build\utopia\Utopia.exe`. While the window is active, the mouse controls the game and the Windows pointer stays inside the window; Alt+Tab releases it. Alt+Enter switches between a window and borderless fullscreen. F11 switches between four picture filters: sharp pixels, smooth edges (stair steps along diagonal and round edges are redrawn as smooth lines), a CRT look with scanlines, and AI upscaling; the choice is remembered for the next start. The AI filter redraws the picture at four times the size with the realesr-animevideov3 network from the Real-ESRGAN project, which runs as Direct3D 12 compute shaders on the graphics card and only works when the picture changes. Because the network's own picture is soft, only the shapes it draws are used: every screen pixel is filled with the closest of the game's own colours around it, so outlines, icons and letters get hard, smooth edges at the full screen resolution, while the game's fine textures and single dots keep their original pixels. F12 saves a screenshot exactly as the picture appears on screen.

Utopia does not play music on its own. On the options screen, click the disk button with the note, choose one of the four tunes, and use the note button to switch the music on and off.

| Option | Effect |
|---|---|
| `--sound adlib` | AdLib music and effects (the default). |
| `--sound beeper` | PC speaker sound. |
| `--sound roland` | Roland MT-32. This driver runs, but its output is not played yet. |
| `--mute` | No sound output. |
| `--language en`, `fr`, `de` or `it` | The language of the game's texts. |
| `--skip-intro` | Starts the game without the intro. |
| `--filter sharp`, `smooth`, `crt` or `ai` | The picture filter for this start, instead of the one last chosen with F11. |
| `--data <folder>` | Uses the game files in another folder. |

## Where files go

The original game files are never changed. Save games and every other file the game writes go to `%LOCALAPPDATA%\Nax Game Restoration\utopia\files`, and the game reads its own saves from there first. Each write goes to a temporary file that replaces the old version only when the game closes the file normally, so an interrupted save keeps the previous one. Screenshots go to the `shots` folder and a runtime log to `runtime.log` in the same place.

## Layout

| Folder | Content |
|---|---|
| `tools/recomp` | The translator: the MZ and EXEPACK loader, code discovery and the C++ generator, written in Python. |
| `tools/ai` | The converter that turns a Real-ESRGAN checkpoint into the AI filter's weight file without PyTorch, and a CPU reference of the network for checking the GPU version. |
| `models` | The AI filter's weights and their license. The build copies them next to the program. |
| `runtime` | The native runtime: CPU helpers, DOS, BIOS, mouse and VGA services, the OPL2 synthesiser, sound output and the Direct3D 12 window. |
| `games/<game>` | What each game needs: the replacement for its launcher and the seed lists of entry points that discovery cannot prove. |
| `tests` | Python tests for the translator and C++ tests for the runtime. |
| `gamedata/<game>` | Your copy of the game. Git ignores this folder. |
| `generated`, `build` | The translated code and the built program, both derived from your copy. Git ignores these folders. |

## Third-party components

The AI filter's weights come from the Real-ESRGAN project and are under the BSD 3-Clause License. See [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).

## License

This software is under the Nax Source-Available License 1.0. You may install and use the official release builds, privately and at work, and you may read the source code. Modifying the software, redistributing it and selling it need the written permission of the owner. See [LICENSE](LICENSE).
