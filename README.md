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

The script translates the original program files into `generated\utopia` and compiles `build\utopia\Utopia.exe`. A different game folder can be given with `-GameFolder <folder>`. With `-Tests` the script first builds and runs the runtime tests, which need no game files. With `-Trace` the translated program also records which instruction wrote each byte of memory, which helps to find a game's drawing routines.

## Playing

Start `build\utopia\Utopia.exe`. While the window is active, the mouse controls the game and the Windows pointer stays inside the window; Alt+Tab releases it. Alt+Enter switches between a window and borderless fullscreen. F11 switches between four picture filters: sharp pixels, smooth edges (stair steps along diagonal and round edges are redrawn as smooth lines), a CRT look with scanlines, and AI upscaling; the choice is remembered for the next start. The AI filter draws the game's map sprites (terrain, buildings and other objects) from AI-upscaled pictures at the full screen resolution, as described in the next section. Everything else is redrawn at four times the size by the realesr-animevideov3 network from the Real-ESRGAN project, which runs as Direct3D 12 compute shaders on the graphics card, and every screen pixel then takes the closest of the game's own colours, so outlines, icons and letters keep hard edges. F12 saves a screenshot exactly as the picture appears on screen.

## AI-upscaled sprites

While the game runs, the runtime watches the game's sprite drawing routines and saves every sprite it draws once as a PNG in `%LOCALAPPDATA%\Nax Game Restoration\utopia\hd\original`. The more of the game you have seen, the more sprites are there. The following command runs the AI network on every saved sprite that has no upscaled picture yet and writes a picture four times as large to the `hd\upscaled` folder next to it:

```powershell
python -I tools\ai\upscale_sprites.py utopia
```

The tool works best with a large ESRGAN model. Download `RealESRGAN_x4plus.pth` from the releases of the Real-ESRGAN project on GitHub (BSD 3-Clause License) and put it in `%LOCALAPPDATA%\Nax Game Restoration\models`, where the tool looks for it first. It takes about a second per sprite on the processor. Without it the tool uses the small network that comes with this repository, which gives softer pictures. Any other 4x ESRGAN checkpoint, for example the pixel art model `4x-PixelPerfectV4` from OpenModelDB, can be given with `--model <file>`. The checkpoints are read without PyTorch by a loader that refuses anything but weights.

A running game picks the new pictures up within a few seconds. With the AI filter, every screen pixel that belongs to a sprite is then drawn from that sprite's upscaled picture. The upscaled pictures are ordinary PNG files, so they can also be improved by hand or with another tool, as long as each keeps exactly four times the size of its original. The tool skips sprites that already have a picture unless `--again` is given, so edited pictures are kept. The pictures are made from your copy of the game and stay on your machine.

## Music and options

Utopia does not play music on its own. On the options screen, click the disk button with the note, choose one of the four tunes, and use the note button to switch the music on and off. The program also takes these options:

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

The original game files are never changed. Save games and every other file the game writes go to `%LOCALAPPDATA%\Nax Game Restoration\utopia\files`, and the game reads its own saves from there first. Each write goes to a temporary file that replaces the old version only when the game closes the file normally, so an interrupted save keeps the previous one. Screenshots go to the `shots` folder, the saved and upscaled sprites to the `hd` folder, and a runtime log to `runtime.log` in the same place.

## Layout

| Folder | Content |
|---|---|
| `tools/recomp` | The translator: the MZ and EXEPACK loader, code discovery and the C++ generator, written in Python. |
| `tools/ai` | The converter that turns a Real-ESRGAN checkpoint into the AI filter's weight file without PyTorch, a CPU reference of the network for checking the GPU version, the ESRGAN network for large models, and the tool that upscales the saved sprites. |
| `models` | The AI filter's weights and their license. The build copies them next to the program. |
| `runtime` | The native runtime: CPU helpers, DOS, BIOS, mouse and VGA services, the OPL2 synthesiser, sound output, the sprite tracker and the Direct3D 12 window. |
| `games/<game>` | What each game needs: the replacement for its launcher and the seed lists of entry points that discovery cannot prove, and the hooks that report the game's sprite drawing. |
| `tests` | Python tests for the translator and C++ tests for the runtime. |
| `gamedata/<game>` | Your copy of the game. Git ignores this folder. |
| `generated`, `build` | The translated code and the built program, both derived from your copy. Git ignores these folders. |

## Third-party components

The AI filter's weights come from the Real-ESRGAN project and are under the BSD 3-Clause License. See [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).

## License

This software is under the Nax Source-Available License 1.0. You may install and use the official release builds, privately and at work, and you may read the source code. Modifying the software, redistributing it and selling it need the written permission of the owner. See [LICENSE](LICENSE).
