# Nax Game Restoration

Nax Game Restoration makes classic DOS games run natively on Windows 11 without an emulator. A translator turns the 16-bit x86 code of a legally owned copy into C++ ahead of time, and a native runtime replaces DOS and the PC hardware with Win32 services, a Direct3D 12 presenter and XAudio2 sound.

The repository contains only the translator and the runtime. It contains no game code, graphics, sound or data. Every build reads the original files from a copy that the user owns, and the translated program stays on that user's machine.

## Status

| Game | State |
|---|---|
| Utopia: The Creation of a Nation (1992) | Playable. The intro, options, planetary radar, advisors, and colony gameplay run natively on Windows 11 with ultrawide Expanded View, floating HUD and dialogs, AI-remastered HD graphics with ambient nighttime dimming, mouse control, AdLib music, sound effects, and PC speaker sound. |
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

The script translates the original program files into `generated\utopia`, compiles `build\utopia\Utopia.exe`, and copies `models\`, `LICENSE`, and `THIRD-PARTY-NOTICES.txt` next to the executable. A different game folder can be given with `-GameFolder <folder>`. With `-Tests` the script first builds and runs the runtime tests, which need no game files. With `-Trace` the translated program also records which instruction wrote each byte of memory, which helps to find a game's drawing routines.

## Playing

Start `build\utopia\Utopia.exe`. On first launch, the game defaults to **AI upscaling** and **Expanded View** (falling back cleanly to sharp pixels if the AI model is unavailable), and remembers any changes you make with `F9` or `F11` in `settings.ini`.

Expanded View renders Utopia's full 75x100 isometric colony map directly at native monitor resolution with smooth mouse-wheel zoom, panning via `WASD`, arrow keys, middle-drag, or right-drag, and click-anywhere building interaction using Utopia's live 10x10 software cursor. Docked HD HUD strips float cleanly over the terrain, the empty bottom message console hides automatically until Utopia prints a message (expanding into a compact dark-glass toast sized to active text lines), and in-game dialogs such as `SELECT A BUILDING..`, building info, Trade/Finance, and Intelligence Reports float cleanly over the dimmed colony map while full-screen sub-screens (Options, Advisors, Planetary Radar/Map) render at crisp 4:3 aspect ratio.

| Control | Action |
|---|---|
| `F9` | Switch between Expanded ultrawide view and classic 4:3 view (saved in `settings.ini`). |
| `F11` | Cycle picture filters: sharp pixels, smooth edges, CRT scanlines, and AI upscaling (saved in `settings.ini`). |
| `F12` | Save a PNG screenshot of the rendered screen to `%LOCALAPPDATA%\Nax Game Restoration\utopia\shots`. |
| `Alt+Enter` | Toggle between windowed mode and borderless fullscreen. |
| Mouse wheel | Zoom the colony map in or out in Expanded View (`0.45x` to `2.00x`). |
| `WASD`, Arrow keys, Middle-drag, Right-drag | Pan smoothly across the 75x100 planet map in Expanded View (a short right-click still cancels build mode). |
| `Right Ctrl`, Middle click, `Alt+Tab`, `Alt+Space`, `Alt+F4` | Release mouse capture cleanly in classic windowed mode or close the window. |

In classic 4:3 windowed mode, clicking inside the game picture activates mouse capture without forwarding a stray click, while clicking outside the picture leaves the Windows cursor free. The AI filter draws remastered sprites, buildings, terrain tiles, and screens from high-resolution pictures with smooth two-layer alpha compositing, double-buffered zero-flicker presentation, and ambient nighttime dimming, while keeping bitmap fonts, cursors, and un-remastered details razor-sharp with sharp-bilinear scaling.

## AI-upscaled sprites

While the game runs, the runtime watches the game's sprite drawing routines and saves every sprite it draws once as a PNG in `%LOCALAPPDATA%\Nax Game Restoration\utopia\hd\original`. The more of the game you have seen, the more sprites are there. The following command runs the AI network on every saved sprite that has no upscaled picture yet and writes a picture four times as large to the `hd\upscaled` folder next to it:

```powershell
python -I tools\ai\upscale_sprites.py utopia
```

The tool works best with a large ESRGAN model. Download `RealESRGAN_x4plus.pth` from the releases of the Real-ESRGAN project on GitHub (BSD 3-Clause License) and put it in `%LOCALAPPDATA%\Nax Game Restoration\models`, where the tool looks for it first. It takes about a second per sprite on the processor. Without it the tool uses the small network that comes with this repository, which gives softer pictures. Any other 4x ESRGAN checkpoint, for example the pixel art model `4x-PixelPerfectV4` from OpenModelDB, can be given with `--model <file>`. The checkpoints are read without PyTorch by a loader that refuses anything but weights.

### Redrawing sprites with an image editing model

An upscaler can only sharpen what a 32-pixel sprite already shows, and on such small pictures it mostly invents noise. With an NVIDIA graphics card with 16 GB, `tools\ai\remaster.py` goes further: the image editing model Qwen-Image-Edit-2511 paints every saved sprite and screen picture again as detailed 24-bit artwork, keeping its shape, layout and colours. Buildings get real walls, windows and roofs, natural geological formations receive biome-matched mountain and ice-chasm prompts that preserve pitch-black abyss voids, toxic-lake shorelines are composited seamlessly from the HD acid water and ground tiles, and compass markers are rendered deterministically at 8x with crisp bevelled edges. Ground tiles are painted as one continuous field of identical tiles, so the ground has no borders between tiles. Flat tiles such as landing pads keep their painted markings where they are. While the game runs, the runtime records in `hd\layout.txt` which sprites it draws side by side, and flat tiles with markings that lie next to each other, such as the four tiles of a landing pad, and the pieces of a building that stands on several tiles, such as the command center, are painted together as one picture and cut apart again, so a pad or a building looks like one and not like four small ones. A tile of such a group that already has a picture keeps it unless `--again` is given. Each sprite is painted several times (`--tries`, 4 by default), and the version closest to the original pixels is kept. Afterwards the broad colours are gently anchored toward the original while letting rich 24-bit lighting and material detail shine through. Sprites drawn while the screen was black or under ambient nighttime dimming are skipped automatically. The tool needs its own Python environment with PyTorch, which is kept apart from the system's Python:

```powershell
py -3.15 -m venv "$env:LOCALAPPDATA\Nax Game Restoration\ai-env"
& "$env:LOCALAPPDATA\Nax Game Restoration\ai-env\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
& "$env:LOCALAPPDATA\Nax Game Restoration\ai-env\Scripts\python.exe" -m pip install diffusers transformers accelerate safetensors huggingface_hub gguf peft
& "$env:LOCALAPPDATA\Nax Game Restoration\ai-env\Scripts\python.exe" -I tools\ai\remaster.py utopia
```

On first use it downloads about 30 GB into `%LOCALAPPDATA%\Nax Game Restoration\hf-cache`: the 4-bit form of the model from unsloth/Qwen-Image-Edit-2511-GGUF, the text encoder and the other parts of Qwen/Qwen-Image-Edit-2511, and the 8-step Lightning adapter from lightx2v/Qwen-Image-Edit-2511-Lightning, all under the Apache License 2.0. Both large parts of the model stay in main memory and come to the graphics card a few layers at a time, so the tool needs about 40 GB of free main memory. Each try of a sprite takes about half a minute.

Once this environment exists, nothing has to be started by hand. Whenever the game has saved new sprites, it runs `remaster.py` in the background on the graphics card, at low priority and only with models that are already on the machine, and the new pictures appear in the game when they are done. During ambient nighttime or weather dimming, the runtime keeps daytime HD sprites active and applies the scene's ambient lighting tint smoothly while keeping emissive lights bright; only non-uniform fades fall back to original pixels. The tool's output goes to `hd\redraw.log`, and `--no-redraw` switches the background runs off for one start. The tests of `remaster.py` run with the Python of this environment, with `-m unittest discover -s tests -p test_remaster.py`; other Pythons skip them.

The older tool `tools\ai\generate_sprites.py` enlarges sprites with RealESRGAN_x4plus and can add Stable Diffusion XL detail and a seamless ground texture made by `tools\ai\generate_texture.py`. It is kept for comparison, but its results are much weaker.

A running game picks the new pictures up within a few seconds. With the AI filter, every screen pixel that belongs to a sprite is then drawn from that sprite's upscaled picture. The upscaled pictures are ordinary PNG files, so they can also be improved by hand or with another tool, as long as each keeps exactly four or eight times the size of its original. The tool skips sprites that already have a picture unless `--again` is given, so edited pictures are kept. The pictures are made from your copy of the game and stay on your machine.

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
| `--view expanded` or `classic` | Selects Expanded ultrawide view or classic 4:3 view for this start, instead of the one last chosen with F9. |
| `--window WxH` | Sets the initial window client size in pixels (for example `2560x1080`). |
| `--data <folder>` | Uses the game files in another folder. |
| `--no-redraw` | Does not redraw new sprites in the background. |

## Where files go

The original game files are never changed (even read-only files from CD-ROMs or archives work through copy-on-write). Save games and every other file the game writes go to `%LOCALAPPDATA%\Nax Game Restoration\utopia\files`, and the game reads its own saves from there first. Each write goes to a non-inheritable temporary file that replaces the old version atomically only when the game closes the file normally, so an interrupted save or background remastering run keeps the previous version intact. Screenshots go to the `shots` folder, the saved and upscaled sprites to the `hd` folder, and a runtime log to `runtime.log` in the same place.

### Standalone local folder layout

When you move `Utopia.exe` outside the build tree on your own machine, it automatically looks for the original game files (`u.mdg`, `i.mdg`, `scenario\`, and the other DOS files) in the same folder as `Utopia.exe`, then in a `game\` subfolder next to `Utopia.exe`, then in `..\gamedata\utopia` and `..\..\gamedata\utopia` (or at the path given with `--data <folder>`). If you also place a local `hd\upscaled` and `hd\original` folder next to `Utopia.exe` (or inside `game\hd\`), the runtime atomically copies any missing PNGs into `%LOCALAPPDATA%\Nax Game Restoration\utopia\hd\` on startup without overwriting newer local files.

## Layout

| Folder | Content |
|---|---|
| `tools/recomp` | The translator: the MZ and EXEPACK loader, code discovery and the C++ generator, written in Python. |
| `tools/ai` | The converter that turns a Real-ESRGAN checkpoint into the AI filter's weight file without PyTorch, a CPU reference of the network for checking the GPU version, the ESRGAN network for large models, the tool that upscales the saved sprites, and the tool that redraws them with an image editing model. |
| `models` | The AI filter's weights and their license. The build copies them next to the program. |
| `runtime` | The native runtime: CPU helpers, DOS, BIOS, mouse and VGA services, the OPL2 synthesiser, sound output, the sprite tracker, Expanded View geometry, and the Direct3D 12 window. |
| `games/<game>` | What each game needs: the replacement for its launcher, the seed lists of entry points that discovery cannot prove, and the hooks that report the game's sprite drawing. |
| `tests` | Python tests for the translator and AI tools, plus C++ tests for the runtime. |
| `gamedata/<game>` | Your copy of the game. Git ignores this folder. |
| `generated`, `build`, `out` | The translated code, compiled program, and any local packaging output derived from your copy. Git ignores these folders. |

## Third-party components

The AI filter's weights come from the Real-ESRGAN project and are under the BSD 3-Clause License. See [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).

## License

This software is under the Nax Source-Available License 1.0. You may install and use the official release builds, privately and at work, and you may read the source code. Modifying the software, redistributing it and selling it need the written permission of the owner. See [LICENSE](LICENSE).
