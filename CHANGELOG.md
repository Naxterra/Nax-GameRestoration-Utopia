# Changelog

All notable changes to Nax Game Restoration are documented in this file.

## v0.1.0 (2026-11-25)

- Added ahead-of-time x86-to-C++ translation (`tools/recomp`) and native Windows 11 runtime (`runtime`) for Utopia: The Creation of a Nation (1992), using Direct3D 12 presentation, Win32 input, XAudio2 OPL2/beeper audio, and atomic copy-on-write save handling that never modifies original game files.
- Added Expanded View (`F9`, `--view expanded|classic`, `--window WxH`) rendering Utopia's full 75x100 isometric colony map natively across widescreen and ultrawide monitors with mouse-wheel zoom, smooth panning via WASD, arrow keys, middle-drag, or right-drag, click-anywhere tile interaction, and Utopia's live 10x10 software cursor.
- Added clean floating ultrawide HUD docking with a compact dark-glass top status pill, diamond-clipped compass, tightly cropped command sidebar and scroll-arrow pad, auto-hiding empty bottom message console that expands into a compact dark-glass toast when Utopia prints a message, sleek dark-gunmetal floating popup dialogs, and L-shaped advisor/intelligence popup support.
- Added four Direct3D 12 display filters (`F11`): sharp pixels, smooth edges, CRT scanlines, and AI upscaling with smooth two-layer alpha compositing, zero-flicker double-buffered presentation marks, ambient nighttime dimming, and immediate HD takeover for sprites and full-screen backdrops.
- Added the local AI remastering pipeline (`tools/ai/remaster.py` and `tools/ai/terrain.py`) supporting biome-aware ground tiles, multi-tile building groups, natural mountain and chasm prompts with pitch-black void preservation, seamless emerald toxic-lake shoreline compositing, deterministic 8x compass-marker rendering, and animation frame stabilization helpers, plus automatic portable `hd\` pack seeding and AI upscaling by default on first launch.
- Hardened windowed mouse capture, file I/O, and window controls so windowed mode captures only on click inside the game picture and releases cleanly on `Right Ctrl`, middle click, `Alt+Tab`, `Alt+Space`, `Alt+F4`, focus loss, and window moves, while DOS file handles stay non-inheritable across background remastering, read-only original files work with copy-on-write, failed rooted `CHDIR` preserves `cwd`, and `FindFirst`/`FindNext` reports exact file sizes in subdirectories.
