# Nax Game Restoration

Nax Game Restoration makes classic DOS games run natively on Windows 11 without an emulator. A translator turns the 16-bit x86 code of a legally owned copy into C++ ahead of time, and a native runtime replaces DOS and the PC hardware with Win32 services, a Direct3D 12 presenter and modern audio.

The repository contains only the translator and the runtime. It contains no game code, graphics, sound or data. Every build reads the original files from a copy that the user owns, and the translated output stays on that user's machine.

## Status

The project is at the analysis stage. Utopia is the first target because its program is small and does not use overlays. Dune II follows once the translator and runtime handle Utopia.

## Layout

The `tools/recomp` folder holds the translator, which is written in Python and uses the Capstone disassembler. Original game files go into `gamedata/<game>/`, which Git ignores. Translated output goes into `generated/`, which Git also ignores.

## License

This software is under the Nax Source-Available License 1.0. You may install and use the official release builds, privately and at work, and you may read the source code. Modifying the software, redistributing it and selling it need the written permission of the owner. See [LICENSE](LICENSE).
