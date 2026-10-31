# Must never happen

This list is a draft and is waiting for the owner's approval.

- The tools must never modify, move or delete the original game files that the user supplies.
- The repository and any release must never contain game code, graphics, sound, text or data, and must never contain translated output derived from a game.
- The translator must never guess a code address. When a target cannot be proven, it must report the address as unresolved instead of decoding data as code.
- The runtime must never jump to code that was not translated without stopping with a clear message that names the address.
- The runtime must never leave a half-written save game or settings file. Every write goes to a temporary file that replaces the old file only when it is complete.
- The runtime must never read or write files outside the game folder and its own settings and save folder.
- Uninstalling must never leave files, registry entries or other system changes behind, apart from save games the user chose to keep.
