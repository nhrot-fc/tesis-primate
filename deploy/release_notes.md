Primate vocalization detector for Windows. Portable: extract and run, nothing to install.

### What's new

- `viewer.exe` (was `Detector.exe`): the same program, simpler name. `detect.exe` is the
  console version.
- A menu bar (File, View, Detect, Help) with every shortcut next to its action, as in Raven
  and Audacity. Opening is split: **Open audio** (`Ctrl+O`), **Open annotations** (`Ctrl+T`)
  and **Open folder** (`Ctrl+Shift+O`).
- Two views again, switched from the toolbar: **Spectrogram** (one recording) and **Batch**
  (a whole folder, with its own Score, Overwrite and Include subfolders; double-click a row to
  open it in the Spectrogram view).
- **Settings** replaces *Adjust*: brightness, contrast, volume and the **audio output
  device**.
- Zoom from the keyboard with Audacity's shortcuts: `Ctrl+1` / `Ctrl+3` in time,
  `Ctrl+↑` / `Ctrl+↓` in frequency, `F` for the full band.
- `Manual.pdf`, the user guide with screenshots, next to the program and under
  *Help → User manual*.
- Models: the same as before. If you already have them in `models\`, keep them.

### Install

1. Extract **one** `win64` zip to a short path, e.g. `C:\detector\`. `cpu` runs anywhere;
   `cuda` uses an NVIDIA GPU and falls back to CPU without one.
2. Unzip each `model` zip (at least one) into that same folder, next to `viewer.exe` —
   not into `models\`: the zip already brings `models\<name>\`.
3. Run `viewer.exe`. Windows may warn about an unsigned program the first time: choose
   *More info → Run anyway*.

A zip in parts (`.zip.001`, `.zip.002`): download all parts and its `join.bat` into one folder
and run it.

### Requirements

| | CPU | GPU |
|---|---|---|
| Windows | 10 / 11, 64-bit | same |
| RAM | 8 GB | 16 GB |
| GPU | — | NVIDIA, 4 GB+ VRAM, driver 570+ |
| Free disk | 3 GB | 7 GB |

No admin rights, no internet, nothing else to install.

### Files
