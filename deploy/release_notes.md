Primate vocalization detector for Windows: portable, nothing to install. Draws
time–frequency boxes on the spectrogram of each recording and writes them as **Raven
selection tables**.

## Install

1. Extract **one** `win64` zip to a short path, e.g. `C:\detector\`.
2. Drag each `model` zip onto `Agregar-modelo.bat` inside that folder (at least one model).
3. `Visor.bat` opens the spectrogram viewer (`Ctrl+O` audio, `Ctrl+M` model, `Ctrl+R` detect).
   `Detectar.bat` processes whole folders: drag a folder onto it and a
   `<audio>.detections.txt` appears next to each recording, ready for Raven Pro
   (*File → Open Selection Table*).

If a zip comes in parts (`.zip.001`, `.zip.002`, …), download them all plus its `unir.bat`
into one folder and run it.

## System requirements

| | Minimum (CPU only) | Recommended (GPU) |
|---|---|---|
| OS | Windows 10 or 11, 64-bit | same |
| CPU | 4 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | none | NVIDIA with 4 GB+ VRAM and driver **570 or newer** (`cuda` zip; falls back to CPU) |
| Free disk | 3 GB | 7 GB |
| To install | nothing: Python, CUDA and the VC++ runtime are inside | nothing but the NVIDIA driver |
| Internet | not needed | not needed |

CPU speed (8 threads, 44.1 kHz audio): Faster R-CNN ~1.4× real time, AST-DETR ~11×, YOLO ~42×.
With a GPU all three process an hour of audio in under a minute. No admin rights needed; the
first launch is slower while the antivirus scans the new files.

## Files
