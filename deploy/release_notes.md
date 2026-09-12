Primate vocalization detector for Windows. Portable: extract and run, nothing to install.

### What's new

First release: spectrogram viewer with detection and review, batch detection over whole
folders, output as Raven selection tables. Models: Faster R-CNN, AST-Deformable DETR, YOLO26.

### Install

1. Extract **one** `win64` zip to a short path, e.g. `C:\detector\`. `cpu` runs anywhere;
   `cuda` uses an NVIDIA GPU and falls back to CPU without one.
2. Drag each `model` zip onto `Agregar-modelo.bat` in that folder (at least one).
3. Run `Visor.bat` (viewer) or drag a folder onto `Detectar.bat` (batch).

A zip in parts (`.zip.001`, `.zip.002`): download all parts and its `unir.bat` into one folder
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
