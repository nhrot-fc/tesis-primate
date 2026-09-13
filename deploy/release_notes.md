Primate vocalization detector for Windows. Portable: extract and run, nothing to install.

### What's new

One program, `Detector.exe`, with two views: **Spectrogram** (one recording: detect, review,
listen, save a Raven table) and **Batch** (a whole folder: one `.detections.txt` per
recording). `detect.exe` runs the batch from a console. Models: Faster R-CNN,
AST-Deformable DETR, YOLO26.

### Install

1. Extract **one** `win64` zip to a short path, e.g. `C:\detector\`. `cpu` runs anywhere;
   `cuda` uses an NVIDIA GPU and falls back to CPU without one.
2. Drag each `model` zip onto `Detector.exe` (at least one).
3. Run `Detector.exe`. Windows may warn about an unsigned program the first time: choose
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
