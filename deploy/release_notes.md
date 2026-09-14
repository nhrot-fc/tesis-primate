Primate vocalization detector for Windows. Portable: extract and run, nothing to install.

### What's new

- One program, `Detector.exe`, replaces the `.bat` files. Two views: **Spectrogram** (one
  recording: detect, review, listen, save a Raven table) and **Batch** (a whole folder: one
  `.detections.txt` per recording, written as each file finishes, only boxes above the chosen
  score). `detect.exe` runs the batch from a console.
- Model chosen once, in the toolbar.
- Every box shows its label and score; green = annotations, blue = detections. Loading an
  audio clears the previous tables; nothing is loaded automatically.
- Loading screen while the program starts; the detection engine loads in the background.
- Interface in English.
- Models: the same three as v0.1.0 (Faster R-CNN, AST-Deformable DETR, YOLO26), repackaged
  under this version. If you already have them in `models\`, keep them.

### Install

1. Extract **one** `win64` zip to a short path, e.g. `C:\detector\`. `cpu` runs anywhere;
   `cuda` uses an NVIDIA GPU and falls back to CPU without one.
2. Unzip each `model` zip (at least one) into that same folder, next to `Detector.exe` —
   not into `models\`: the zip already brings `models\<name>\`.
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
