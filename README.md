# Primate vocalization detector

Finds primate calls in audio recordings. A neural network draws a box around each call on
the spectrogram and saves them as a **Raven selection table**, ready to open in Raven Pro or
in the viewer that comes with the program.

Windows only. Portable: unzip and run. **Nothing to install** (no Python, no CUDA).

## Download

From [Releases](https://github.com/nhrot-fc/tesis-primate/releases):

| Zip | What it is | Size |
|---|---|---|
| `detector-<version>-win64-cpu.zip` | the program, for any 64-bit PC | ~0.5 GB |
| `detector-<version>-win64-cuda.zip` | the program with NVIDIA GPU support (uses the CPU if there is no GPU) | ~3 GB |
| `detector-<version>-model-<name>.zip` | a model; you need at least one | 25 MB – 0.7 GB |

1. Unzip **one** `win64` zip into a short path, for example `C:\detector\`.
2. Unzip each `model` zip into **that same folder**, next to `viewer.exe`.
3. Double-click `viewer.exe`.

If a zip comes in parts (`.zip.001`, `.zip.002`, …), download all the parts and the
`join.bat` into one folder and run `join.bat`.

## Use

**Watch the demo: [resources/demo.mp4](resources/demo.mp4)** (70 seconds, one recording
from start to finish). The user guide with screenshots is
[docs/manual/build/manual.pdf](docs/manual/build/manual.pdf); it is also inside the
program folder as `Manual.pdf` and under *Help → User manual*. `F1` lists every control.

`viewer.exe` has two views, switched from the toolbar:

- **Spectrogram** — one recording. Drag an audio file (WAV, FLAC, MP3) onto the window, press
  **Detect**, then **Review** the detections one by one (`A` accepts, `R` rejects, `Esc`
  leaves) and *File → Save annotations table*. *File → Open annotations* (`Ctrl+T`) opens an
  existing Raven table over the spectrogram.
- **Batch** — a whole folder. Drag it onto the window, press **Run**: every recording gets a
  `<name>.detections.txt` next to it. Double-click a row to open it in the Spectrogram view.

`detect.exe` does the same as Batch from the command line:

```
detect.exe --model models\frcnn D:\recordings
```

In Raven Pro: open the audio, then *File → Open Selection Table* with its `.detections.txt`.
The `Score` column lets you filter inside Raven.

## Requirements

| | CPU | GPU |
|---|---|---|
| Windows | 10 or 11, 64-bit | same |
| RAM | 8 GB | 16 GB |
| Graphics card | not needed | NVIDIA with 4 GB+, driver 570 or newer |
| Free disk | 3 GB | 7 GB |

No admin rights and no internet connection. Processing speed per model:
[docs/system_requirements.md](docs/system_requirements.md).

## If something goes wrong

- *Windows protected your PC* the first time: **More info → Run anyway**. It happens once.
- The first launch is slow: the antivirus checks the new files.
- The program does not open: read `viewer.log`, next to `viewer.exe`.
- Errors while unzipping: use a shorter path, such as `C:\detector\`.

## Building a release

`deploy/release.sh v0.1.0 runs/frcnn` builds the zips and uploads them to GitHub (its header
explains the options). `make manual` compiles the guide, `make screenshots` retakes its
figures and `make demo` records the video.
