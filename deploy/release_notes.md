Primate vocalization detector for Windows. Portable: unzip and run, nothing to install.

### What's new

- `viewer.exe` replaces `Detector.exe`; `detect.exe` is the console version.
- Two views: **Spectrogram** (one recording) and **Batch** (a whole folder). Menu bar with
  shortcuts; *Open audio*, *Open annotations* and *Open folder* are separate.
- **Settings**: brightness, contrast, volume and audio output device.
- `Manual.pdf`, the user guide with screenshots (also under *Help → User manual*), and a
  [demo video](https://github.com/nhrot-fc/tesis-primate/blob/main/resources/demo.mp4).
- Models: unchanged. Use the model zips from
  [v0.2.0](https://github.com/nhrot-fc/tesis-primate/releases/tag/v0.2.0), or keep your
  `models\` folder.

### Install

Unzip **one** `win64` zip into a short path (`C:\detector\`), unzip a model zip into that
same folder, run `viewer.exe`. The first time Windows warns: *More info → Run anyway*.

Windows 10/11 64-bit, 8 GB of RAM. `cuda` needs an NVIDIA card (4 GB+, driver 570+) and
falls back to the CPU without one. No internet needed.

### Files
