Primate vocalization detector for Windows. Portable: unzip and run, nothing to install.

### Changes in {version}

- **Older NVIDIA cards work.** The `cuda` package now runs from the GTX 900 and GTX 10 series
  (such as the GTX 1050) up to the RTX 40 series. RTX 50 cards use the new `cuda13` package.
- **No more batches where every file fails.** If the graphics card cannot run the model, the
  program uses the CPU instead and says so in the status bar and in `viewer.log`.
- Fixed: after one recording failed, YOLO failed every following one with
  `model='yolo26s.yaml' is not a supported model format`.
- Fixed: Faster R-CNN needed an internet connection the first time it loaded.
- Graphics cards with 4 GB process smaller batches, so Faster R-CNN fits in their memory.
- Clear messages when a model is missing, damaged or made for another version.

### Install

Unzip **one** `win64` zip into a short path (`C:\detector\`), unzip a model zip into that
same folder, run `viewer.exe`. The first time Windows warns: *More info → Run anyway*.
Updating: copy `models\` (and `hf\`, if there is one) from the old folder into the new one.

Windows 10/11 64-bit, 8 GB of RAM. `cuda` needs an NVIDIA card with 4 GB+ from the GTX 900
to the RTX 40 series (driver 560+); `cuda13` is for the RTX 50 series (driver 580+). Both fall
back to the CPU without a supported card. No internet needed.

### Files
