Primate vocalization detector for Windows. Portable: unzip and run, nothing to install.

### What's new

- `viewer.exe` replaces `Detector.exe`: a menu bar with shortcuts and two views, **Spectrogram**
  (one recording) and **Batch** (a whole folder). `detect.exe` is still the console version.
- **Batch** counts the detections of every recording, per species and call if you want, and
  exports them as CSV. Moving *Score* only recounts: the model does not run again.
- **Settings** (brightness, contrast, volume, audio output) and `Manual.pdf`, the user guide
  (also under *Help → User manual*). There is also a
  [demo video](https://github.com/nhrot-fc/tesis-primate/blob/main/resources/demo.mp4).

### Fixes in {version}

- Fixed: when a recording failed to open, its detections showed up on the next one opened.
- Fixed: in Review, after editing a label and pressing Enter, the next keys were typed into
  the label.
- Fixed: the box under review could be dragged outside the recording.
- Fixed: annotation tables saved after Review lacked the `View` and `Channel` columns of a
  Raven table.
- Fixed: Raven tables saved with the waveform view open showed every selection twice.
- Fixed: error in Batch when ticking *Count per species and call* before any recording had a
  table.
- Fixed: choosing another folder while Batch was still listing the first one kept the first
  one's list.
- Fixed: Batch listed the `._*.wav` files that macOS leaves on memory cards.
- Tables from Excel in Spanish (`;` and decimal comma) or with accents now open; when a file
  cannot be opened, the message says why.
- Better performance with 1 h recordings (up to 2 h supported).

### Install

Unzip **one** `win64` zip into a short path (`C:\detector\`), unzip a model zip into that
same folder, run `viewer.exe`. The first time Windows warns: *More info → Run anyway*.

Windows 10/11 64-bit, 8 GB of RAM. `cuda` needs an NVIDIA card (4 GB+, driver 570+) and
falls back to the CPU without one. No internet needed.

### Files
