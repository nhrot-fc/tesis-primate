Primate Vocalization Detector - {version} (Windows 64-bit, {variant})
======================================================================

Everything is in this folder: no Python or libraries to install. Unzip it to a
short path (for example C:\detector\), add at least one model and run:

  Detector.exe   The program. One window: the folder's recordings on the left,
                 the spectrogram in the middle, the table of boxes on the right.

                 One recording   drag and drop an audio (WAV, FLAC, MP3) or press
                              Ctrl+O and press Detect (the model is remembered
                              between runs). Open a Raven table over it the same
                              way, filter by score, listen, remove boxes, save the
                              table or an image. Review goes through the
                              detections one by one: A/Enter accepts, R/Del
                              rejects, Esc leaves; it picks up where you left off.
                              F1 lists the controls.

                 A folder     drag and drop it (or Ctrl+Shift+O): its recordings
                              are listed on the left, one click opens each. Detect
                              all runs the model over the whole list and leaves a
                              <name>.detections.txt next to every recording, ready
                              for Raven Pro (File > Open Selection Table).

  detect.exe     The same batch detection from a console, for scripts:
                   detect.exe --model models\frcnn D:\recordings

  Models         They come as separate zips (detector-*-model-*.zip). Unzip each
                 one into THIS folder, the one with Detector.exe (not into models\):
                 the zip already brings models\<name>\, and hf\ when the model
                 needs it. Each model is a subfolder of models\.

  viewer.log     Appears next to this file if the viewer fails, and says why.

Requirements: 64-bit Windows 10 or 11. The "cuda" package uses an NVIDIA GPU
with driver 570 or newer, and falls back to the CPU without one. Hardware and
speed per model: docs\system_requirements.md at
https://github.com/nhrot-fc/tesis-primate

The first launch is slow (Windows scans the new files); later ones are not.
Windows may warn about an unsigned program the first time: choose "More info"
and "Run anyway". No internet connection is needed or used.
