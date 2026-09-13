Primate Vocalization Detector - {version} (Windows 64-bit, {variant})
======================================================================

Everything is in this folder: no Python or libraries to install. Unzip it to a
short path (for example C:\detector\), add at least one model and run:

  Detector.exe   The program. Two views, switched at the top left:

                 Spectrogram  one recording at a time. Drag and drop an audio
                              (WAV, FLAC, MP3) or press Ctrl+O; choose a model in
                              the toolbar and press Detect. Open a Raven table
                              over it, filter by score, listen, remove boxes,
                              save the table or an image. F1 lists the controls.

                 Batch        a whole folder. Drag and drop the folder (or click
                              Browse), set the score, press Run: every recording
                              gets a <name>.detections.txt next to it, ready for
                              Raven Pro (File > Open Selection Table) or the
                              Spectrogram view (double-click the row).

  detect.exe     The same batch detection from a console, for scripts:
                   detect.exe --model models\frcnn D:\recordings

  Models         They come as separate zips (detector-*-model-*.zip). Drag the
                 zip onto Detector.exe or onto its window, or use "Add model from
                 zip..." in the Model list. Each model is a subfolder of models\.

  viewer.log     Appears next to this file if the viewer fails, and says why.

Requirements: 64-bit Windows 10 or 11. The "cuda" package uses an NVIDIA GPU
with driver 570 or newer, and falls back to the CPU without one. Hardware and
speed per model: docs\system_requirements.md at
https://github.com/nhrot-fc/tesis-primate

The first launch is slow (Windows scans the new files); later ones are not.
Windows may warn about an unsigned program the first time: choose "More info"
and "Run anyway". No internet connection is needed or used.
