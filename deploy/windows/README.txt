Primate Vocalization Detector - {version} (Windows 64-bit, {variant})
======================================================================

Everything is in this folder: nothing to install. Keep it in a short path
(for example C:\detector\), add at least one model and double-click viewer.exe.

  viewer.exe     The program. Two views, switched from the toolbar:

                 Spectrogram  one recording. Drag an audio file (WAV, FLAC,
                              MP3) onto the window, press Detect, then Review
                              the detections one by one (A accepts, R rejects,
                              Esc leaves) and save the table from File.
                              File > Open annotations opens an existing Raven
                              table over the spectrogram.

                 Batch        a whole folder. Drag it onto the window and press
                              Run: every recording gets a <name>.detections.txt
                              next to it, ready for Raven Pro (File > Open
                              Selection Table). Double-click a row to open it.

  detect.exe     The same as Batch from the command line:
                   detect.exe --model models\frcnn D:\recordings

  Manual.pdf     The user guide, also under Help > User manual. F1 lists the
                 keyboard shortcuts.

  models\        One subfolder per model. To add one, unzip its zip
                 (detector-*-model-*.zip) into THIS folder, next to viewer.exe.

  viewer.log     Appears here if the program fails, and says why.

Requirements: Windows 10 or 11, 64-bit, 8 GB of RAM. The "cuda" package uses an
NVIDIA graphics card (4 GB+, driver 570 or newer) and works on the CPU without
one. No internet connection is needed.

The first time: Windows says "Windows protected your PC" - click "More info",
then "Run anyway". The first launch is slow while the antivirus checks the
new files; later ones are not.

More: https://github.com/nhrot-fc/tesis-primate
