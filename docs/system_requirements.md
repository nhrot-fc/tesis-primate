# Hardware requirements

For running the detector on your own PC (the viewer or batch processing). The program is a
self-contained folder: nothing else to install.

## Summary

| | Minimum (CPU only) | Recommended (GPU) |
|---|---|---|
| Processor | 4 cores, x86-64 | 6+ cores, x86-64 |
| RAM | 8 GB (6 GB free while processing) | 16 GB |
| Graphics card | not needed | NVIDIA with **4 GB+ of VRAM** (GTX 1650 / RTX 3050 or better) and an up-to-date driver¹ |
| Free disk | 3 GB (CPU package) | 7 GB (GPU package) |

¹ Only the NVIDIA driver: the package brings its own CUDA libraries.

## Processing time per model

Measured on a CPU with 8 threads over a 108 s recording at 44.1 kHz. On an ordinary laptop
expect 1.5 to 2 times longer; with a GPU all three models process an hour of audio in under
a minute.

| Model | Size | Speed on CPU | RAM used | 1 h of audio on CPU |
|---|---|---|---|---|
| Faster R-CNN (most accurate) | 166 MB | 1.4× real time | 4.8 GB | ~45 min |
| AST-Deformable DETR | 340 MB (+ 380 MB backbone) | 11× real time | 1.5 GB | ~6 min |
| YOLO26s (fastest) | 39 MB | 42× real time | 1.4 GB | ~1.5 min |

Size reference: a mono 44.1 kHz 16-bit WAV takes **~5.3 MB per minute**, so 250 MB is about
47 minutes of recording (24 in stereo).

## Disk per package

| Package | Unzipped | For whom |
|---|---|---|
| CPU | ~2 GB | any 64-bit PC |
| GPU (CUDA) | ~6 GB | a PC with an NVIDIA card; also works without one, on the CPU |

Both include the interpreter and the libraries; models come as separate zips. Unzip into a
short path (for example `C:\detector\`): long Windows paths cause errors when unzipping.
