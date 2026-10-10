from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile
import soxr

from viewer.tasks import UnreadableError

# Piso del log del STFT en pantalla. Más bajo que `P.eps` (el del mel del modelo) a propósito:
# el STFT crudo no suma bandas y en una grabación silenciosa un 13 % de los bins queda bajo
# -60 dB; con el piso del modelo se aplastarían.
EPS = 1e-10
# Rango en dB de una grabación para el contraste inicial (`utils.audio.DB_PERCENTILES` es el
# del caché para normalizar la entrada de los modelos)
DISPLAY_PERCENTILES = (5.0, 99.5)
# Muestras por tramo al pasar a PCM: unos 95 s a 44,1 kHz
PCM_BLOCK = 1 << 22

Waveform = npt.NDArray[np.float32]


# Sin torch a propósito: el visor abre audio aunque el motor de detección no cargue. El mismo
# paso (mono, resampleo) lo hace `utils.audio.read_clip` para el modelo, con torchaudio.
def load_audio(path: Path, target_sr: int) -> Waveform:
    if not path.is_file():
        raise UnreadableError(f"'{path.name}' is not in {path.parent} any more.")
    try:
        frames, source_sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        # Mono: la columna ya es la señal, sin la copia del promedio (una hora son 600 MB).
        waveform = frames[:, 0] if frames.shape[1] == 1 else frames.mean(axis=1)
        if source_sr != target_sr:
            waveform = soxr.resample(waveform, source_sr, target_sr)
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    except soundfile.LibsndfileError as exc:
        # libsndfile dice "File does not exist" de un archivo que existe pero está roto.
        raise UnreadableError(
            f"Could not open '{path.name}': it is not a WAV, FLAC or MP3 recording, "
            "or the file is damaged."
        ) from exc
    except MemoryError as exc:
        raise UnreadableError(
            f"Not enough memory to open '{path.name}'. Close other programs, or cut the "
            "recording into shorter files."
        ) from exc
    # Un WAV en coma flotante puede traer NaN o infinitos: dejarían el espectrograma en negro.
    return np.nan_to_num(waveform, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


# Las grabaciones de campo llegan a -45 dBFS RMS: sin llevarlas al pico no se oyen. Se convierte
# por tramos: con una hora de audio, cada copia entera en coma flotante son 600 MB.
def pcm16(waveform: Waveform, gain: float = 1.0) -> bytes:
    peak = max(float(waveform.max(initial=0.0)), -float(waveform.min(initial=0.0)))
    scale = 32767.0 * (gain / peak if peak > 0.0 else 1.0)
    pcm = np.empty(waveform.size, dtype=np.int16)
    for start in range(0, waveform.size, PCM_BLOCK):
        block = waveform[start : start + PCM_BLOCK] * scale
        pcm[start : start + PCM_BLOCK] = np.clip(block, -32767.0, 32767.0, out=block)
    return pcm.tobytes()


def sliding_frames(
    waveform: Waveform, n_fft: int, hop_length: int, pad: bool = True
) -> npt.NDArray[np.float32]:
    padded = np.pad(waveform, n_fft // 2, mode="reflect") if pad else waveform
    count = (padded.size - n_fft) // hop_length + 1
    stride = padded.strides[0]
    return np.lib.stride_tricks.as_strided(
        padded, shape=(count, n_fft), strides=(stride * hop_length, stride)
    )


def stft_db(
    waveform: Waveform, n_fft: int, hop_length: int, pad: bool = True
) -> npt.NDArray[np.float32]:
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))
    # periodica, como torch.hann_window; en float32 la FFT no pasa a doble precisión
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(sliding_frames(waveform, n_fft, hop_length, pad) * window, axis=-1)
    power = spectrum.real**2 + spectrum.imag**2
    return (10.0 * np.log10(power.T + EPS)).astype(np.float32)


def db_baseline(waveform: Waveform, sr: int, n_fft: int) -> tuple[float, float]:
    # Sin el relleno de los bordes: a dos percentiles no les cambia nada, y rellenar copia el
    # audio entero.
    spec = stft_db(waveform, n_fft, max(sr // 10, waveform.size // 3000, 1), pad=False)
    lo, hi = np.percentile(spec, DISPLAY_PERCENTILES)
    return float(lo), float(hi)


def db_levels(
    baseline: tuple[float, float], brightness: float, contrast: float
) -> tuple[float, float]:
    lo, hi = baseline
    center = 0.5 * (lo + hi) - brightness
    half_range = 0.5 * (hi - lo) / max(contrast, 1e-3)
    return center - half_range, center + half_range
