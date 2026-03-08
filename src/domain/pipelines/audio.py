from typing import Sequence
from src.domain.pipelines.types import AudioArray, AnnotationBox
from src.domain.pipelines.types import AnnotationBox, AudioArray


def slice_audio_window(
    audio: AudioArray,
    annotations: Sequence[AnnotationBox],
    sample_rate: int,
    start_time_sec: float,
    window_duration_sec: float,
) -> tuple[AudioArray, list[AnnotationBox]]:
    """
    Corta un segmento de audio y ajusta los tiempos de las anotaciones para
    que sean relativas a esta nueva ventana.

    Descarta las anotaciones que caen fuera de la ventana e intercepta
    aquellas que son cortadas por los bordes.

    Retorno:
        El segmento de audio y las anotaciones ajustadas al "tiempo cero" del recorte.
    """
    raise NotImplementedError("slice_audio_window is not implemented yet")


def apply_time_stretch(
    audio: AudioArray, annotations: Sequence[AnnotationBox], stretch_factor: float
) -> tuple[AudioArray, list[AnnotationBox]]:
    raise NotImplementedError("apply_time_stretch is not implemented yet")


def apply_pitch_shift(
    audio: AudioArray,
    annotations: Sequence[AnnotationBox],
    sample_rate: int,
    semitones: float,
) -> tuple[AudioArray, list[AnnotationBox]]:
    raise NotImplementedError("apply_pitch_shift is not implemented yet")
