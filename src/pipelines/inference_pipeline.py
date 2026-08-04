from collections.abc import Callable
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torchvision.ops import batched_nms

from architectures.deformable_detr import postprocess
from core.config import P, Parameters
from domain.raven import RAVEN_COLUMNS
from domain.species import LabelSet
from utils.audio import MelSpectrogram, load_clips, window_starts, y_to_hz


def species_and_call(name: str) -> tuple[str, str]:
    """Etiqueta del `LabelSet` -> columnas Species/Call type de Raven.

    `'lw/cc'` -> `('LW', 'CC')`, pero una etiqueta sin barra (`'other'`, o cualquier
    `LABEL_BY` que no sea `species/call_type`) es válida y sale con el tipo vacío.
    """
    species, _, call_type = name.partition("/")
    return species.upper(), call_type.upper()


@torch.no_grad()
def predict(
    model: nn.Module,
    audio_path: str | Path,
    labels: LabelSet,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.5,
    nms_iou: float = 0.3,
    batch_size: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
    params: Parameters = P,
) -> pd.DataFrame:
    model.eval()
    duration_s = sf.info(str(audio_path)).duration
    starts = window_starts(duration_s, params)
    mel = MelSpectrogram(params)
    if on_progress is not None:
        on_progress(0, len(starts))

    x0, x1, y0, y1, score, label = [], [], [], [], [], []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        images = torch.stack(
            [mel(clip) for clip in load_clips(audio_path, chunk, params)]
        ).unsqueeze(1)
        detections = postprocess(model(images.to(device)), score_threshold)

        for clip_start, det in zip(chunk, detections, strict=True):
            cx, cy, w, h = det.boxes.T.cpu()
            # Tiempo en unidades de clip (no en segundos) y frecuencia en el eje mel
            # normalizado: es el mismo espacio en el que el modelo predice y en el que
            # `evaluation_pipeline` mide IoU, así que el NMS de acá suprime exactamente
            # lo mismo que la evaluación da por suprimido.
            offset = float(clip_start) / params.clip_len_s
            x0.append(offset + cx - w / 2)
            x1.append(offset + cx + w / 2)
            y0.append(cy - h / 2)
            y1.append(cy + h / 2)
            score.append(det.scores.cpu())
            label.append(det.labels.cpu())

        if on_progress is not None:
            on_progress(min(i + batch_size, len(starts)), len(starts))

    x0, x1, y0, y1 = (torch.cat(t) for t in (x0, x1, y0, y1))
    score, label = torch.cat(score), torch.cat(label)

    keep = batched_nms(torch.stack([x0, y0, x1, y1], dim=-1), score, label, nms_iou)
    order = keep[x0[keep].argsort()]

    # La caja de una query puede salirse del clip (y la última ventana se pasa del final
    # del audio): sin recortar, la tabla sale con tiempos negativos o más allá del
    # archivo, que Raven no acepta. Lo que queda sin duración se descarta.
    begin = (x0[order] * params.clip_len_s).clamp(0.0, duration_s)
    end = (x1[order] * params.clip_len_s).clamp(0.0, duration_s)
    inside = end > begin
    order, begin, end = order[inside], begin[inside], end[inside]

    low = torch.from_numpy(y_to_hz(y0[order].numpy(), params)).float()
    high = torch.from_numpy(y_to_hz(y1[order].numpy(), params)).float()

    names = [species_and_call(labels.name(class_id)) for class_id in label[order].tolist()]
    return pd.DataFrame(
        {
            "Selection": range(1, len(order) + 1),
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin Time (s)": begin.numpy(),
            "End Time (s)": end.numpy(),
            "Low Freq (Hz)": low.numpy(),
            "High Freq (Hz)": high.numpy(),
            "Species": [species for species, _ in names],
            "Call type": [call_type for _, call_type in names],
            "Score": score[order].numpy(),
        },
        columns=RAVEN_COLUMNS,
    )
