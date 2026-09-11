from collections.abc import Callable
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

from core.config import MAX_DETECTIONS, NMS_IOU, SCORE_THRESHOLD, P, Parameters
from data.raven import raven_table
from data.species import split_label
from models.registry import LoadedModel
from utils.audio import load_clips, mel_spectrogram, window_starts, y_to_hz
from utils.boxes import postprocess, suppress_nested

BATCH_SIZE = 16


def species_and_call(name: str) -> tuple[str, str]:
    # Raven escribe la especie y la llamada en mayúscula y por separado.
    species, call_type = split_label(name)
    return species.upper(), call_type.upper()


@torch.no_grad()
def predict(
    loaded: LoadedModel,
    audio_path: str | Path,
    device: str | torch.device = "cpu",
    score_threshold: float = SCORE_THRESHOLD,
    merge_iou: float = NMS_IOU,
    batch_size: int = BATCH_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
    params: Parameters = P,
) -> pd.DataFrame:
    # Cada ventana se posprocesa como en la evaluación (el NMS propio del modelo y el tope de
    # detecciones); después se funden con `merge_iou` los duplicados entre ventanas solapadas,
    # que es lo único que la evaluación por ventana no tiene.
    model, labels = loaded.model, loaded.labels
    model.eval()
    duration_s = sf.info(str(audio_path)).duration
    starts = window_starts(duration_s, params)
    mel = mel_spectrogram(params)
    if on_progress is not None:
        on_progress(0, len(starts))

    x0, x1, y0, y1, score, label = [], [], [], [], [], []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        images = torch.stack(
            [mel(clip) for clip in load_clips(audio_path, chunk, params)]
        ).unsqueeze(1)
        detections = model.detect(images.to(device), score_threshold)

        for clip_start, det in zip(chunk, detections, strict=True):
            det = postprocess(det, model.nms_iou, MAX_DETECTIONS)
            cx, cy, w, h = det.boxes.T.cpu()
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

    keep = suppress_nested(torch.stack([x0, y0, x1, y1], dim=-1), score, label, merge_iou)
    order = keep[x0[keep].argsort()]

    begin = (x0[order] * params.clip_len_s).clamp(0.0, duration_s)
    end = (x1[order] * params.clip_len_s).clamp(0.0, duration_s)
    has_duration = end > begin
    order, begin, end = order[has_duration], begin[has_duration], end[has_duration]

    low = torch.from_numpy(y_to_hz(y0[order].numpy(), params)).float()
    high = torch.from_numpy(y_to_hz(y1[order].numpy(), params)).float()

    names = [species_and_call(labels.name(class_id)) for class_id in label[order].tolist()]
    return raven_table(
        begin=begin.numpy(),
        end=end.numpy(),
        low=low.numpy(),
        high=high.numpy(),
        species=[species for species, _ in names],
        call_type=[call_type for _, call_type in names],
        score=score[order].numpy(),
    )
