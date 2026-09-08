from collections.abc import Callable
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

from core.config import P, Parameters
from models.registry import LoadedModel
from utils.audio import load_clips, mel_spectrogram, window_starts, y_to_hz
from utils.boxes import suppress_nested

# IoU con el que se funden las detecciones repetidas por el solape entre ventanas (3 s cada
# 1.5 s: cada vocalización cae en dos). No es la NMS de `Architecture.nms_iou`, que saca los
# duplicados *dentro* de una ventana y que DETR y DINO no necesitan: ésta corre para todos.
MERGE_IOU = 0.3


def species_and_call(name: str) -> tuple[str, str]:
    species, _, call_type = name.partition("/")
    return species.upper(), call_type.upper()


@torch.no_grad()
def predict(
    loaded: LoadedModel,
    audio_path: str | Path,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.5,
    nms_iou: float = MERGE_IOU,
    batch_size: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
    params: Parameters = P,
) -> pd.DataFrame:
    labels = loaded.labels
    loaded.model.eval()
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
        detections = loaded.detect(images.to(device), score_threshold)

        for clip_start, det in zip(chunk, detections, strict=True):
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

    keep = suppress_nested(torch.stack([x0, y0, x1, y1], dim=-1), score, label, nms_iou)
    order = keep[x0[keep].argsort()]

    begin = (x0[order] * params.clip_len_s).clamp(0.0, duration_s)
    end = (x1[order] * params.clip_len_s).clamp(0.0, duration_s)
    has_duration = end > begin
    order, begin, end = order[has_duration], begin[has_duration], end[has_duration]

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
        }
    )
