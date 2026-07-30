from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torchvision.ops import batched_nms

from architectures.deformable_detr import postprocess
from core.config import P, Parameters
from domain.species import LabelSet
from utils.audio import LogMelSpectrogram, load_clip, window_starts, y_to_hz

RAVEN_COLUMNS = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "Call type",
    "Score",
]


@torch.no_grad()
def predict(
    model: nn.Module,
    audio_path: str | Path,
    labels: LabelSet,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.5,
    nms_iou: float = 0.3,
    batch_size: int = 16,
    params: Parameters = P,
) -> pd.DataFrame:
    """Desliza ventanas de `params.clip_len_s` sobre el audio, corre el modelo en cada
    una y funde las detecciones superpuestas con NMS por clase (en tiempo/frecuencia
    absolutos, donde IoU es igual de válido que en coordenadas normalizadas)."""
    model.eval()
    starts = window_starts(sf.info(str(audio_path)).duration, params)
    mel = LogMelSpectrogram(params)

    begin, end, low, high, score, label = [], [], [], [], [], []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        images = torch.stack(
            [mel(load_clip(str(audio_path), float(t), params)) for t in chunk]
        ).unsqueeze(1)
        detections = postprocess(model(images.to(device)), score_threshold)

        for clip_start, det in zip(chunk, detections, strict=True):
            cx, cy, w, h = det.boxes.T.cpu()
            begin.append(clip_start + (cx - w / 2) * params.clip_len_s)
            end.append(clip_start + (cx + w / 2) * params.clip_len_s)
            low.append(torch.from_numpy(y_to_hz((cy - h / 2).numpy(), params)).float())
            high.append(torch.from_numpy(y_to_hz((cy + h / 2).numpy(), params)).float())
            score.append(det.scores.cpu())
            label.append(det.labels.cpu())

    begin, end, low, high = (torch.cat(t) for t in (begin, end, low, high))
    score, label = torch.cat(score), torch.cat(label)

    boxes = torch.stack([begin, low, end, high], dim=-1)  # ya en formato xyxy (tiempo, freq)
    keep = batched_nms(boxes, score, label, nms_iou)
    order = keep[begin[keep].argsort()]

    names = [labels.name(class_id).split("/") for class_id in label[order].tolist()]
    return pd.DataFrame(
        {
            "Selection": range(1, len(order) + 1),
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin Time (s)": begin[order].numpy(),
            "End Time (s)": end[order].numpy(),
            "Low Freq (Hz)": low[order].numpy(),
            "High Freq (Hz)": high[order].numpy(),
            "Species": [species.upper() for species, _ in names],
            "Call type": [call_type.upper() for _, call_type in names],
            "Score": score[order].numpy(),
        },
        columns=RAVEN_COLUMNS,
    )
