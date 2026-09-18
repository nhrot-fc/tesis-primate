"""Predicciones frente a anotaciones: el volcado de una corrida, el score que un modelo da a cada
clase sobre una caja anotada, y qué checkpoint no vio cada grabación de train (k-fold)."""

import functools
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.config import MAX_DETECTIONS, RUNS_DIR, SCORE_FLOOR, P
from data import cache
from evaluation.evaluator import RawPredictions, path_for
from evaluation.metrics import MATCH_IOU
from evaluation.protocol import equalize
from models.registry import LoadedModel, load_checkpoint
from utils.audio import hz_to_y, load_clip, mel_spectrogram
from utils.boxes import Detections, postprocess


def load_dump(run: str, split: str, names: list[str] | None = None) -> RawPredictions:
    # El volcado de `runs/<run>/` sobre `split`, con las predicciones igualadas como en la
    # comparación (`equalize`) y las clases contrastadas con las del caché
    dump = RawPredictions.load(path_for(RUNS_DIR / run, run, split))
    if names is not None and dump.labels != names:
        raise ValueError(f"{run}/{split}: las clases del volcado no son las del caché")
    return dump._replace(predictions=equalize(dump.predictions))


def class_scores(
    detections: Detections, targets: Tensor, n_classes: int, iou: float = MATCH_IOU
) -> Tensor:
    # (T, C): para cada caja `targets` (cxcywh), el mayor score de cada clase entre las
    # detecciones que la solapan con IoU >= `iou`; 0 donde ninguna
    scores = torch.zeros(len(targets), n_classes)
    if not len(detections.boxes) or not len(targets):
        return scores
    overlap = box_iou(
        box_convert(targets.float(), "cxcywh", "xyxy"),
        box_convert(detections.boxes.float(), "cxcywh", "xyxy"),
    )
    overlapping = torch.where(overlap >= iou, detections.scores.float()[None, :], 0.0)
    labels = detections.labels.long()[None, :].expand_as(overlapping)
    return scores.scatter_reduce(1, labels, overlapping, reduce="amax")


def centered_window(begin_s: float, end_s: float) -> float:
    # Inicio de la ventana de un clip centrada en la anotación, sin salirse del principio del audio
    return max(0.0, (begin_s + end_s) / 2 - P.clip_len_s / 2)


def window_box(
    begin_s: float, end_s: float, low_hz: float, high_hz: float, start_s: float
) -> Tensor:
    # La anotación como caja (1, 4) cxcywh en la ventana que empieza en `start_s`
    x0, x1 = (begin_s - start_s) / P.clip_len_s, (end_s - start_s) / P.clip_len_s
    y0, y1 = hz_to_y(np.array([low_hz, high_hz], dtype=float), P).clip(0, 1)
    return box_convert(torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32), "xyxy", "cxcywh")


def fold_checkpoints(kfold: str) -> dict[str, Path]:
    # Grabación de train -> `best.pt` del pliegue que la dejó fuera. Vacío si el k-fold no es de
    # este caché (otro número de ventanas)
    folds = json.loads((RUNS_DIR / kfold / "folds.json").read_text())["fold_of_window"]
    sources = cache.Sources(**json.loads(cache.sources_path(cache.TRAIN).read_text()))
    if len(folds) != len(sources.recording_of_window):
        return {}
    checkpoints: dict[str, Path] = {}
    for window, recording in enumerate(sources.recording_of_window):
        stem = Path(sources.recordings[recording]).stem
        checkpoints.setdefault(stem, RUNS_DIR / kfold / f"fold{folds[window]}" / "best.pt")
    return checkpoints


@functools.cache
def load_model(checkpoint: Path, device: str) -> LoadedModel:
    return load_checkpoint(checkpoint, device)


@functools.cache
def to_mel() -> torch.nn.Module:
    return mel_spectrogram()


def detect_clip(loaded: LoadedModel, audio: str | Path, start_s: float, device: str) -> Detections:
    # Las detecciones sobre la ventana que empieza en `start_s`, posprocesadas como en la evaluación
    mel = to_mel()(load_clip(audio, start_s)).unsqueeze(0).unsqueeze(0)
    detections = loaded.model.detect(mel.to(device), SCORE_FLOOR)[0]
    return postprocess(detections, loaded.model.nms_iou, MAX_DETECTIONS)
