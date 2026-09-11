import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.config import P
from utils.audio import load_clip, mel_spectrogram
from utils.boxes import Target, to_pixel_xyxy, to_unit_cxcywh

MIN_EVENT_FRAMES = 2
FLOOR_QUANTILE = 0.25
FLOOR_CHUNK = 256


@dataclass(frozen=True)
# Sobre el audio, recalculando el mel
class AugmentConfig:
    p_gain: float = 0.8
    gain_db: tuple[float, float] = (-6.0, 6.0)

    p_background: float = 0.5  # mezcla una ventana vacía a este SNR
    snr_db: tuple[float, float] = (6.0, 24.0)

    p_shift: float = 0.5  # corrimiento temporal, en fracción del clip
    max_shift: float = 0.10
    min_overlap: float = 0.5

    p_paste: float = 0.5  # pega llamadas de otras ventanas (copy-paste)
    max_events: int = 3
    max_event_area: float = 0.25
    paste_jitter_db: tuple[float, float] = (-3.0, 3.0)
    paste_max_iou: float = 0.3


def band_floor(mel: Tensor) -> Tensor:
    return mel.quantile(FLOOR_QUANTILE, dim=-1, keepdim=True)


# Recortes de llamadas pegables, muestreados con peso inverso a su clase
class EventBank:
    def __init__(
        self, images: Tensor, boxes: list[Tensor], labels: list[Tensor], max_area: float
    ) -> None:
        n_mels, n_frames = images.shape[-2:]
        window_of_box = torch.repeat_interleave(
            torch.arange(len(boxes)), torch.tensor([len(window) for window in boxes])
        )
        every_box = torch.cat(boxes)

        corners = to_pixel_xyxy(every_box)
        low = corners[:, :2].floor().clamp_min(0).to(torch.int64)
        high = torch.minimum(
            corners[:, 2:].ceil().to(torch.int64), torch.tensor([n_frames, n_mels])
        )
        duration, bandwidth = (high - low).unbind(1)

        by_band = torch.cat([band_floor(chunk) for chunk in images.split(FLOOR_CHUNK)])[:, 0, :, 0]
        cumulative = torch.cat([by_band.new_zeros(len(by_band), 1), by_band.cumsum(1)], dim=1)
        spanned = cumulative[window_of_box, high[:, 1]] - cumulative[window_of_box, low[:, 1]]
        floors = spanned / bandwidth.clamp_min(1)

        pasteable = (
            (duration >= MIN_EVENT_FRAMES)
            & (duration < n_frames)
            & (bandwidth > 0)
            & (floors > 0)
            & (every_box[:, 2] * every_box[:, 3] <= max_area)
        )

        window = window_of_box[pasteable].tolist()
        starts = low[pasteable].tolist()
        ends = high[pasteable].tolist()
        self.patches = [
            images[index, 0, f0:f1, t0:t1]
            for index, (t0, f0), (t1, f1) in zip(window, starts, ends, strict=True)
        ]
        self.rows = [f0 for _, f0 in starts]
        self.floors = floors[pasteable]
        self.labels = torch.cat(labels)[pasteable]
        weights = 1 / self.labels.bincount()[self.labels]
        self.probs = weights / weights.sum()


def shift_boxes(
    boxes: Tensor, labels: Tensor, delta: float, min_overlap: float
) -> tuple[Tensor, Tensor]:
    widths = boxes[:, 2:3]
    starts = boxes[:, :1] - widths / 2 + delta
    low, high = starts.clamp(0.0, 1.0), (starts + widths).clamp(0.0, 1.0)

    keep = ((high - low) >= min_overlap * widths).flatten()
    moved = torch.cat([(low + high) / 2, boxes[:, 1:2], high - low, boxes[:, 3:4]], dim=1)
    return moved[keep], labels[keep]


def copy_paste(
    mel: Tensor,
    floor: Tensor,
    boxes: Tensor,
    labels: Tensor,
    bank: EventBank,
    config: AugmentConfig,
) -> tuple[Tensor, Tensor]:
    ceiling = mel.max()
    n_events = random.randint(1, config.max_events)
    for index in torch.multinomial(bank.probs, n_events, replacement=True).tolist():
        patch = bank.patches[index]
        height, width = patch.shape
        row = bank.rows[index]
        column = random.randrange(mel.shape[-1] - width + 1)
        pasted = to_unit_cxcywh(boxes.new_tensor([[column, row, column + width, row + height]]))

        corners = box_convert(torch.cat([pasted, boxes]), "cxcywh", "xyxy")
        if (box_iou(corners[:1], corners[1:]) > config.paste_max_iou).any():
            continue

        jitter = 10 ** (random.uniform(*config.paste_jitter_db) / 10)
        matched = floor[0, row : row + height].mean() / bank.floors[index]
        gain = torch.minimum(matched * jitter, ceiling / patch.max())
        mel[..., row : row + height, column : column + width] += patch * gain
        boxes = torch.cat([boxes, pasted])
        labels = torch.cat([labels, bank.labels[index].reshape(1)])
    return boxes, labels


class Augmenter:
    def __init__(
        self,
        images: Tensor,
        boxes: list[Tensor],
        labels: list[Tensor],
        clips: list[tuple[str, float]],
        config: AugmentConfig,
    ) -> None:
        self.boxes, self.labels, self.clips, self.config = boxes, labels, clips, config
        self.bank = EventBank(images, boxes, labels, config.max_event_area)
        self.to_mel = mel_spectrogram()
        self.background_clips = [
            clip for clip, window in zip(clips, boxes, strict=True) if not len(window)
        ]

    def __call__(self, index: int) -> tuple[Tensor, Target]:
        config = self.config
        boxes, labels = self.boxes[index], self.labels[index]
        wave = load_clip(*self.clips[index])

        if random.random() < config.p_gain:
            amplitude = 10 ** (random.uniform(*config.gain_db) / 20)
            wave = wave * amplitude
        if self.background_clips and random.random() < config.p_background:
            background = load_clip(*random.choice(self.background_clips))
            signal_power = wave.square().mean()
            background_power = background.square().mean().clamp_min(P.eps)
            snr = 10 ** (random.uniform(*config.snr_db) / 10)
            wave = wave + background * (signal_power / background_power / snr).sqrt()

        mel = self.to_mel(wave)[None]
        n_frames = mel.shape[-1]
        floor = band_floor(mel)

        if random.random() < config.p_shift:
            span = int(config.max_shift * n_frames)
            frames = random.randint(-span, span)
            edge = floor.expand(-1, -1, span)
            padded = torch.cat([edge, mel, edge], dim=-1)
            mel = padded[..., span - frames : span - frames + n_frames]
            boxes, labels = shift_boxes(boxes, labels, frames / n_frames, config.min_overlap)
        if random.random() < config.p_paste:
            boxes, labels = copy_paste(mel, floor, boxes, labels, self.bank, config)

        return mel, {"boxes": boxes, "labels": labels}
