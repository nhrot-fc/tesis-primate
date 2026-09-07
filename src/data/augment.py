import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.config import P
from utils.audio import load_clip, mel_spectrogram
from utils.boxes import Target, to_pixel_xyxy, to_unit_cxcywh

MIN_EVENT_FRAMES = 2


@dataclass(frozen=True)
class AugmentConfig:
    p_gain: float = 0.8
    gain_db: tuple[float, float] = (-6.0, 6.0)

    p_mix: float = 0.25
    mix_range: tuple[float, float] = (0.3, 0.7)

    p_background: float = 0.5
    snr_db: tuple[float, float] = (3.0, 20.0)

    p_shift: float = 0.5
    max_shift: float = 0.25
    min_overlap: float = 0.5

    p_paste: float = 0.5
    max_events: int = 3
    max_event_area: float = 0.25
    paste_gain_db: tuple[float, float] = (-6.0, 3.0)
    paste_max_iou: float = 0.3

    p_mask: float = 0.4
    mask_event_frac: float = 0.25


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
        pasteable = (
            (duration >= MIN_EVENT_FRAMES)
            & (duration < n_frames)
            & (bandwidth > 0)
            & (every_box[:, 2] * every_box[:, 3] <= max_area)
        )

        self.patches = [
            images[index, 0, f0:f1, t0:t1]
            for index, (t0, f0), (t1, f1) in zip(
                window_of_box[pasteable].tolist(),
                low[pasteable].tolist(),
                high[pasteable].tolist(),
                strict=True,
            )
        ]
        self.rows = low[pasteable, 1].tolist()
        self.labels = torch.cat(labels)[pasteable]
        weights = 1 / self.labels.bincount()[self.labels]
        self.probs = weights / weights.sum()


def shift_boxes(
    boxes: Tensor, labels: Tensor, delta: float, min_overlap: float
) -> tuple[Tensor, Tensor]:
    widths = boxes[:, 2:3]
    laps = boxes.new_tensor([-1.0, 0.0, 1.0])
    starts = boxes[:, :1] - widths / 2 + delta + laps
    low, high = starts.clamp(0.0, 1.0), (starts + widths).clamp(0.0, 1.0)

    keep = (high - low) >= min_overlap * widths
    origin = keep.nonzero()[:, 0]
    low, high = low[keep], high[keep]
    moved = torch.stack([(low + high) / 2, boxes[origin, 1], high - low, boxes[origin, 3]], dim=1)
    return moved, labels[origin]


def copy_paste(
    mel: Tensor, boxes: Tensor, labels: Tensor, bank: EventBank, config: AugmentConfig
) -> tuple[Tensor, Tensor]:
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

        power_gain = 10 ** (random.uniform(*config.paste_gain_db) / 10)
        mel[..., row : row + height, column : column + width] += patch * power_gain
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
        if random.random() < config.p_mix:
            other = random.randrange(len(self.clips))
            weight = random.uniform(*config.mix_range)
            wave = weight * wave + (1 - weight) * load_clip(*self.clips[other])
            boxes = torch.cat([boxes, self.boxes[other]])
            labels = torch.cat([labels, self.labels[other]])
        if self.background_clips and random.random() < config.p_background:
            background = load_clip(*random.choice(self.background_clips))
            signal_power = wave.square().mean()
            background_power = background.square().mean().clamp_min(P.eps)
            snr = 10 ** (random.uniform(*config.snr_db) / 10)
            wave = wave + background * (signal_power / background_power / snr).sqrt()

        mel = self.to_mel(wave)[None]
        n_mels, n_frames = mel.shape[-2:]

        if random.random() < config.p_shift:
            span = int(config.max_shift * n_frames)
            frames = random.randint(-span, span)
            mel = mel.roll(frames, dims=-1)
            boxes, labels = shift_boxes(boxes, labels, frames / n_frames, config.min_overlap)
        if random.random() < config.p_paste:
            boxes, labels = copy_paste(mel, boxes, labels, self.bank, config)
        if random.random() < config.p_mask:
            fill = mel.mean()
            smallest = torch.cat([boxes[:, 2:], mel.new_ones(1, 2)]).amin(0)
            for dim, size, event in ((-1, n_frames, smallest[0]), (-2, n_mels, smallest[1])):
                span = random.randint(1, max(1, int(event * size * config.mask_event_frac)))
                start = random.randrange(size - span + 1)
                mel.movedim(dim, -1)[..., start : start + span] = fill

        return mel, {"boxes": boxes, "labels": labels}
