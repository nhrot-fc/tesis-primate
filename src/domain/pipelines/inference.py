from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from domain.pipelines.annotations import annotations_to_df
from domain.pipelines.audio import load_audio_torchaudio
from domain.pipelines.image import compute_spectrogram, normalize_spectrogram, scale_min_max
from domain.pipelines.pipeline import WindowConfig
from domain.pipelines.types import YoloBox, yolo_to_annotation


class BioacousticInference:
    def __init__(self, model_path, cfg: WindowConfig):
        self.model = YOLO(model_path)
        self.cfg = cfg

    def _get_spec(self, clip_wav):
        spec_raw = compute_spectrogram(clip_wav, self.cfg.n_fft, self.cfg.hop_length).flip(
            dims=[-2]
        )
        channels = [normalize_spectrogram(spec_raw, m) for m in self.cfg.channel_methods]
        spec = torch.cat(channels, dim=0)
        spec = scale_min_max(spec)
        return F.interpolate(
            spec.unsqueeze(0), size=(self.cfg.img_size, self.cfg.img_size)
        ).squeeze(0)

    def run(self, wav_path: Path, conf_threshold=0.3):
        wav = load_audio_torchaudio(wav_path, self.cfg.sample_rate)
        total_sec = wav.shape[0] / self.cfg.sample_rate
        max_freq = self.cfg.sample_rate / 2
        detections = []
        start_sec = 0.0

        while start_sec + self.cfg.duration_sec <= total_sec:
            start_idx = int(start_sec * self.cfg.sample_rate)
            end_idx = start_idx + int(self.cfg.duration_sec * self.cfg.sample_rate)

            spec = self._get_spec(wav[start_idx:end_idx])
            results = self.model.predict(spec.unsqueeze(0), conf=conf_threshold, verbose=False)

            if results[0].boxes:
                for box in results[0].boxes:
                    xc, yc, w, h = box.xywhn[0].tolist()
                    yolo_box = YoloBox(
                        class_id=int(box.cls[0].item()),
                        xc_rel=xc,
                        yc_rel=yc,
                        w_rel=w,
                        h_rel=h,
                    )

                    class_name = results[0].names[yolo_box.class_id]
                    ann = yolo_to_annotation(
                        yolo_box=yolo_box,
                        window_start_sec=start_sec,
                        window_duration_sec=self.cfg.duration_sec,
                        sample_rate=self.cfg.sample_rate,
                        class_name=class_name,
                    )
                    detections.append(ann)

            start_sec += self.cfg.hop_sec

        return annotations_to_df(detections)
