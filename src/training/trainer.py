import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from core.config import MAX_DETECTIONS, SCORE_THRESHOLD, SEED, WORKERS
from core.runtime import progress
from data import cache
from data.augment import AugmentConfig
from data.datasets import BoxJitter
from data.species import LabelSet
from evaluation.evaluator import add_losses, average_loss, evaluate, loss_terms, mean_losses
from evaluation.metrics import MATCH_IOU, DetectionMetrics
from evaluation.report import format_line
from models.base import Detector
from training import checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup: float = 0.05  # fracción de pasos del OneCycle en subida
    workers: int = WORKERS
    seed: int = SEED
    # Umbral del recall/precisión que se loguea; `best.pt` se elige por mAP@0.3
    score_threshold: float = SCORE_THRESHOLD
    jitter: BoxJitter | None = BoxJitter()  # sólo en train
    augment: AugmentConfig | None = AugmentConfig()


class Trainer:
    def __init__(
        self,
        model: Detector,
        architecture: str,  # clave en `models.registry.ARCHITECTURES`
        hparams: dict[str, Any],
        labels: LabelSet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path,
        config: TrainConfig,
        device: str,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.hparams = hparams
        self.labels = labels
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.config = config
        self.device = device
        self.run_config = {
            "architecture": architecture,
            "hparams": hparams,
            "dataset": cache.meta(),
            "match_iou": MATCH_IOU,
            "nms_iou": model.nms_iou,
            **asdict(config),
        }

        self.trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.epochs * len(train_loader),
            pct_start=config.warmup,
            anneal_strategy="cos",
        )
        logger.info(
            "%s | %.1fM parámetros (%.1fM entrenables) | %d clases | %s",
            architecture,
            sum(p.numel() for p in model.parameters()) / 1e6,
            sum(p.numel() for p in self.trainable) / 1e6,
            len(labels),
            hparams,
        )

    def train_epoch(self, desc: str) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        batches = progress(self.train_loader, desc)
        for step, batch in enumerate(batches, start=1):
            losses = loss_terms(self.model, batch, self.device)

            self.optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.trainable, self.model.clip_grad)
            self.optimizer.step()
            self.scheduler.step()

            add_losses(totals, losses)
            batches.set_postfix(
                loss=totals["total"] / step, lr=self.optimizer.param_groups[0]["lr"]
            )
        return mean_losses(totals, len(self.train_loader))

    def validate(self, desc: str) -> DetectionMetrics:
        self.model.eval()
        try:
            return evaluate(
                self.model.detect,
                self.val_loader,
                n_classes=len(self.labels),
                device=self.device,
                nms_iou=self.model.nms_iou,
                score_threshold=self.config.score_threshold,
                max_detections=MAX_DETECTIONS,
                desc=desc,
            )
        finally:
            self.model.train()

    def record_epoch(
        self, epoch: int, lr: float, losses: dict, val: DetectionMetrics, val_losses: dict
    ) -> None:
        record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train": losses,
            "val": val._asdict(),
            "val_loss": val_losses,
        }
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(
        self, name: str, epoch: int, val: DetectionMetrics, **resumable: Any
    ) -> None:
        checkpoint.save(
            self.run_dir / name,
            architecture=self.architecture,
            model=self.model,
            hparams=self.hparams,
            labels=self.labels,
            config=self.run_config,
            epoch=epoch,
            metrics=val._asdict(),
            **resumable,
        )

    def fit(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(self.run_config, indent=2, ensure_ascii=False, default=str)
        )
        start, best_score = checkpoint.resume(
            self.run_dir, self.model, self.optimizer, self.scheduler
        )

        for epoch in range(start, self.config.epochs):
            progress = f"{epoch + 1}/{self.config.epochs}"
            learning_rate = self.optimizer.param_groups[0]["lr"]
            losses = self.train_epoch(f"train {progress}")
            val = self.validate(f"val {progress}")
            val_losses = average_loss(
                self.model, self.val_loader, self.device, f"val loss {progress}"
            )
            score = val.map_30 or 0.0

            logger.info(
                "[%4s] loss=%.3f val_loss=%.3f %s",
                progress,
                losses["total"],
                val_losses["total"],
                format_line(val),
            )
            self.record_epoch(epoch, learning_rate, losses, val, val_losses)
            if score > best_score:
                best_score = score
                self.save_checkpoint(checkpoint.BEST, epoch, val)
                logger.info("nuevo mejor mAP%.0f=%.3f", MATCH_IOU * 100, score)
            self.save_checkpoint(
                checkpoint.LAST,
                epoch,
                val,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best_score=best_score,
            )

        logger.info("mejor mAP%.0f=%.3f | %s", MATCH_IOU * 100, best_score, self.run_dir)
