import json
import logging
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.config import SEED, WORKERS
from data import cache
from data.augment import AugmentConfig
from data.datasets import BoxJitter, to_device
from data.species import LabelSet
from evaluation.evaluator import average_loss, evaluate
from evaluation.metrics import BETA, MATCH_IOU, MAX_DETECTIONS, DetectionMetrics
from evaluation.report import format_line
from models.registry import architecture, detector
from training import checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    workers: int = WORKERS
    seed: int = SEED
    # El checkpoint se elige por mAP@`MATCH_IOU`, que no depende de ningún umbral de score.
    # Estos dos sólo fijan el punto de operación que se loguea para mirar la corrida; el
    # definitivo lo busca `evaluation.protocol` sobre val al comparar los modelos.
    score_threshold: float = 0.5
    beta: float = BETA
    jitter: BoxJitter | None = BoxJitter()  # aumentación de cajas, sólo en train
    augment: AugmentConfig | None = AugmentConfig()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        name: str,  # clave en `models.registry.ARCHITECTURES`
        hparams: dict[str, Any],  # con lo que `load_checkpoint` rearma el grafo
        labels: LabelSet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path,
        config: TrainConfig,
        device: str,
    ) -> None:
        self.model = model
        self.name = name
        self.hparams = hparams
        self.labels = labels
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.config = config
        self.device = device
        self.spec = architecture(name)
        self.clip_grad = self.spec.clip_grad
        self.detect = partial(detector(name), model)
        self.run_config = {
            "architecture": name,
            "hparams": hparams,
            "dataset": cache.meta(),
            "match_iou": MATCH_IOU,
            "nms_iou": self.spec.nms_iou,
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
            pct_start=0.05,
            anneal_strategy="cos",
        )
        logger.info(
            "%s | %.1fM parámetros (%.1fM entrenables) | %d clases | %s",
            name,
            sum(p.numel() for p in model.parameters()) / 1e6,
            sum(p.numel() for p in self.trainable) / 1e6,
            len(labels),
            hparams,
        )

    def train_epoch(self, desc: str) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}

        progress = tqdm(self.train_loader, desc=desc, unit="batch", leave=False, disable=None)
        for step, batch in enumerate(progress, start=1):
            images, targets = to_device(batch, self.device)
            terms: dict[str, Tensor] = self.model(images, targets)
            total = torch.stack(list(terms.values())).sum()

            self.optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.trainable, self.clip_grad)
            self.optimizer.step()
            self.scheduler.step()

            losses = {"total": total, **terms}
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            progress.set_postfix(
                loss=totals["total"] / step, lr=self.optimizer.param_groups[0]["lr"]
            )

        n = max(len(self.train_loader), 1)
        return {key.removeprefix("loss_"): value / n for key, value in totals.items()}

    def validate(self, desc: str) -> DetectionMetrics:
        self.model.eval()
        try:
            return evaluate(
                self.detect,
                self.val_loader,
                n_classes=len(self.labels),
                device=self.device,
                score_threshold=self.config.score_threshold,
                nms_iou=self.spec.nms_iou,
                # El mismo tope que `protocol.equalize` en la comparación final: si acá se
                # midiera sin límite, el mAP que elige `best.pt` no sería el de la tabla.
                max_detections=MAX_DETECTIONS,
                beta=self.config.beta,
                desc=desc,
            )
        finally:
            self.model.train()

    def record_epoch(
        self,
        epoch: int,
        lr: float,
        losses: dict[str, float],
        val: DetectionMetrics,
        val_losses: dict[str, float],
    ) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            record = {
                "epoch": epoch + 1,
                "lr": lr,
                "train": losses,
                "val": val._asdict(),
                "val_loss": val_losses,  # mismos términos que `train`
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(
        self, name: str, epoch: int, val: DetectionMetrics, **resumable: Any
    ) -> None:
        checkpoint.save(
            self.run_dir / name,
            architecture=self.name,
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
            # Sin la pérdida de val, un mAP que se aplana no distingue sobreajuste de un
            # scheduler que llegó a cero: las dos curvas se ven igual.
            val_losses = average_loss(
                self.model, self.val_loader, self.device, f"val loss {progress}"
            )
            score = 0.0 if val.map_30 is None else val.map_30

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

        logger.info(
            "épocas nuevas: %d | mejor mAP%.0f=%.3f | %s",
            self.config.epochs - start,
            MATCH_IOU * 100,
            best_score,
            self.run_dir,
        )
