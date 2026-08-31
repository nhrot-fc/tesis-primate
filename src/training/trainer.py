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

from core.config import SEED
from data.datasets import BoxJitter, to_device
from data.species import LabelSet
from evaluation.evaluator import evaluate
from evaluation.metrics import BETA, DetectionMetrics
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
    workers: int = 0
    seed: int = SEED
    iou_threshold: float = 0.5  # con el que una detección cuenta como acierto
    score_threshold: float = 0.5  # punto de operación que reportan recall y precisión
    nms_iou: float = 0.3
    beta: float = BETA  # el de la F-beta que elige el checkpoint
    jitter: BoxJitter | None = BoxJitter()  # aumentación de cajas, sólo en train


# Todo detector devuelve pérdidas con `model(imágenes, targets)` y cajas con su `detect`,
# así que acá no hay una rama por arquitectura.
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
        dataset_meta: dict[str, Any],
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
        self.clip_grad = architecture(name).clip_grad
        self.detect = partial(detector(name), model)
        self.run_config = {
            "architecture": name,
            "hparams": hparams,
            "dataset": dataset_meta,
            **asdict(config),
        }

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay
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
            sum(p.numel() for p in trainable) / 1e6,
            len(labels),
            hparams,
        )

    def _train_epoch(self, desc: str) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}

        progress = tqdm(self.train_loader, desc=desc, unit="batch", leave=False)
        for step, batch in enumerate(progress, start=1):
            images, targets = to_device(batch, self.device)
            terms: dict[str, Tensor] = self.model(images, targets)
            total = torch.stack(list(terms.values())).sum()

            self.optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
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

    def _validate(self, desc: str) -> DetectionMetrics:
        return evaluate(
            self.detect,
            self.val_loader,
            n_classes=len(self.labels),
            device=self.device,
            iou_threshold=self.config.iou_threshold,
            score_threshold=self.config.score_threshold,
            nms_iou=self.config.nms_iou,
            beta=self.config.beta,
            desc=desc,
        )

    def _record(
        self, epoch: int, lr: float, losses: dict[str, float], val: DetectionMetrics
    ) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            record = {"epoch": epoch + 1, "lr": lr, "train": losses, "val": val._asdict()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save(self, name: str, epoch: int, val: DetectionMetrics, **resumable: Any) -> None:
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

    def fit(self) -> DetectionMetrics | None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(self.run_config, indent=2, ensure_ascii=False, default=str)
        )
        start, best_score = checkpoint.resume(
            self.run_dir, self.name, self.model, self.optimizer, self.scheduler
        )
        best: DetectionMetrics | None = None

        for epoch in range(start, self.config.epochs):
            progress = f"{epoch + 1}/{self.config.epochs}"
            learning_rate = self.optimizer.param_groups[0]["lr"]

            losses = self._train_epoch(f"train {progress}")
            val = self._validate(f"val {progress}")
            score = 0.0 if val.f_beta is None else val.f_beta

            logger.info("[%4s] loss=%.3f %s", progress, losses["total"], format_line(val))
            self._record(epoch, learning_rate, losses, val)

            # `best.pt` primero: si el proceso muere entre los dos, `last.pt` no queda
            # afirmando un mejor score que en disco no existe.
            if score > best_score:
                best_score, best = score, val
                self._save(checkpoint.BEST, epoch, val)
                logger.info("nuevo mejor F%g=%.3f", self.config.beta, score)
            self._save(
                checkpoint.LAST,
                epoch,
                val,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best_score=best_score,
            )

        return best
