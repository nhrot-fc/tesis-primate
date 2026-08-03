import json
import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from architectures.criterion import HungarianMatcher, SetCriterion
from architectures.deformable_detr import ASTDeformableDETR
from core.config import settings
from core.setup import setup_logging, setup_project_path
from domain.dataset import BoxJitter, CachedCallBoxDataset, collate_fn
from domain.species import LabelSet
from pipelines.training_pipeline import (
    EvalMetrics,
    Losses,
    evaluate,
    format_metric,
    train_one_epoch,
)

logger = logging.getLogger("training")

PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "data" / "processed"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

# --- Preprocesado ---------------------------------------------------------------
# El dataset (manifest + mel + cajas) se materializa aparte con `create_dataset.py`;
# acá sólo se carga lo que ya quedó cacheado en `CACHE_DIR`.
SEED = 42

# --- Entrenamiento ------------------------------------------------------------
MODEL_DIM, N_QUERIES, N_LEVELS = 128, 16, 3
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS = 20, 64, 2e-4, 1e-4, 0
DETAIL_EVERY = 4
BOX_JITTER = BoxJitter(scale=0.15, shift=0.10, min_size=0.02)
MATCHER_IOU_TYPE = "iou"
METRIC_IOU_THRESHOLD = 0.5
OPERATING_SCORE_THRESHOLD = 0.5
NMS_IOU = 0.3
CHECKPOINT_SELECTION_BETA = 3.0


def training_config() -> dict[str, object]:
    dataset_meta = json.loads((CACHE_DIR / "meta.json").read_text())
    return {
        "seed": SEED,
        "dataset": dataset_meta,
        "model_dim": MODEL_DIM,
        "n_queries": N_QUERIES,
        "n_levels": N_LEVELS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "box_jitter": asdict(BOX_JITTER),
        "matcher_iou_type": MATCHER_IOU_TYPE,
        "metric_iou_threshold": METRIC_IOU_THRESHOLD,
        "operating_score_threshold": OPERATING_SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "checkpoint_selection_beta": CHECKPOINT_SELECTION_BETA,
    }


def operating_score(
    recall: float | None, fp_per_tp: float | None, beta: float = CHECKPOINT_SELECTION_BETA
) -> float:
    # `None` = métrica indefinida (split sin GT, o ni un TP en el punto de operación).
    # Un checkpoint así no puede ganar la selección, así que vale 0.
    if recall is None or fp_per_tp is None:
        return 0.0
    precision = 1.0 / (1.0 + fp_per_tp)
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)


def save_labels_json(labels: LabelSet, path: Path) -> None:
    path.write_text(json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False))


def make_loader(dataset: CachedCallBoxDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def load_datasets() -> tuple[LabelSet, dict, CachedCallBoxDataset, CachedCallBoxDataset]:
    if not (CACHE_DIR / "meta.json").exists():
        raise FileNotFoundError(
            f"no hay dataset cacheado en {CACHE_DIR}. Corré `python src/create_dataset.py` primero."
        )
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    # El modelo ya no usa estas estadísticas (normaliza la salida del PCEN, ver
    # `ASTDeformableDETR.pcen_norm`), pero su ausencia sigue delatando un caché viejo:
    # esos guardaban el mel ya estandarizado por clip, o sea otra entrada.
    if not meta.get("normalization"):
        raise ValueError(
            f"{CACHE_DIR / 'meta.json'} no tiene las estadísticas de normalización: es un caché "
            "viejo, con el mel ya estandarizado por clip. Regenerálo con "
            "`python src/create_dataset.py`."
        )

    label_names = json.loads((CACHE_DIR / "labels.json").read_text())
    labels = LabelSet(label_names.values())

    train_dataset = CachedCallBoxDataset(CACHE_DIR / "train.pt", jitter=BOX_JITTER)
    val_dataset = CachedCallBoxDataset(CACHE_DIR / "val.pt")

    class_counts = Counter(
        labels.name(int(class_id))
        for window_labels in train_dataset.labels
        for class_id in window_labels
    )
    logger.info("ventanas -> train %d | val %d", len(train_dataset), len(val_dataset))
    logger.info("cajas en train -> %s", dict(class_counts.most_common()))
    logger.info("jitter de cajas en train -> %s", BOX_JITTER)
    logger.info("estadísticas del mel de potencia (informativas) -> %s", meta["normalization"])
    return labels, meta, train_dataset, val_dataset


def format_confusion(confusion: Tensor, names: list[str]) -> str:
    columns = [*names, "∅"]  # última columna: la query dijo "no-objeto"
    header = "true\\pred".rjust(10) + "".join(f"{name:>10}" for name in columns)
    rows = [header]
    for i, name in enumerate(names):
        row = f"{name:>10}" + "".join(f"{int(confusion[i, j]):>10}" for j in range(len(columns)))
        rows.append(row)
    return "\n".join(rows)


def format_recall_per_class(recall_per_class: dict[int, float | None], names: list[str]) -> str:
    return ", ".join(
        f"{names[class_id]}={format_metric(recall)}"
        for class_id, recall in sorted(recall_per_class.items())
    )


class TrainingComponents(NamedTuple):
    model: ASTDeformableDETR
    matcher: HungarianMatcher
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: OneCycleLR


def build_model(n_classes: int, steps_per_epoch: int, device: str) -> TrainingComponents:
    # Sin estadísticas de normalización acá: las del caché son del mel de potencia, y lo
    # que hay que normalizar es la salida del PCEN, que se mueve mientras entrena. El
    # modelo las estima solo (`ASTDeformableDETR.pcen_norm`).
    model = ASTDeformableDETR(
        dim=MODEL_DIM,
        n_queries=N_QUERIES,
        n_classes=n_classes,
        n_levels=N_LEVELS,
    ).to(device)
    matcher = HungarianMatcher(iou_type=MATCHER_IOU_TYPE)
    criterion = SetCriterion(n_classes=n_classes, matcher=matcher, iou_type=MATCHER_IOU_TYPE).to(
        device
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.05,
        anneal_strategy="cos",
    )
    logger.info(
        "%.1fM parámetros (%.1fM entrenables) | %d clases",
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in trainable) / 1e6,
        n_classes,
    )
    return TrainingComponents(model, matcher, criterion, optimizer, scheduler)


def log_epoch(epoch: int, train_losses: Losses, val_metrics: EvalMetrics) -> None:
    logger.info(
        "[%4d/%d] train=%.3f val=%.3f cls_acc=%.3f IoU=%.3f recall_agn@%.2f=%s FP/TP=%s",
        epoch + 1,
        EPOCHS,
        train_losses.total,
        val_metrics.losses.total,
        val_metrics.accuracy,
        val_metrics.mean_iou,
        METRIC_IOU_THRESHOLD,
        format_metric(val_metrics.recall_agnostic),
        format_metric(val_metrics.fp_per_tp_agnostic, digits=2),
    )


def log_detail(val_metrics: EvalMetrics, labels: LabelSet) -> None:
    logger.info(
        "Recall por clase -> %s",
        format_recall_per_class(val_metrics.recall_per_class, labels.names),
    )
    logger.info(
        "Precisión por clase -> %s",
        format_recall_per_class(val_metrics.precision_per_class, labels.names),
    )
    logger.info(
        "AP agnóstico de clase -> %s",
        ", ".join(
            f"{threshold}={format_metric(ap)}"
            for threshold, ap in sorted(val_metrics.ap_agnostic.items())
        ),
    )
    logger.info(
        "Matriz de confusión (queries emparejadas):\n%s",
        format_confusion(val_metrics.confusion, labels.names),
    )


def append_metrics(
    path: Path,
    epoch: int,
    train_losses: Losses,
    val_metrics: EvalMetrics,
    learning_rate: float,
    score: float,
) -> None:
    record = {
        "epoch": epoch + 1,
        "lr": learning_rate,
        "train": train_losses._asdict(),
        "val": {
            "losses": val_metrics.losses._asdict(),
            "accuracy": val_metrics.accuracy,
            "mean_iou": val_metrics.mean_iou,
            # `*_agnostic`: agnósticas de clase, ver `EvalMetrics`. El nombre viaja al
            # jsonl para que nadie las lea después como si fueran recall/AP por especie.
            "recall_agnostic": val_metrics.recall_agnostic,
            "precision_agnostic": val_metrics.precision_agnostic,
            "fp_per_tp_agnostic": val_metrics.fp_per_tp_agnostic,
            "operating_score": score,
            "ap_agnostic": {
                str(threshold): ap for threshold, ap in sorted(val_metrics.ap_agnostic.items())
            },
            "recall_per_class": val_metrics.recall_per_class,
            "precision_per_class": val_metrics.precision_per_class,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def model_state_dict(model: nn.Module) -> dict[str, Tensor]:
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    return {
        key: value.detach().cpu() for key, value in model.state_dict().items() if key not in frozen
    }


class BestTracker:
    def __init__(self, checkpoint_path: Path, labels: LabelSet, config: dict[str, object]) -> None:
        self.checkpoint_path = checkpoint_path
        self.labels = labels
        self.config = config
        self.best_score = float("-inf")
        self.best_metrics: EvalMetrics | None = None

    def consider(
        self, epoch: int, model: ASTDeformableDETR, val_metrics: EvalMetrics, score: float
    ) -> None:
        if score <= self.best_score:
            return
        self.best_score, self.best_metrics = score, val_metrics
        torch.save(
            {
                "state_dict": model_state_dict(model),
                "labels": self.labels.names,
                "dim": MODEL_DIM,
                "n_queries": N_QUERIES,
                "n_levels": N_LEVELS,
                "n_frames": model.backbone.n_frames,
                "time_stride": model.backbone.time_stride,
                "config": self.config,
                "epoch": epoch,
                "recall_agn": val_metrics.recall_agnostic,
                "fp_per_tp": val_metrics.fp_per_tp_agnostic,
            },
            self.checkpoint_path,
        )
        logger.info("Nuevo mejor score=%.3f -> %s", score, self.checkpoint_path)


def train(
    labels: LabelSet,
    train_dataset: CachedCallBoxDataset,
    val_dataset: CachedCallBoxDataset,
    device: str,
    metrics_path: Path,
) -> None:
    torch.manual_seed(SEED)
    n_classes = len(labels)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    model, matcher, criterion, optimizer, scheduler = build_model(
        n_classes, len(train_loader), device
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = f"{MODEL_DIM}d_{N_QUERIES}q_{MATCHER_IOU_TYPE}iou_{n_classes}cls"
    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"
    labels_path = CHECKPOINT_DIR / f"{name}_labels.json"

    save_labels_json(labels, labels_path)
    logger.info(
        "class_id -> label -> %s",
        {class_id: label for class_id, label in enumerate(labels.names)},
    )

    tracker = BestTracker(checkpoint_path, labels, training_config())

    for epoch in range(EPOCHS):
        progress = f"{epoch + 1}/{EPOCHS}"
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            desc=f"train {progress}",
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            matcher,
            device,
            n_classes=n_classes,
            iou_threshold=METRIC_IOU_THRESHOLD,
            score_threshold=OPERATING_SCORE_THRESHOLD,
            nms_iou=NMS_IOU,
            desc=f"val {progress}",
        )
        score = operating_score(val_metrics.recall_agnostic, val_metrics.fp_per_tp_agnostic)

        log_epoch(epoch, train_losses, val_metrics)
        append_metrics(metrics_path, epoch, train_losses, val_metrics, learning_rate, score)
        tracker.consider(epoch, model, val_metrics, score)

        is_last = epoch + 1 == EPOCHS
        if (epoch + 1) % DETAIL_EVERY == 0 or is_last:
            log_detail(val_metrics, labels)

    if tracker.best_metrics is None:
        raise RuntimeError("No se completó ninguna época.")
    logger.info(
        "mejor recall_agn@%.2f de validación: %s (FP/TP=%s, score=%.3f) -> %s",
        METRIC_IOU_THRESHOLD,
        format_metric(tracker.best_metrics.recall_agnostic),
        format_metric(tracker.best_metrics.fp_per_tp_agnostic, digits=2),
        tracker.best_score,
        checkpoint_path,
    )
    logger.info("métricas por época -> %s", metrics_path)


if __name__ == "__main__":
    run_dir = LOG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(settings.LOG_LEVEL, log_file=run_dir / "train.log")
    setup_project_path(PROJECT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    labels, _meta, train_dataset, val_dataset = load_datasets()
    (run_dir / "config.json").write_text(
        json.dumps(training_config(), indent=2, ensure_ascii=False, default=str)
    )
    train(labels, train_dataset, val_dataset, device, run_dir / "metrics.jsonl")
