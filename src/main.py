import copy
import json
import logging
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from architectures.criterion import HungarianMatcher, SetCriterion
from architectures.deformable_detr import ASTDeformableDETR
from core.config import settings
from core.setup import setup_logging, setup_project_path
from domain.dataset import CachedCallBoxDataset, collate_fn
from domain.species import LabelSet
from pipelines.training_pipeline import EvalMetrics, Metrics, evaluate, train_one_epoch

logger = logging.getLogger("training")

PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "data" / "processed"

# --- Preprocesado ---------------------------------------------------------------
# El dataset (manifest + log-mel + cajas) se materializa aparte con `create_dataset.py`;
# acá sólo se carga lo que ya quedó cacheado en `CACHE_DIR`.
SEED = 42

# --- Entrenamiento ------------------------------------------------------------
MODEL_DIM, N_QUERIES = 128, 64
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS = 20, 16, 2e-4, 1e-4, 0
EVAL_EVERY = 4
MATCHER_IOU_TYPE = "iou"
METRIC_IOU_THRESHOLD = 0.5
OPERATING_SCORE_THRESHOLD = 0.5
CHECKPOINT_SELECTION_BETA = 3.0

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"


def training_config() -> dict[str, object]:
    """Snapshot de los hiperparámetros de esta corrida, para guardar dentro del
    checkpoint: sin esto un `.pth` no es reproducible más allá de `dim`/`n_queries`."""
    dataset_meta = json.loads((CACHE_DIR / "meta.json").read_text())
    return {
        "seed": SEED,
        "dataset": dataset_meta,
        "model_dim": MODEL_DIM,
        "n_queries": N_QUERIES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "matcher_iou_type": MATCHER_IOU_TYPE,
        "metric_iou_threshold": METRIC_IOU_THRESHOLD,
        "operating_score_threshold": OPERATING_SCORE_THRESHOLD,
        "checkpoint_selection_beta": CHECKPOINT_SELECTION_BETA,
    }


def operating_score(
    recall: float, fp_per_tp: float, beta: float = CHECKPOINT_SELECTION_BETA
) -> float:
    precision = 1.0 / (1.0 + fp_per_tp)
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)


def save_labels_json(labels: LabelSet, path: Path) -> None:
    """class_id -> label, para inspección rápida y para que `infer.py` los lea
    sin tener que cargar el checkpoint completo."""
    path.write_text(json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False))


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def preprocess() -> tuple[LabelSet, DataLoader, DataLoader]:
    if not (CACHE_DIR / "meta.json").exists():
        raise FileNotFoundError(
            f"no hay dataset cacheado en {CACHE_DIR}. Corré `python src/create_dataset.py` primero."
        )

    label_names = json.loads((CACHE_DIR / "labels.json").read_text())
    labels = LabelSet(label_names.values())

    train_dataset = CachedCallBoxDataset(CACHE_DIR / "train.pt")
    val_dataset = CachedCallBoxDataset(CACHE_DIR / "val.pt")

    class_counts = Counter(
        labels.name(int(class_id))
        for window_labels in train_dataset.labels
        for class_id in window_labels
    )
    logger.info("ventanas -> train %d | val %d", len(train_dataset), len(val_dataset))
    logger.info("cajas en train -> %s", dict(class_counts.most_common()))

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    return labels, train_loader, val_loader


def format_confusion(confusion: torch.Tensor, names: list[str]) -> str:
    header = "true\\pred".rjust(10) + "".join(f"{name:>10}" for name in names)
    rows = [header]
    for i, name in enumerate(names):
        row = f"{name:>10}" + "".join(f"{int(confusion[i, j]):>10}" for j in range(len(names)))
        rows.append(row)
    return "\n".join(rows)


def format_recall_per_class(recall_per_class: dict[int, float | None], names: list[str]) -> str:
    return ", ".join(
        f"{names[class_id]}={recall:.3f}" if recall is not None else f"{names[class_id]}=n/a"
        for class_id, recall in sorted(recall_per_class.items())
    )


class TrainingComponents(NamedTuple):
    model: nn.Module
    matcher: HungarianMatcher
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: OneCycleLR


def build_model(n_classes: int, train_loader: DataLoader, device: str) -> TrainingComponents:
    model = ASTDeformableDETR(dim=MODEL_DIM, n_queries=N_QUERIES, n_classes=n_classes).to(device)
    matcher = HungarianMatcher(iou_type=MATCHER_IOU_TYPE)
    criterion = SetCriterion(n_classes=n_classes, matcher=matcher, iou_type=MATCHER_IOU_TYPE).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )
    logger.info(
        "%.1fM parámetros | %d clases",
        sum(p.numel() for p in model.parameters()) / 1e6,
        n_classes,
    )
    return TrainingComponents(model, matcher, criterion, optimizer, scheduler)


def log_epoch(epoch: int, train_metrics: Metrics, val_metrics: EvalMetrics) -> None:
    logger.info(
        "[%4d/%d] train=%.3f val=%.3f cls_acc=%.3f IoU=%.3f recall_agn@%.2f=%.3f FP/TP=%.2f",
        epoch + 1,
        EPOCHS,
        train_metrics["total"],
        val_metrics.losses.total,
        val_metrics.classification.accuracy,
        val_metrics.framing.mean_iou,
        METRIC_IOU_THRESHOLD,
        val_metrics.detection.recall,
        val_metrics.detection.fp_per_tp,
    )


def log_detail(val_metrics: EvalMetrics, labels: LabelSet) -> None:
    logger.info(
        "Recall por clase -> %s",
        format_recall_per_class(val_metrics.classification.recall_per_class, labels.names),
    )
    logger.info(
        "AP agnóstico de clase -> %s",
        ", ".join(
            f"{threshold}={ap:.3f}" if ap is not None else f"{threshold}=n/a"
            for threshold, ap in sorted(val_metrics.framing.ap_agnostic.items())
        ),
    )
    logger.info(
        "matriz de confusión (queries emparejadas)\n%s",
        format_confusion(val_metrics.classification.confusion, labels.names),
    )


class BestTracker:
    """Selecciona y persiste el mejor checkpoint según `operating_score` (recall_agn
    con fp_per_tp de penalización, ver docstring de `operating_score`)."""

    def __init__(self, checkpoint_path: Path, labels: LabelSet) -> None:
        self.checkpoint_path = checkpoint_path
        self.labels = labels
        self.best_score = float("-inf")
        self.best_metrics: EvalMetrics | None = None

    def consider(self, epoch: int, model: nn.Module, val_metrics: EvalMetrics) -> None:
        score = operating_score(val_metrics.detection.recall, val_metrics.detection.fp_per_tp)
        if score <= self.best_score:
            return
        self.best_score, self.best_metrics = score, val_metrics
        torch.save(
            {
                "state_dict": copy.deepcopy(model.state_dict()),
                "labels": self.labels.names,
                "dim": MODEL_DIM,
                "n_queries": N_QUERIES,
                "config": training_config(),
                "epoch": epoch,
                "recall_agn": val_metrics.detection.recall,
                "fp_per_tp": val_metrics.detection.fp_per_tp,
            },
            self.checkpoint_path,
        )


def train(labels: LabelSet, train_loader: DataLoader, val_loader: DataLoader, device: str) -> None:
    torch.manual_seed(SEED)
    n_classes = len(labels)

    model, matcher, criterion, optimizer, scheduler = build_model(n_classes, train_loader, device)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = f"{MODEL_DIM}d_{N_QUERIES}q_{MATCHER_IOU_TYPE}iou_{n_classes}cls"
    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"
    labels_path = CHECKPOINT_DIR / f"{name}_labels.json"

    save_labels_json(labels, labels_path)
    logger.info(
        "class_id -> label -> %s",
        {class_id: label for class_id, label in enumerate(labels.names)},
    )

    tracker = BestTracker(checkpoint_path, labels)

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch=epoch,
            epochs=EPOCHS,
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
            epoch=epoch,
            epochs=EPOCHS,
        )
        log_epoch(epoch, train_metrics, val_metrics)
        tracker.consider(epoch, model, val_metrics)

        is_last = epoch + 1 == EPOCHS
        if (epoch + 1) % EVAL_EVERY == 0 or is_last:
            log_detail(val_metrics, labels)

    if tracker.best_metrics is None:
        raise RuntimeError("No se completó ninguna época.")
    logger.info(
        "mejor recall_agn@%.2f de validación: %.3f (FP/TP=%.2f, score=%.3f) -> %s",
        METRIC_IOU_THRESHOLD,
        tracker.best_metrics.detection.recall,
        tracker.best_metrics.detection.fp_per_tp,
        tracker.best_score,
        checkpoint_path,
    )


if __name__ == "__main__":
    setup_logging(settings.LOG_LEVEL)
    setup_project_path(PROJECT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    labels, train_loader, val_loader = preprocess()
    train(labels, train_loader, val_loader, device)
