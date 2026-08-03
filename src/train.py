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
from tqdm.auto import tqdm

from architectures.criterion import HungarianMatcher, SetCriterion
from architectures.deformable_detr import ASTDeformableDETR
from core.config import settings
from core.setup import setup_logging, setup_project_path
from domain.dataset import BoxJitter, CachedCallBoxDataset, collate_fn
from domain.species import LabelSet
from pipelines.training_pipeline import EvalMetrics, Losses, evaluate, train_one_epoch

logger = logging.getLogger("training")

PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "data" / "processed"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

# --- Preprocesado ---------------------------------------------------------------
# El dataset (manifest + log-mel + cajas) se materializa aparte con `create_dataset.py`;
# acá sólo se carga lo que ya quedó cacheado en `CACHE_DIR`.
SEED = 42

# --- Entrenamiento ------------------------------------------------------------
MODEL_DIM, N_QUERIES, N_LEVELS = 128, 64, 3
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS = 20, 16, 2e-4, 1e-4, 0
DETAIL_EVERY = 4  # cada cuántas épocas se loguea el detalle por clase, no la evaluación
BOX_JITTER = BoxJitter(scale=0.15, shift=0.10, min_size=0.02)
MATCHER_IOU_TYPE = "iou"
METRIC_IOU_THRESHOLD = 0.5
OPERATING_SCORE_THRESHOLD = 0.5
NMS_IOU = 0.3  # el mismo que usa `inference_pipeline.predict`
CHECKPOINT_SELECTION_BETA = 3.0
# Presupuesto de RAM para guardar las features del backbone congelado. Si entran, el
# AST se corre una sola vez en vez de una por época (ver `precompute_features`).
BACKBONE_CACHE_MAX_GB = 6.0


def training_config() -> dict[str, object]:
    """Snapshot de los hiperparámetros de esta corrida, para guardar dentro del
    checkpoint: sin esto un `.pth` no es reproducible más allá de `dim`/`n_queries`."""
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
    recall: float, fp_per_tp: float, beta: float = CHECKPOINT_SELECTION_BETA
) -> float:
    precision = 1.0 / (1.0 + fp_per_tp)
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)


def save_labels_json(labels: LabelSet, path: Path) -> None:
    """class_id -> label, para inspección rápida y para que `infer.py` los lea
    sin tener que cargar el checkpoint completo."""
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
    if not meta.get("normalization"):
        raise ValueError(
            f"{CACHE_DIR / 'meta.json'} no tiene las estadísticas de normalización: es un caché "
            "viejo, con el log-mel ya estandarizado por clip. Regenerálo con "
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
    logger.info("normalización del log-mel -> %s", meta["normalization"])
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
        f"{names[class_id]}={recall:.3f}" if recall is not None else f"{names[class_id]}=n/a"
        for class_id, recall in sorted(recall_per_class.items())
    )


class TrainingComponents(NamedTuple):
    model: ASTDeformableDETR
    matcher: HungarianMatcher
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: OneCycleLR


def build_model(
    n_classes: int, steps_per_epoch: int, normalization: dict[str, float], device: str
) -> TrainingComponents:
    model = ASTDeformableDETR(
        dim=MODEL_DIM,
        n_queries=N_QUERIES,
        n_classes=n_classes,
        n_levels=N_LEVELS,
        mel_mean=normalization["mean"],
        mel_std=normalization["std"],
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


def backbone_cache_gb(model: ASTDeformableDETR, n_windows: int) -> float:
    tokens = model.backbone.freq_out * model.backbone.time_out
    return n_windows * tokens * model.backbone.hidden_size * 2 / 1024**3  # fp16


@torch.no_grad()
def precompute_features(
    model: ASTDeformableDETR, dataset: CachedCallBoxDataset, device: str, desc: str
) -> None:
    """Corre el AST congelado una sola vez sobre todo el split y deja los tokens en el
    dataset. Como las imágenes no se aumentan (el jitter toca sólo las cajas), su salida
    es idéntica en las 20 épocas: sin esto se recalcula ~30 veces el resto del modelo."""
    model.eval()
    images = dataset.images
    features: Tensor | None = None
    # autocast sólo en GPU: es una pasada única sobre todo el split y el resultado se
    # guarda en fp16 igual, así que calcularlo en fp32 no aporta nada.
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        for start in tqdm(range(0, len(images), BATCH_SIZE), desc=desc, unit="batch", leave=False):
            block = model.encode(images[start : start + BATCH_SIZE].to(device)).half().cpu()
            if features is None:
                features = torch.empty((len(images), *block.shape[1:]), dtype=torch.float16)
            features[start : start + len(block)] = block
    if features is not None:
        dataset.use_precomputed(features)


def maybe_cache_backbone(
    model: ASTDeformableDETR, datasets: list[CachedCallBoxDataset], device: str
) -> nn.Module:
    """Devuelve el módulo a entrenar: la cabeza sola si se pudo cachear el backbone,
    el modelo entero si no."""
    if not model.backbone.freeze:
        return model

    estimate = backbone_cache_gb(model, sum(len(dataset) for dataset in datasets))
    if estimate > BACKBONE_CACHE_MAX_GB:
        logger.info(
            "sin caché del backbone: harían falta %.1f GB > %.1f GB de presupuesto",
            estimate,
            BACKBONE_CACHE_MAX_GB,
        )
        return model

    logger.info("precomputando features del backbone congelado (~%.1f GB en fp16)...", estimate)
    for dataset, name in zip(datasets, ("train", "val"), strict=True):
        precompute_features(model, dataset, device, desc=f"backbone {name}")

    # Ya no se vuelve a usar en toda la corrida: sacarlo de la GPU libera ~350 MB de
    # VRAM. `trainable_state_dict` guarda desde CPU, así que el checkpoint no cambia.
    model.backbone.to("cpu")
    return model.head


def log_epoch(epoch: int, train_losses: Losses, val_metrics: EvalMetrics) -> None:
    logger.info(
        "[%4d/%d] train=%.3f val=%.3f cls_acc=%.3f IoU=%.3f recall_agn@%.2f=%.3f FP/TP=%.2f",
        epoch + 1,
        EPOCHS,
        train_losses.total,
        val_metrics.losses.total,
        val_metrics.accuracy,
        val_metrics.mean_iou,
        METRIC_IOU_THRESHOLD,
        val_metrics.recall,
        val_metrics.fp_per_tp,
    )


def log_detail(val_metrics: EvalMetrics, labels: LabelSet) -> None:
    logger.info(
        "Recall por clase -> %s",
        format_recall_per_class(val_metrics.recall_per_class, labels.names),
    )
    logger.info(
        "AP agnóstico de clase -> %s",
        ", ".join(
            f"{threshold}={ap:.3f}" if ap is not None else f"{threshold}=n/a"
            for threshold, ap in sorted(val_metrics.ap.items())
        ),
    )
    logger.info(
        "matriz de confusión (queries emparejadas)\n%s",
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
    """Una línea JSON por época: queda todo el historial de la corrida para graficar
    después sin tener que parsear el log de texto."""
    record = {
        "epoch": epoch + 1,
        "lr": learning_rate,
        "train": train_losses._asdict(),
        "val": {
            "losses": val_metrics.losses._asdict(),
            "accuracy": val_metrics.accuracy,
            "mean_iou": val_metrics.mean_iou,
            "recall": val_metrics.recall,
            "fp_per_tp": val_metrics.fp_per_tp,
            "operating_score": score,
            "ap": {str(threshold): ap for threshold, ap in sorted(val_metrics.ap.items())},
            "recall_per_class": val_metrics.recall_per_class,
            "confusion": val_metrics.confusion.tolist(),
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """`state_dict` sin los pesos congelados del AST: `ASTBackbone` los vuelve a cargar
    de la copia local, así que guardarlos son ~340 MB por checkpoint al pedo. En CPU,
    para que el `.pth` no dependa de que haya GPU al cargarlo."""
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    return {
        key: value.detach().cpu() for key, value in model.state_dict().items() if key not in frozen
    }


class BestTracker:
    """Selecciona y persiste el mejor checkpoint según `operating_score` (recall_agn
    con fp_per_tp de penalización, ver docstring de `operating_score`)."""

    def __init__(self, checkpoint_path: Path, labels: LabelSet, config: dict[str, object]) -> None:
        self.checkpoint_path = checkpoint_path
        self.labels = labels
        self.config = config  # se arma una sola vez: lee `meta.json` de disco
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
                "state_dict": trainable_state_dict(model),
                "labels": self.labels.names,
                "dim": MODEL_DIM,
                "n_queries": N_QUERIES,
                "n_levels": N_LEVELS,
                # geometría del backbone: no viaja en los pesos (el pos-embed se
                # re-interpola al construir), así que sin esto `infer.py` lo rearma
                # con los `Parameters` de hoy y no con los de esta corrida.
                "n_frames": model.backbone.n_frames,
                "time_stride": model.backbone.time_stride,
                "config": self.config,
                "epoch": epoch,
                "recall_agn": val_metrics.recall,
                "fp_per_tp": val_metrics.fp_per_tp,
            },
            self.checkpoint_path,
        )
        logger.info("nuevo mejor (score=%.3f) -> %s", score, self.checkpoint_path)


def train(
    labels: LabelSet,
    meta: dict,
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
        n_classes, len(train_loader), meta["normalization"], device
    )
    # Ojo: muta los datasets en el lugar, así que va antes de la primera época (los
    # loaders leen el dataset en cada iteración, no lo copian).
    trained_module = maybe_cache_backbone(model, [train_dataset, val_dataset], device)

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
            trained_module,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            desc=f"train {progress}",
        )

        val_metrics = evaluate(
            trained_module,
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
        score = operating_score(val_metrics.recall, val_metrics.fp_per_tp)

        log_epoch(epoch, train_losses, val_metrics)
        append_metrics(metrics_path, epoch, train_losses, val_metrics, learning_rate, score)
        tracker.consider(epoch, model, val_metrics, score)

        is_last = epoch + 1 == EPOCHS
        if (epoch + 1) % DETAIL_EVERY == 0 or is_last:
            log_detail(val_metrics, labels)

    if tracker.best_metrics is None:
        raise RuntimeError("No se completó ninguna época.")
    logger.info(
        "mejor recall_agn@%.2f de validación: %.3f (FP/TP=%.2f, score=%.3f) -> %s",
        METRIC_IOU_THRESHOLD,
        tracker.best_metrics.recall,
        tracker.best_metrics.fp_per_tp,
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

    labels, meta, train_dataset, val_dataset = load_datasets()
    (run_dir / "config.json").write_text(
        json.dumps(training_config(), indent=2, ensure_ascii=False, default=str)
    )
    train(labels, meta, train_dataset, val_dataset, device, run_dir / "metrics.jsonl")
