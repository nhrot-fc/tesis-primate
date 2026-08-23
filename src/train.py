import argparse
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
from torch.utils.data import Dataset, Subset

from architectures.criterion import HungarianMatcher, SetCriterion
from architectures.deformable_detr import ASTDeformableDETR
from architectures.registry import save_checkpoint, save_labels_json
from core.config import settings
from core.setup import setup_logging, setup_project_path
from domain.dataset import BoxJitter, CachedCallBoxDataset
from domain.species import LabelSet
from pipelines.common import Losses, format_metric, make_loader
from pipelines.evaluation_pipeline import EvalMetrics, evaluate
from pipelines.training_pipeline import train_one_epoch

logger = logging.getLogger("training")

ARCHITECTURE = "ast_deformable_detr"

PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "data" / "processed"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

# --- Preprocesado ---------------------------------------------------------------
SEED = 42

# --- Entrenamiento ------------------------------------------------------------
MODEL_DIM, N_QUERIES, N_LEVELS = 128, 64, 3
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS = 30, 16, 2e-4, 1e-4, 0
DETAIL_EVERY = 10
DEVICE_INDEX = 2

# --- Ablaciones (docs/experimentos_ablacion.md) -------------------------------
# Los tres factores que se barren; estos son los valores del baseline.
FRONTEND = "pcen"  # "pcen" o "logmel"
TIME_STRIDE = 2  # paso temporal del parcheo del AST: 2, 5 o 10
FREEZE_BACKBONE = True

# --- Punto de operación y métricas --------------------------------------------
BOX_JITTER = BoxJitter(scale=0.15, shift=0.10, min_size=0.02)
METRIC_IOU_THRESHOLD = 0.5
OPERATING_SCORE_THRESHOLD = 0.5
NMS_IOU = 0.3
CHECKPOINT_SELECTION_BETA = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el AST + Deformable-DETR.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None, help="'cuda', 'cuda:1', 'cpu'")
    # Los tres factores de `docs/experimentos_ablacion.md`. Cambiar uno solo por corrida.
    parser.add_argument(
        "--frontend",
        choices=("pcen", "logmel"),
        default=FRONTEND,
        help="entrada al AST: PCEN entrenable o compresión logarítmica fija",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=TIME_STRIDE,
        help="paso temporal del parcheo del AST; menos paso, más tokens y más VRAM",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="fine-tunea el AST entero en vez de dejarlo congelado",
    )
    parser.add_argument("--limit", type=int, default=None, help="usa sólo N ventanas (pruebas)")
    parser.add_argument("--name", default=None, help="nombre de la corrida y del checkpoint")
    return parser.parse_args()


def run_name(args: argparse.Namespace) -> str:
    """Identifica la corrida por su ablación: dos combinaciones no se pisan el checkpoint."""
    if args.name:
        return args.name
    state = "ft" if args.unfreeze else "frozen"
    return f"detr_{args.frontend}_ts{args.time_stride}_{state}"


def training_config(args: argparse.Namespace) -> dict[str, object]:
    dataset_meta = json.loads((CACHE_DIR / "meta.json").read_text())
    return {
        "seed": SEED,
        "dataset": dataset_meta,
        "architecture": ARCHITECTURE,
        "model_dim": MODEL_DIM,
        "n_queries": N_QUERIES,
        "n_levels": N_LEVELS,
        "frontend": args.frontend,
        "time_stride": args.time_stride,
        "freeze_backbone": not args.unfreeze,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "box_jitter": asdict(BOX_JITTER),
        "metric_iou_threshold": METRIC_IOU_THRESHOLD,
        "operating_score_threshold": OPERATING_SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "checkpoint_selection_beta": CHECKPOINT_SELECTION_BETA,
    }


def operating_score(
    recall: float | None, precision: float | None, beta: float = CHECKPOINT_SELECTION_BETA
) -> float:
    if recall is None or precision is None:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)


def load_datasets() -> tuple[LabelSet, dict, CachedCallBoxDataset, CachedCallBoxDataset]:
    if not (CACHE_DIR / "meta.json").exists():
        raise FileNotFoundError(
            f"no hay dataset cacheado en {CACHE_DIR}. Corré `python src/create_dataset.py` primero."
        )
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
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


def format_top_confusions(confusion: Tensor, names: list[str], top_k: int = 5) -> str:
    off = confusion.clone()
    off.fill_diagonal_(0)
    off[:, -1] = 0
    flat = off.flatten()
    k = min(top_k, int((flat > 0).sum()))
    if k == 0:
        return "  (ninguna)"
    counts, indices = flat.topk(k)
    lines = []
    for count, index in zip(counts.tolist(), indices.tolist(), strict=False):
        i, j = divmod(index, off.shape[1])
        lines.append(f"  {names[i]} -> {names[j]}: {int(count)}")
    return "\n".join(lines)


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


def model_hparams(args: argparse.Namespace) -> dict[str, object]:
    """Lo que hace falta para rearmar el grafo; los pesos entran por el `state_dict`.

    La geometría no viaja en los pesos (el pos-embed del AST se re-interpola al
    construir) y la ablación tampoco: sin `frontend` ni `freeze` el checkpoint se
    reconstruye con otra entrada y otra lista de parámetros.
    """
    return {
        "dim": MODEL_DIM,
        "n_queries": N_QUERIES,
        "n_levels": N_LEVELS,
        "time_stride": args.time_stride,
        "frontend": args.frontend,
        "freeze": not args.unfreeze,
    }


def build_model(
    n_classes: int, steps_per_epoch: int, device: str, args: argparse.Namespace
) -> TrainingComponents:
    model = ASTDeformableDETR(n_classes=n_classes, **model_hparams(args)).to(device)  # type: ignore
    matcher = HungarianMatcher()
    criterion = SetCriterion(n_classes=n_classes, matcher=matcher).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.05,
        anneal_strategy="cos",
    )
    logger.info(
        "%.1fM parámetros (%.1fM entrenables) | %d clases | frontend=%s time_stride=%d "
        "backbone=%s | %d x %d tokens",
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in trainable) / 1e6,
        n_classes,
        args.frontend,
        args.time_stride,
        "congelado" if model.backbone.freeze else "fine-tune",
        model.backbone.freq_out,
        model.backbone.time_out,
    )
    return TrainingComponents(model, matcher, criterion, optimizer, scheduler)


def log_epoch(
    epoch: int, epochs: int, train_losses: Losses, val_metrics: EvalMetrics, score: float
) -> None:
    logger.info(
        "[%4d/%d] train=%.3f val=%.3f cls_acc=%.3f IoU=%.3f recall_agn@%.2f=%s "
        "precision_agn=%s mAP50=%s mAP50-95=%s score=%.3f",
        epoch + 1,
        epochs,
        train_losses.total,
        val_metrics.losses.total,
        val_metrics.accuracy,
        val_metrics.mean_iou,
        METRIC_IOU_THRESHOLD,
        format_metric(val_metrics.recall_agnostic),
        format_metric(val_metrics.precision_agnostic),
        format_metric(val_metrics.map_50),
        format_metric(val_metrics.map_50_95),
        score,
    )


def log_detail(val_metrics: EvalMetrics, labels: LabelSet) -> None:
    logger.info(
        "Recall por clase -> %s",
        format_recall_per_class(val_metrics.recall_per_class, labels.names),
    )
    logger.info(
        "AP agnóstico de clase -> %s",
        ", ".join(
            f"{threshold}={format_metric(ap)}"
            for threshold, ap in sorted(val_metrics.ap_agnostic.items())
        ),
    )
    logger.info(
        "AP por clase @0.5 -> %s",
        format_recall_per_class(val_metrics.ap_per_class_50, labels.names),
    )
    logger.info(
        "Confusiones más frecuentes (queries emparejadas):\n%s",
        format_top_confusions(val_metrics.confusion, labels.names),
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
            "recall_agnostic": val_metrics.recall_agnostic,
            "precision_agnostic": val_metrics.precision_agnostic,
            "operating_score": score,
            "map_50": val_metrics.map_50,
            "map_50_95": val_metrics.map_50_95,
            "ap_agnostic": {
                str(threshold): ap for threshold, ap in sorted(val_metrics.ap_agnostic.items())
            },
            "map_per_threshold": {
                str(threshold): value
                for threshold, value in sorted(val_metrics.map_per_threshold.items())
            },
            "ap_per_class_50": val_metrics.ap_per_class_50,
            "recall_per_class": val_metrics.recall_per_class,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class BestTracker:
    def __init__(
        self,
        checkpoint_path: Path,
        labels: LabelSet,
        config: dict[str, object],
        hparams: dict[str, object],
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.labels = labels
        self.config = config
        self.hparams = hparams
        self.best_score = float("-inf")
        self.best_metrics: EvalMetrics | None = None

    def consider(
        self, epoch: int, model: ASTDeformableDETR, val_metrics: EvalMetrics, score: float
    ) -> None:
        if score <= self.best_score:
            return
        self.best_score, self.best_metrics = score, val_metrics
        save_checkpoint(
            self.checkpoint_path,
            architecture=ARCHITECTURE,
            model=model,
            hparams={**self.hparams, "n_frames": model.backbone.n_frames},
            labels=self.labels,
            config=self.config,
            epoch=epoch,
            recall_agn=val_metrics.recall_agnostic,
            precision_agn=val_metrics.precision_agnostic,
            map_50=val_metrics.map_50,
            map_50_95=val_metrics.map_50_95,
        )
        logger.info("Nuevo mejor score=%.3f -> %s", score, self.checkpoint_path)


def train(
    labels: LabelSet,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str,
    metrics_path: Path,
    args: argparse.Namespace,
) -> None:
    torch.manual_seed(SEED)
    n_classes = len(labels)

    train_loader = make_loader(train_dataset, args.batch, args.workers, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch, args.workers, shuffle=False)

    model, matcher, criterion, optimizer, scheduler = build_model(
        n_classes, len(train_loader), device, args
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = f"{run_name(args)}_{n_classes}cls"
    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"
    labels_path = CHECKPOINT_DIR / f"{name}_labels.json"

    save_labels_json(labels, labels_path)
    logger.info(
        "class_id -> label -> %s",
        {class_id: label for class_id, label in enumerate(labels.names)},
    )

    tracker = BestTracker(checkpoint_path, labels, training_config(args), model_hparams(args))

    for epoch in range(args.epochs):
        progress = f"{epoch + 1}/{args.epochs}"
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

        is_last = epoch + 1 == args.epochs
        detailed = (epoch + 1) % DETAIL_EVERY == 0 or is_last
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
            detailed=detailed,
            desc=f"val {progress}",
        )
        score = operating_score(val_metrics.recall_agnostic, val_metrics.precision_agnostic)

        log_epoch(epoch, args.epochs, train_losses, val_metrics, score)
        append_metrics(metrics_path, epoch, train_losses, val_metrics, learning_rate, score)
        tracker.consider(epoch, model, val_metrics, score)

        if detailed:
            log_detail(val_metrics, labels)

    if tracker.best_metrics is None:
        raise RuntimeError("No se completó ninguna época.")
    logger.info(
        "mejor recall_agn@%.2f de validación: %s (precision_agn=%s, mAP50=%s, mAP50-95=%s, "
        "score=%.3f) -> %s",
        METRIC_IOU_THRESHOLD,
        format_metric(tracker.best_metrics.recall_agnostic),
        format_metric(tracker.best_metrics.precision_agnostic),
        format_metric(tracker.best_metrics.map_50),
        format_metric(tracker.best_metrics.map_50_95),
        tracker.best_score,
        checkpoint_path,
    )
    logger.info("métricas por época -> %s", metrics_path)
    logger.info(
        "evaluá con: python src/eval_detector.py --checkpoint %s --split test", checkpoint_path
    )


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.set_device(DEVICE_INDEX)
    return f"cuda:{DEVICE_INDEX}"


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"{run_name(args)}_{stamp}"
    setup_logging(settings.LOG_LEVEL, log_file=run_dir / "train.log")
    setup_project_path(PROJECT_DIR)

    device = resolve_device(args.device)
    logger.info("device: %s", device)

    labels, _meta, cached_train, cached_val = load_datasets()
    train_dataset: Dataset = cached_train
    val_dataset: Dataset = cached_val
    if args.limit:
        train_dataset = Subset(cached_train, range(min(args.limit, len(cached_train))))
        val_dataset = Subset(cached_val, range(min(args.limit, len(cached_val))))

    (run_dir / "config.json").write_text(
        json.dumps(training_config(args), indent=2, ensure_ascii=False, default=str)
    )
    train(labels, train_dataset, val_dataset, device, run_dir / "metrics.jsonl", args)


if __name__ == "__main__":
    main()
