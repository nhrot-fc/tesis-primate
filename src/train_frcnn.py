"""Entrena un Faster R-CNN sobre las mismas ventanas cacheadas que el Deformable-DETR.

Torchvision calcula sus pérdidas adentro del modelo, así que no hay criterio ni matcher.
El checkpoint se elige con el mismo `operating_score` (beta=3) que `src/train.py`.
"""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from architectures.faster_rcnn import (
    ANCHOR_RATIOS,
    MAX_SIZE,
    MIN_SIZE,
    SpectrogramFasterRCNN,
    detect,
    to_torchvision_targets,
)
from architectures.registry import save_checkpoint, save_labels_json
from core.config import settings
from core.setup import setup_logging, setup_project_path
from pipelines.common import format_metric, make_loader, to_device
from pipelines.detection_pipeline import collect_detections
from pipelines.metrics import detection_metrics
from train import (
    BOX_JITTER,
    CACHE_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    METRIC_IOU_THRESHOLD,
    NMS_IOU,
    OPERATING_SCORE_THRESHOLD,
    SEED,
    format_recall_per_class,
    load_datasets,
    operating_score,
)
from utils.audio import mel_db_range

logger = logging.getLogger("train_frcnn")

ARCHITECTURE = "faster_rcnn"
PROJECT_DIR = Path.cwd()

EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS = 30, 8, 1e-4, 1e-4, 4
TRAINABLE_BACKBONE_LAYERS = 3  # de 5; congelar las primeras ahorra memoria y sobreajuste
CLIP_GRAD = 10.0
DETAIL_EVERY = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena Faster R-CNN sobre el mel.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None, help="'cuda', 'cuda:1', 'cpu'")
    parser.add_argument(
        "--scratch", action="store_true", help="entrena desde cero, sin los pesos de COCO"
    )
    parser.add_argument("--limit", type=int, default=None, help="usa sólo N ventanas (pruebas)")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    device: str,
    desc: str,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}

    progress = tqdm(loader, desc=desc, unit="batch", leave=False)
    for step, batch in enumerate(progress, start=1):
        images, targets = to_device(batch, device)
        losses: dict[str, torch.Tensor] = model(images, to_torchvision_targets(targets))
        total = torch.stack(list(losses.values())).sum()

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        scheduler.step()

        totals["total"] = totals.get("total", 0.0) + total.item()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.item()
        progress.set_postfix(loss=totals["total"] / step, lr=optimizer.param_groups[0]["lr"])

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def training_config(args: argparse.Namespace, hparams: dict) -> dict:
    return {
        "seed": SEED,
        "architecture": ARCHITECTURE,
        "pretrained": not args.scratch,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "box_jitter": asdict(BOX_JITTER),
        "metric_iou_threshold": METRIC_IOU_THRESHOLD,
        "operating_score_threshold": OPERATING_SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "hparams": hparams,
        "dataset": json.loads((CACHE_DIR / "meta.json").read_text()),
    }


def train(args: argparse.Namespace, device: str, run_dir: Path) -> None:
    torch.manual_seed(SEED)
    labels, _meta, cached_train, cached_val = load_datasets()

    # El rango se estima sobre el train completo, no sobre el recorte de `--limit`:
    # es parte de la representación de la entrada y viaja en el checkpoint.
    db_low, db_high = mel_db_range(cached_train.images[:, 0])
    logger.info("rango de dB del mel (train) -> [%.2f, %.2f]", db_low, db_high)

    train_dataset: Dataset = cached_train
    val_dataset: Dataset = cached_val
    if args.limit:
        train_dataset = Subset(cached_train, range(min(args.limit, len(cached_train))))
        val_dataset = Subset(cached_val, range(min(args.limit, len(cached_val))))

    train_loader = make_loader(train_dataset, args.batch, args.workers, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch, args.workers, shuffle=False)

    # lo mínimo para reconstruir el grafo; los pesos entran por el `state_dict`
    hparams: dict[str, Any] = {
        "db_low": db_low,
        "db_high": db_high,
        "min_size": MIN_SIZE,
        "max_size": MAX_SIZE,
        "anchor_ratios": ANCHOR_RATIOS,
        "trainable_layers": TRAINABLE_BACKBONE_LAYERS,
    }
    model = SpectrogramFasterRCNN(n_classes=len(labels), pretrained=not args.scratch, **hparams).to(
        device
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "%.1fM parámetros (%.1fM entrenables) | %d clases | anchors %s",
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in trainable) / 1e6,
        len(labels),
        ANCHOR_RATIOS,
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = args.name or f"frcnn_{'scratch' if args.scratch else 'coco'}_{len(labels)}cls"
    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"
    save_labels_json(labels, CHECKPOINT_DIR / f"{name}_labels.json")

    config = training_config(args, hparams)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str)
    )
    metrics_path = run_dir / "metrics.jsonl"

    best_score = float("-inf")
    for epoch in range(args.epochs):
        progress = f"{epoch + 1}/{args.epochs}"
        learning_rate = optimizer.param_groups[0]["lr"]
        losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, desc=f"train {progress}"
        )

        detailed = (epoch + 1) % DETAIL_EVERY == 0 or epoch + 1 == args.epochs
        predictions, truth = collect_detections(
            lambda batch, threshold: detect(model, batch, threshold),
            val_loader,
            device,
            nms_iou=NMS_IOU,
            desc=f"val {progress}",
        )
        metrics = detection_metrics(
            predictions,
            truth,
            n_classes=len(labels),
            iou_threshold=METRIC_IOU_THRESHOLD,
            score_threshold=OPERATING_SCORE_THRESHOLD,
            detailed=detailed,
        )
        score = operating_score(metrics.recall_agnostic, metrics.precision_agnostic)

        logger.info(
            "[%4d/%d] train=%.3f recall_agn@%.2f=%s precision_agn=%s score=%.3f",
            epoch + 1,
            args.epochs,
            losses["total"],
            METRIC_IOU_THRESHOLD,
            format_metric(metrics.recall_agnostic),
            format_metric(metrics.precision_agnostic),
            score,
        )
        if detailed:
            logger.info(
                "Recall por clase -> %s",
                format_recall_per_class(metrics.recall_per_class, labels.names),
            )

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "lr": learning_rate,
                        "train": losses,
                        "val": {
                            **metrics._asdict(),
                            "ap_agnostic": {str(k): v for k, v in metrics.ap_agnostic.items()},
                            "operating_score": score,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if score > best_score:
            best_score = score
            save_checkpoint(
                checkpoint_path,
                architecture=ARCHITECTURE,
                model=model,
                hparams=hparams,
                labels=labels,
                config=config,
                epoch=epoch,
                recall_agn=metrics.recall_agnostic,
                precision_agn=metrics.precision_agnostic,
            )
            logger.info("Nuevo mejor score=%.3f -> %s", score, checkpoint_path)

    logger.info("mejor operating_score de validación: %.3f -> %s", best_score, checkpoint_path)
    logger.info(
        "evaluá con: python src/eval_detector.py --checkpoint %s --split test", checkpoint_path
    )


def main() -> None:
    args = parse_args()
    run_dir = LOG_DIR / f"frcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    setup_logging(settings.LOG_LEVEL, log_file=run_dir / "train.log")
    setup_project_path(PROJECT_DIR)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    train(args, device, run_dir)


if __name__ == "__main__":
    main()
