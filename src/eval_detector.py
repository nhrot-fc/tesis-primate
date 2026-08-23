"""Métricas de un checkpoint --de la arquitectura que sea-- sobre un split del caché.

Los tres detectores se cargan por el registro, corren sobre las mismas ventanas de
`data/processed/*.pt` y se miden con el mismo código, así que sus números son
comparables. `src/eval.py` sigue siendo el informe detallado del DETR (pérdidas,
accuracy y matriz de confusión del matcher), que no tiene equivalente con anchors.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from architectures.registry import load_checkpoint
from core.setup import setup_logging, setup_project_path
from domain.dataset import CachedCallBoxDataset, collate_fn
from pipelines.common import format_metric
from pipelines.detection_pipeline import collect_detections
from pipelines.metrics import AP_THRESHOLDS, DetectionMetrics, detection_metrics
from train import CACHE_DIR, CHECKPOINT_SELECTION_BETA, operating_score

logger = logging.getLogger("eval_detector")

PROJECT_DIR = Path.cwd()

BATCH_SIZE = 16
IOU_THRESHOLD = 0.5


def format_report(
    checkpoint: Path,
    architecture: str,
    split: str,
    n_windows: int,
    names: list[str],
    metrics: DetectionMetrics,
    score_threshold: float,
) -> str:
    score = operating_score(metrics.recall_agnostic, metrics.precision_agnostic)
    ap_values = [ap for ap in metrics.ap_agnostic.values() if ap is not None]
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else None

    lines = [
        f"checkpoint: {checkpoint}",
        f"arquitectura: {architecture}",
        f"split: {split} ({n_windows} ventanas, {metrics.n_gt} cajas anotadas)",
        f"detecciones: {metrics.n_predictions} (con score >= {score_threshold}: "
        f"{metrics.n_above_threshold})",
        "",
        f"Punto de operación (score >= {score_threshold}, IoU >= {IOU_THRESHOLD}):",
        f"  recall_agn={format_metric(metrics.recall_agnostic)} "
        f"precision_agn={format_metric(metrics.precision_agnostic)} "
        f"operating_score(beta={CHECKPOINT_SELECTION_BETA:g})={score:.3f}",
        "  AP agnóstico de clase -> "
        + ", ".join(
            f"{threshold}={format_metric(ap)}"
            for threshold, ap in sorted(metrics.ap_agnostic.items())
        )
        + f" | medio sobre umbrales={format_metric(mean_ap)}",
        "",
        "mAP por clase (independiente del punto de operación):",
        f"  mAP@0.5={format_metric(metrics.map_50)} "
        f"mAP@0.5:0.95={format_metric(metrics.map_50_95)}",
        "  por umbral de IoU -> "
        + ", ".join(
            f"{threshold:g}={format_metric(value)}"
            for threshold, value in sorted(metrics.map_per_threshold.items())
        ),
        "",
        "Por clase (recall al punto de operación, AP sobre toda la curva):",
        f"{'clase':<20}{'recall':>10}{'AP@0.5':>10}",
    ]
    lines += [
        f"{name:<20}{format_metric(metrics.recall_per_class.get(class_id)):>10}"
        f"{format_metric(metrics.ap_per_class_50.get(class_id)):>10}"
        for class_id, name in enumerate(names)
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas de un checkpoint sobre un split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--score-threshold", type=float, default=None, help="por defecto, el del checkpoint"
    )
    parser.add_argument(
        "--nms-iou", type=float, default=None, help="por defecto, el del checkpoint"
    )
    parser.add_argument("--limit", type=int, default=None, help="usa sólo las primeras N ventanas")
    parser.add_argument(
        "--output", type=Path, default=None, help="por defecto, junto al checkpoint"
    )
    args = parser.parse_args()

    setup_logging()
    setup_project_path(PROJECT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = load_checkpoint(args.checkpoint, device)
    score_threshold = (
        args.score_threshold if args.score_threshold is not None else loaded.score_threshold
    )
    nms_iou = args.nms_iou if args.nms_iou is not None else loaded.nms_iou

    dataset = CachedCallBoxDataset(CACHE_DIR / f"{args.split}.pt")
    subset = (
        torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
        if args.limit
        else dataset
    )
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    predictions, truth = collect_detections(loaded.detect, loader, device, nms_iou)
    metrics = detection_metrics(
        predictions,
        truth,
        n_classes=len(loaded.labels),
        iou_threshold=IOU_THRESHOLD,
        score_threshold=score_threshold,
        ap_thresholds=AP_THRESHOLDS,
    )

    report = format_report(
        args.checkpoint,
        loaded.architecture,
        args.split,
        len(subset),
        loaded.labels.names,
        metrics,
        score_threshold,
    )
    output_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_{args.split}_metrics.txt"
    )
    output_path.write_text(report)
    (output_path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "architecture": loaded.architecture,
                "split": args.split,
                "score_threshold": score_threshold,
                "nms_iou": nms_iou,
                "metrics": {
                    **metrics._asdict(),
                    "ap_agnostic": {str(k): v for k, v in metrics.ap_agnostic.items()},
                    "map_per_threshold": {str(k): v for k, v in metrics.map_per_threshold.items()},
                    "operating_score": operating_score(
                        metrics.recall_agnostic, metrics.precision_agnostic
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("\n%s", report)
    logger.info("métricas -> %s", output_path)


if __name__ == "__main__":
    main()
