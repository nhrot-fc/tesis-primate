import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from architectures.criterion import HungarianMatcher, SetCriterion
from core.setup import setup_logging, setup_project_path
from domain.dataset import CachedCallBoxDataset, collate_fn
from infer import load_model
from pipelines.training_pipeline import EvalMetrics, evaluate, format_metric
from train import CACHE_DIR, format_confusion, operating_score

logger = logging.getLogger("eval")

PROJECT_DIR = Path.cwd()
BATCH_SIZE = 64


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def format_class_table(metrics: EvalMetrics, labels_names: list[str]) -> str:
    header = f"{'clase':<20}{'recall':>10}{'precisión':>12}{'F1':>10}"
    rows = [header]
    for class_id, name in enumerate(labels_names):
        recall = metrics.recall_per_class.get(class_id)
        precision = metrics.precision_per_class.get(class_id)
        f1 = _f1(precision, recall)
        rows.append(
            f"{name:<20}{format_metric(recall):>10}"
            f"{format_metric(precision):>12}{format_metric(f1):>10}"
        )
    return "\n".join(rows)


def format_report(
    checkpoint: Path, split: str, n_windows: int, labels_names: list[str], metrics: EvalMetrics
) -> str:
    score = operating_score(metrics.recall_agnostic, metrics.fp_per_tp_agnostic)
    f1_agn = _f1(metrics.precision_agnostic, metrics.recall_agnostic)
    ap_values = [ap for ap in metrics.ap_agnostic.values() if ap is not None]
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else None
    lines = [
        f"checkpoint: {checkpoint}",
        f"split: {split} ({n_windows} ventanas)",
        "",
        f"loss total={metrics.losses.total:.3f} cls={metrics.losses.cls:.3f} "
        f"bbox={metrics.losses.bbox:.3f} iou={metrics.losses.iou:.3f}",
        f"accuracy (clase, queries emparejadas)={metrics.accuracy:.3f}",
        f"IoU medio (queries emparejadas)={metrics.mean_iou:.3f}",
        f"recall_agn={format_metric(metrics.recall_agnostic)} "
        f"precision_agn={format_metric(metrics.precision_agnostic)} "
        f"F1_agn={format_metric(f1_agn)} "
        f"FP/TP={format_metric(metrics.fp_per_tp_agnostic, digits=2)} "
        f"operating_score={score:.3f}",
        "AP agnóstico de clase -> "
        + ", ".join(
            f"{threshold}={format_metric(ap)}"
            for threshold, ap in sorted(metrics.ap_agnostic.items())
        )
        + f" | AP_agn medio sobre umbrales={format_metric(mean_ap)}",
        "",
        "Métricas por clase (punto de operación):",
        format_class_table(metrics, labels_names),
        "",
        "Matriz de confusión (queries emparejadas por el matcher húngaro; filas=clase "
        "real, columnas=clase predicha, '∅'=no-objeto):",
        format_confusion(metrics.confusion, labels_names),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas de un checkpoint sobre un split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--output", type=Path, default=None, help="por defecto, junto al checkpoint"
    )
    args = parser.parse_args()

    setup_logging()
    setup_project_path(PROJECT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = load_model(args.checkpoint, device)
    config = loaded.config

    dataset = CachedCallBoxDataset(CACHE_DIR / f"{args.split}.pt")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )

    matcher = HungarianMatcher()
    criterion = SetCriterion(n_classes=len(loaded.labels), matcher=matcher).to(device)

    metrics = evaluate(
        loaded.model,
        loader,
        criterion,
        matcher,
        device,
        n_classes=len(loaded.labels),
        iou_threshold=config.get("metric_iou_threshold", 0.5),
        score_threshold=loaded.score_threshold,
        nms_iou=loaded.nms_iou,
        desc=args.split,
    )

    report = format_report(args.checkpoint, args.split, len(dataset), loaded.labels.names, metrics)
    output_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_{args.split}_metrics.txt"
    )
    output_path.write_text(report)
    logger.info("\n%s", report)
    logger.info("métricas -> %s", output_path)


if __name__ == "__main__":
    main()
