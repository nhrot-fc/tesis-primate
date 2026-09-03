import argparse
import json
import logging
from pathlib import Path

from core.config import settings
from core.runtime import resolve_device, setup_logging
from data import cache
from data.datasets import SpectrogramDataset, make_loader
from evaluation.evaluator import evaluate
from evaluation.metrics import MATCH_IOU
from evaluation.report import format_report
from models.registry import load_checkpoint
from training.checkpoint import BEST

logger = logging.getLogger("evaluate")

BATCH_SIZE = 16
IOU_THRESHOLD = MATCH_IOU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Métricas de un checkpoint sobre un split del caché."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help=f"nombre de una corrida en runs/; usa su {BEST}")
    source.add_argument("--checkpoint", type=Path, help="ruta a un .pt/.pth cualquiera")
    parser.add_argument("--split", choices=cache.SPLITS, default="test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--score-threshold", type=float, default=None, help="por defecto, el del checkpoint"
    )
    parser.add_argument(
        "--nms-iou", type=float, default=None, help="por defecto, el del checkpoint"
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="techo de cajas por ventana; sin esto, el que ponga cada framework",
    )
    parser.add_argument("--limit", type=int, default=None, help="usa sólo las primeras N ventanas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    device = resolve_device(args.device)

    path = args.checkpoint or settings.runs_dir / args.run / BEST
    loaded = load_checkpoint(path, device)
    score_threshold = args.score_threshold or loaded.score_threshold
    nms_iou = args.nms_iou if args.nms_iou is not None else loaded.nms_iou

    dataset = SpectrogramDataset(cache.split_path(args.split))
    if args.limit:
        dataset.images = dataset.images[: args.limit]
    loader = make_loader(dataset, args.batch_size)

    metrics = evaluate(
        loaded.detect,
        loader,
        n_classes=len(loaded.labels),
        device=device,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=score_threshold,
        nms_iou=nms_iou,
        max_detections=args.max_detections,
        desc=args.split,
    )

    report = format_report(
        f"{path} ({loaded.architecture}) | split {args.split}: {len(dataset)} ventanas",
        loaded.labels.names,
        metrics,
        score_threshold,
        IOU_THRESHOLD,
    )
    output = path.with_name(f"{path.stem}_{args.split}_metrics")
    output.with_suffix(".txt").write_text(report)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(path),
                "architecture": loaded.architecture,
                "split": args.split,
                "score_threshold": score_threshold,
                "nms_iou": nms_iou,
                "max_detections": args.max_detections,
                "metrics": metrics._asdict(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("\n%s", report)
    logger.info("métricas -> %s", output.with_suffix(".txt"))


if __name__ == "__main__":
    main()
