import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from core.config import settings
from core.runtime import setup_logging
from evaluation.evaluator import TEST, VAL, RawPredictions, path_for
from evaluation.metrics import (
    BETA,
    COCO_THRESHOLDS,
    MATCH_IOU,
    Boxes,
    average_precision_per_class,
    concat,
    detection_classes,
    mean_average_precision,
    mean_average_precision_over,
    sort_by_score,
)
from evaluation.protocol import (
    MAX_DETECTIONS,
    TARGET_PRECISION,
    THRESHOLD_GRID,
    Criterion,
    equalize,
    match,
    max_recall_at_precision,
    operating_point,
    select,
)
from utils.boxes import suppress_nested

logger = logging.getLogger("sweep_nms")

Fila = dict[str, float | None]  # una celda de la rejilla, o un punto de su curva

NMS_IOU = (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)  # 1.0 apaga la etapa
IOMIN = (0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="nombre de una corrida en runs/")
    parser.add_argument("--dir", type=Path, help="dónde leer y escribir; por defecto, la corrida")
    parser.add_argument("--nms-iou", nargs="+", type=float, default=list(NMS_IOU))
    parser.add_argument("--iomin", nargs="+", type=float, default=list(IOMIN))
    return parser.parse_args()


def by_window(boxes: Boxes) -> list[Boxes]:
    """Parte las cajas por ventana: suprimir sólo tiene sentido dentro de una."""
    ordered = boxes.select(boxes.image_ids.to(torch.int64).argsort(stable=True))
    counts = torch.bincount(ordered.image_ids.to(torch.int64)).tolist()
    return [Boxes(*group) for group in zip(*(f.split(counts) for f in ordered), strict=True)]


def suppress(windows: list[Boxes], nms_iou: float, iomin: float) -> Boxes:
    kept = [
        window.select(
            suppress_nested(
                box_convert(window.boxes, "cxcywh", "xyxy"),
                window.scores,
                window.labels,
                nms_iou,
                iomin,
            )
        )
        for window in windows
        if len(window.boxes)
    ]
    return sort_by_score(concat(kept))  # el protocolo asume orden de score global


def evaluate(
    val: RawPredictions,
    test: RawPredictions,
    windows: dict[str, list[Boxes]],
    nms_iou: float,
    iomin: float,
) -> tuple[Fila, list[Fila]]:
    boxes = {
        split: equalize(suppress(windows[split], nms_iou, iomin), MAX_DETECTIONS)
        for split in (VAL, TEST)
    }
    matched_val = match(boxes[VAL], val.truth, val.n_images)
    matched = match(boxes[TEST], test.truth, test.n_images)

    chosen = select(matched_val, Criterion("f_beta", BETA), BETA)  # el umbral sale de val
    point = operating_point(matched, chosen.threshold, BETA) if chosen else None

    ap = average_precision_per_class(boxes[TEST], test.truth, len(test.labels))
    classes = detection_classes(test.truth, len(test.labels))
    combo = {"nms_iou": nms_iou, "iomin": iomin}

    row = combo | {
        "cajas_por_ventana": len(boxes[TEST].boxes) / max(test.n_images, 1),
        "map_30": mean_average_precision(ap[MATCH_IOU], classes),
        "map_50": mean_average_precision(ap[0.5], classes),
        "map_50_95": mean_average_precision_over(ap, COCO_THRESHOLDS, classes),
        "umbral": chosen.threshold if chosen else None,
        "recall_techo": max_recall_at_precision(matched, TARGET_PRECISION),
    }
    if point:
        row |= {
            campo: getattr(point, campo)
            for campo in ("recall", "precision", "f_beta", "fp_per_hour")
        }

    curve = [
        combo | operating_point(matched, round(float(t), 2), BETA)._asdict() for t in THRESHOLD_GRID
    ]
    return row, curve


def load(directory: Path, run: str) -> dict[str, RawPredictions]:
    dumps = {split: RawPredictions.load(path_for(directory, run, split)) for split in (VAL, TEST)}
    for split, dump in dumps.items():
        if dump.nms_iou is not None:
            raise ValueError(
                f"el volcado de {split} ya trae NMS a {dump.nms_iou}; el barrido tiene que "
                "partir de las cajas crudas."
            )
        logger.info(
            "%s %s: %d ventanas, %.1f cajas por ventana",
            run,
            split,
            dump.n_images,
            dump.detections_per_window,
        )
    return dumps


def main() -> None:
    args = parse_args()
    setup_logging()

    directory = args.dir or settings.runs_dir / args.run
    dumps = load(directory, args.run)
    windows = {split: by_window(dump.predictions) for split, dump in dumps.items()}

    rows, curves = [], []
    grid = list(itertools.product(args.nms_iou, args.iomin))
    for nms_iou, iomin in tqdm(grid, desc="rejilla", unit="combo"):
        row, curve = evaluate(dumps[VAL], dumps[TEST], windows, nms_iou, iomin)
        rows.append(row)
        curves.extend(curve)
        logger.info(
            "nms=%.2f iomin=%.2f | umbral=%s F2=%.3f mAP30=%.3f",
            nms_iou,
            iomin,
            row["umbral"],
            row.get("f_beta") or 0.0,
            row["map_30"] or 0.0,
        )

    directory.mkdir(parents=True, exist_ok=True)
    for name, table in (("nms_sweep", rows), ("nms_sweep_curves", curves)):
        path = directory / f"{name}.csv"
        pd.DataFrame(table).to_csv(path, index=False)
        logger.info("escrito %s", path)


if __name__ == "__main__":
    main()
