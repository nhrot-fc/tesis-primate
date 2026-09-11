import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.runtime import setup_logging
from evaluation.evaluator import TEST, VAL, RawPredictions, load_pairs, path_for
from evaluation.metrics import MATCH_IOU, Boxes, hits, overlaps, rows_by_image, sort_by_score

logger = logging.getLogger("fuse_predictions")

CLUSTER_IOU = 0.55  # el de Solovyev et al. 2021 (WBF)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusiona los volcados de varios modelos (WBF) en uno nuevo que `compare_models.py` "
        "mide como a cualquier otro. Antes calibra el score de cada modelo a su precisión en val: "
        "sin eso, el 0.22 de un DETR y el 0.02 de un YOLO no son comparables."
    )
    parser.add_argument("dumps", nargs="+", type=Path, help="los *_predictions.pt, val y test")
    parser.add_argument("--name", required=True, help="con el que entra a la tabla")
    parser.add_argument("--output", type=Path, required=True, help="directorio")
    parser.add_argument("--iou", type=float, default=CLUSTER_IOU, help="para agrupar cajas")
    return parser.parse_args()


def isotonic(scores: Tensor, hit: Tensor) -> tuple[np.ndarray, np.ndarray]:
    # Pool-adjacent-violators: precisión no decreciente en el score, ajustada en val.
    order = scores.argsort()
    x = scores[order].numpy().astype(np.float64)
    y = hit[order].numpy().astype(np.float64)
    values, weights, starts = [], [], []
    for i in range(len(y)):
        values.append(y[i])
        weights.append(1.0)
        starts.append(i)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            weights[-2] = w
            values.pop()
            weights.pop()
            starts.pop()
    return x[np.array(starts)], np.array(values)


def calibrate(model: RawPredictions) -> tuple[np.ndarray, np.ndarray]:
    found = hits(
        overlaps(model.predictions, model.truth, True), len(model.predictions.boxes), MATCH_IOU
    )
    return isotonic(model.predictions.scores, found)


def apply(curve: tuple[np.ndarray, np.ndarray], scores: Tensor) -> Tensor:
    knots, values = curve
    index = np.searchsorted(knots, scores.numpy(), side="right") - 1
    return torch.from_numpy(values[index.clip(0)]).float()


def fuse_window(
    boxes: Tensor, scores: Tensor, labels: Tensor, members: Tensor, n_models: int, iou: float
) -> tuple[Tensor, Tensor, Tensor]:
    # Agrupa por clase e IoU, promedia las cajas por score; score = 1 - ∏(1 - p) sobre los modelos.
    order = scores.argsort(descending=True)
    boxes, scores, labels, members = boxes[order], scores[order], labels[order], members[order]
    xyxy = box_convert(boxes, "cxcywh", "xyxy")
    clusters: list[list[int]] = []
    fused: list[Tensor] = []
    for i in range(len(boxes)):
        for c, rows in enumerate(clusters):
            if labels[rows[0]] == labels[i] and box_iou(fused[c][None], xyxy[i][None]) >= iou:
                rows.append(i)
                w = scores[rows][:, None]
                fused[c] = (xyxy[rows] * w).sum(0) / w.sum()
                break
        else:
            clusters.append([i])
            fused.append(xyxy[i].clone())
    out_boxes, out_scores, out_labels = [], [], []
    for c, rows in enumerate(clusters):
        rows_t = torch.tensor(rows)
        best = torch.zeros(n_models)
        for m in range(n_models):
            mine = scores[rows_t][members[rows_t] == m]
            if len(mine):
                best[m] = mine.max()
        score = 1.0 - torch.prod(1.0 - best)
        out_boxes.append(box_convert(fused[c][None], "xyxy", "cxcywh")[0])
        out_scores.append(score)
        out_labels.append(labels[rows[0]])
    if not out_boxes:
        return boxes[:0], scores[:0], labels[:0]
    return torch.stack(out_boxes), torch.stack(out_scores), torch.stack(out_labels)


def fuse(models: list[RawPredictions], curves: list, iou: float, name: str) -> RawPredictions:
    calibrated = [
        Boxes(
            m.predictions.boxes,
            m.predictions.image_ids,
            m.predictions.labels,
            apply(c, m.predictions.scores),
        )
        for m, c in zip(models, curves, strict=True)
    ]
    members = torch.cat([torch.full((len(b.boxes),), k) for k, b in enumerate(calibrated)])
    pool = Boxes(*(torch.cat([getattr(b, f) for b in calibrated]) for f in Boxes._fields))
    out: list[Boxes] = []
    for image_id, rows in rows_by_image(pool.image_ids).items():
        rows_t = torch.tensor(rows)
        boxes, scores, labels = fuse_window(
            pool.boxes[rows_t],
            pool.scores[rows_t],
            pool.labels[rows_t],
            members[rows_t],
            len(models),
            iou,
        )
        out.append(Boxes(boxes, torch.full((len(boxes),), image_id), labels, scores))
    merged = sort_by_score(Boxes(*(torch.cat([getattr(b, f) for b in out]) for f in Boxes._fields)))
    return models[0]._replace(model=name, predictions=merged)


def main() -> None:
    args = parse_args()
    setup_logging()
    pairs = load_pairs(args.dumps)
    curves = [calibrate(val) for val, _ in pairs]
    for (val, _), (knots, values) in zip(pairs, curves, strict=True):
        logger.info(
            "%s: calibración isotónica con %d tramos, p en [%.3f, %.3f]",
            val.model,
            len(knots),
            values.min(),
            values.max(),
        )
    for split, index in ((VAL, 0), (TEST, 1)):
        fused = fuse([p[index] for p in pairs], curves, args.iou, args.name)
        out = fused.save(path_for(args.output, args.name, split))
        logger.info("%s %s: %d cajas -> %s", args.name, split, len(fused.predictions.boxes), out)


if __name__ == "__main__":
    main()
