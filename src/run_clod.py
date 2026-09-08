import argparse
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from clod.cluster import BEGIN, END, HIGH, LOW
from clod.folds import by_group
from clod.issues import CATEGORY, ISSUE, SCORE, rank, scan
from core.config import P, Parameters, settings
from core.runtime import resolve_device, set_seed, setup_logging
from data import cache
from data.datasets import BoxJitter, SpectrogramDataset, make_loader
from data.species import LabelSet
from evaluation.evaluator import collect_detections
from evaluation.metrics import Boxes
from models.registry import build_model, load_checkpoint
from training.checkpoint import BEST
from training.trainer import TrainConfig, Trainer
from utils.audio import y_to_hz

logger = logging.getLogger("clod")

AUDITED_RUN = "detr_pcen_ts10_ft"
N_FOLDS, EPOCHS_PER_FOLD = 5, 14
SCORE_FLOOR, IOU_THRESHOLD, TOP = 0.05, 0.3, 0.05

TRAIN, VAL, TEST = "train", "val", "test"
TRUTH, PREDICTIONS = "truth", "predictions"
RECORDING, WINDOW, BOX = "recording", "window", "box"
FULL_WINDOW_CLASSES = ("as/hc", "pt/dc")


class Detected(NamedTuple):
    truth: Boxes
    predictions: Boxes
    windows: np.ndarray


class Audited(NamedTuple):
    architecture: str
    hparams: dict[str, Any]
    config: TrainConfig
    labels: LabelSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corre CLOD sobre las anotaciones: k pliegues por grabación en train "
        "para tener predicciones fuera de muestra, y el checkpoint auditado en test, que "
        "nunca lo vio."
    )
    parser.add_argument("--run", default=AUDITED_RUN, help="corrida en runs/ que se audita")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_FOLD)
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--top", type=float, default=TOP)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="N ventanas por pliegue (prueba)")
    parser.add_argument(
        "--analyze-only", action="store_true", help="rehace el análisis sobre las cajas ya volcadas"
    )
    return parser.parse_args()


def audited(name: str, epochs: int) -> Audited:
    stored = json.loads((settings.runs_dir / name / "config.json").read_text())
    return Audited(
        stored["architecture"],
        stored["hparams"],
        TrainConfig(
            epochs=epochs,
            batch_size=stored["batch_size"],
            learning_rate=stored["learning_rate"],
            weight_decay=stored["weight_decay"],
            workers=stored["workers"],
            jitter=BoxJitter(**stored["jitter"]) if stored.get("jitter") else None,
            augment=None,
        ),
        cache.labels(),
    )


def to_table(
    found: Boxes,
    windows: np.ndarray,
    clips: list[tuple[str, float]],
    class_names: list[str],
    params: Parameters = P,
) -> pd.DataFrame:
    window_of_box = windows[found.image_ids.numpy()]
    clip_start_s = np.array([clips[window][1] for window in window_of_box])
    center_s, center_hz, duration, bandwidth = found.boxes.numpy().T
    return pd.DataFrame(
        {
            BEGIN: clip_start_s + (center_s - duration / 2) * params.clip_len_s,
            END: clip_start_s + (center_s + duration / 2) * params.clip_len_s,
            LOW: y_to_hz(center_hz - bandwidth / 2, params),
            HIGH: y_to_hz(center_hz + bandwidth / 2, params),
            CATEGORY: [class_names[label] for label in found.labels.numpy()],
            SCORE: found.scores.numpy(),
            RECORDING: [clips[window][0] for window in window_of_box],
            WINDOW: window_of_box,
        }
    )


def detect_on(
    checkpoint: Path,
    windows: np.ndarray,
    dataset: SpectrogramDataset,
    batch_size: int,
    device: str,
) -> Detected:
    model = load_checkpoint(checkpoint, device)
    predictions, truth, n_windows = collect_detections(
        model.detect,
        make_loader(Subset(dataset, windows.tolist()), batch_size),
        device,
        nms_iou=model.nms_iou,
        max_detections=None,
        desc=checkpoint.parent.name,
    )
    logger.info(
        "%s: %d ventanas, %d anotaciones, %d detecciones (%.1f por ventana)",
        checkpoint.parent.name,
        n_windows,
        len(truth.boxes),
        len(predictions.boxes),
        len(predictions.boxes) / max(n_windows, 1),
    )
    return Detected(truth, predictions, windows)


def cross_validate(
    args: argparse.Namespace, reference: Audited, device: str, out: Path
) -> tuple[list[Detected], int]:
    train_set = SpectrogramDataset(cache.split_path(TRAIN))
    val_set = SpectrogramDataset(cache.split_path(VAL))
    validation = Subset(val_set, range(len(val_set))[: args.limit])
    sources = cache.sources(TRAIN, len(train_set.boxes))
    fold_of_window = by_group(np.array(sources.recording_of_window), args.folds)

    detected = []
    for fold in range(args.folds):
        run_dir = out / f"fold{fold}"
        held_out = np.flatnonzero(fold_of_window == fold)[: args.limit]
        rest = np.flatnonzero(fold_of_window != fold)[: args.limit]

        if not (run_dir / BEST).is_file():
            logger.info(
                "pliegue %d/%d: entrena con %d ventanas, predice sobre %d",
                fold + 1,
                args.folds,
                len(rest),
                len(held_out),
            )
            set_seed(reference.config.seed)
            train_set.jitter = reference.config.jitter
            Trainer(
                build_model(reference.architecture, len(reference.labels), reference.hparams).to(
                    device
                ),
                reference.architecture,
                reference.hparams,
                reference.labels,
                make_loader(
                    Subset(train_set, rest.tolist()), reference.config.batch_size, shuffle=True
                ),
                make_loader(validation, reference.config.batch_size),
                run_dir,
                reference.config,
                device,
            ).fit()

        train_set.jitter = None
        detected.append(
            detect_on(run_dir / BEST, held_out, train_set, reference.config.batch_size, device)
        )
    return detected, len(fold_of_window)


def merge(
    detected: list[Detected], split: str, n_windows: int, class_names: list[str], out: Path
) -> None:
    clips = cache.clips(split, n_windows)
    truth = pd.concat(
        [to_table(part.truth, part.windows, clips, class_names) for part in detected],
        ignore_index=True,
    ).drop(columns=SCORE)
    truth[BOX] = truth.groupby(WINDOW).cumcount()
    predictions = pd.concat(
        [to_table(part.predictions, part.windows, clips, class_names) for part in detected],
        ignore_index=True,
    )

    torch.save({TRUTH: truth, PREDICTIONS: predictions}, out / f"{split}_cajas.pt")
    logger.info("%s: %d anotaciones y %d detecciones", split, len(truth), len(predictions))


def analyze(split: str, args: argparse.Namespace, out: Path) -> None:
    tables = torch.load(out / f"{split}_cajas.pt", weights_only=False)
    wide = tables[TRUTH][CATEGORY].isin(FULL_WINDOW_CLASSES)
    annotations = tables[TRUTH][~wide]
    predictions = tables[PREDICTIONS]
    predictions = predictions[
        predictions[SCORE].ge(args.score_floor) & ~predictions[CATEGORY].isin(FULL_WINDOW_CLASSES)
    ]

    found = scan(annotations, predictions, group=RECORDING, iou_threshold=args.iou)
    found.to_csv(out / f"{split}_calidad.csv", index=False)
    to_review = rank(found, args.top)
    to_review.to_csv(out / f"{split}_candidatos.csv", index=False)

    logger.info("%s | hallazgos por tipo:\n%s", split, found[ISSUE].value_counts().to_string())
    logger.info("%s | a revisar:\n%s", split, to_review[ISSUE].value_counts().to_string())


def main() -> None:
    args = parse_args()
    out = settings.runs_dir / f"clod_{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out / "clod.log")
    device = resolve_device(args.device)

    reference = audited(args.run, args.epochs)
    logger.info(
        "auditando %s | %s | %s | %d épocas por pliegue",
        args.run,
        reference.architecture,
        reference.hparams,
        args.epochs,
    )

    if not args.analyze_only:
        detected, n_windows = cross_validate(args, reference, device, out)
        merge(detected, TRAIN, n_windows, reference.labels.names, out)

        test_set = SpectrogramDataset(cache.split_path(TEST))
        windows = np.arange(len(test_set.boxes))[: args.limit]
        merge(
            [
                detect_on(
                    settings.runs_dir / args.run / BEST,
                    windows,
                    test_set,
                    reference.config.batch_size,
                    device,
                )
            ],
            TEST,
            len(test_set.boxes),
            reference.labels.names,
            out,
        )

    for split in (TRAIN, TEST):
        analyze(split, args, out)


if __name__ == "__main__":
    main()
