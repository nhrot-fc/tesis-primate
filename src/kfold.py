"""Predicciones fuera de muestra sobre train: K modelos, cada uno predice el pliegue que no vio.

Los pliegues van por grabación. Cada modelo elige su `best.pt` sobre val, como `train.py`; el
pliegue retenido sólo se predice. Los volcados por pliegue se funden en
`runs/<nombre>/<nombre>_train_predictions.pt`, con el formato de `dump_predictions.py`, que es
lo que leen `find_issues.py` (CLOD) y `compare_models.py`. YOLO entrena por Ultralytics sobre
el export de `export_yolo.py`, restringido a las ventanas del pliegue.

    python src/kfold.py --arch detr --hp frontend=logmel --cfg epochs=20
    python src/kfold.py --arch detr --hp frontend=logmel --cfg epochs=20 --fold 1 3   # otra GPU
    python src/kfold.py --arch yolo
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from core.config import RUNS_DIR, SEED
from core.runtime import resolve_device, set_seed, setup_logging
from data import cache
from data.datasets import SpectrogramDataset, make_loader
from evaluation.evaluator import RawPredictions, collect_detections, path_for
from evaluation.metrics import Boxes, concat, sort_by_score
from models.registry import ARCHITECTURES, build_model, load_checkpoint
from train import PRESETS, overrides
from training import checkpoint, yolo
from training.trainer import Trainer

logger = logging.getLogger("kfold")

FOLDS = "folds.json"
# Predicciones de un modelo sobre su pliegue retenido, con los ids de ventana de train
HELD_OUT = "held_out_predictions.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arch", choices=tuple(PRESETS), default="detr")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold", type=int, nargs="+", help="sólo estos pliegues; por defecto todos"
    )
    parser.add_argument("--name", help="carpeta en runs/; por defecto <arch>_kfold<K>")
    parser.add_argument("--device", help="'cuda', 'cuda:1', 'cpu'")
    parser.add_argument("--limit", type=int, help="usa sólo N ventanas de train (pruebas)")
    parser.add_argument(
        "--hp", nargs="+", default=[], metavar="CLAVE=VALOR", help="pisa `Preset.hparams`"
    )
    parser.add_argument(
        "--cfg", nargs="+", default=[], metavar="CLAVE=VALOR", help="pisa `TrainConfig`"
    )
    return parser.parse_args()


def assign_folds(recording_of_window: list[int], n_folds: int, seed: int = SEED) -> list[int]:
    # -> pliegue de cada ventana. Cada grabación entera al pliegue con menos ventanas, de la
    # más grande a la más chica, para que queden parejos.
    windows: dict[int, list[int]] = defaultdict(list)
    for window, recording in enumerate(recording_of_window):
        windows[recording].append(window)
    recordings = list(windows)
    random.Random(seed).shuffle(recordings)
    recordings.sort(key=lambda r: -len(windows[r]))

    sizes = [0] * n_folds
    fold_of_window = [0] * len(recording_of_window)
    for recording in recordings:
        fold = sizes.index(min(sizes))
        for window in windows[recording]:
            fold_of_window[window] = fold
        sizes[fold] += len(windows[recording])
    return fold_of_window


def read_or_assign_folds(run_dir: Path, recording_of_window: list[int], n_folds: int) -> list[int]:
    path = run_dir / FOLDS
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored["n_folds"] != n_folds or len(stored["fold_of_window"]) != len(
            recording_of_window
        ):
            raise SystemExit(f"{path} es de otro reparto; borralo o usá otro --name")
        return stored["fold_of_window"]
    fold_of_window = assign_folds(recording_of_window, n_folds)
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_folds": n_folds, "fold_of_window": fold_of_window}))
    return fold_of_window


def predict_held_out(
    run_dir: Path, dataset: SpectrogramDataset, held_out: list[int], batch_size: int, device: str
) -> None:
    loaded = load_checkpoint(run_dir / checkpoint.BEST, device)
    # Sin aumento ni jitter: el pliegue retenido se predice tal como está en el caché.
    augmenter, jitter = dataset.augmenter, dataset.jitter
    dataset.augmenter, dataset.jitter = None, None
    try:
        predictions, truth, _ = collect_detections(
            loaded.model.detect,
            make_loader(Subset(dataset, held_out), batch_size),
            device,
            loaded.model.nms_iou,
            desc=run_dir.name,
        )
    finally:
        dataset.augmenter, dataset.jitter = augmenter, jitter
    index = torch.tensor(held_out)
    torch.save(
        {
            "predictions": predictions._replace(image_ids=index[predictions.image_ids.long()]),
            "truth": truth._replace(image_ids=index[truth.image_ids.long()]),
        },
        run_dir / HELD_OUT,
    )


def merge(name: str, n_folds: int, n_windows: int, labels: list[str]) -> Path:
    parts = [
        torch.load(RUNS_DIR / name / f"fold{k}" / HELD_OUT, weights_only=False)
        for k in range(n_folds)
    ]
    sources = cache.sources(cache.TRAIN, n_windows)
    present, recordings = torch.unique(
        torch.tensor(sources.recording_of_window), return_inverse=True
    )
    dump = RawPredictions(
        model=name,
        split=cache.TRAIN,
        labels=labels,
        predictions=sort_by_score(concat([Boxes(*part["predictions"]) for part in parts])),
        truth=concat([Boxes(*part["truth"]) for part in parts]),
        n_images=n_windows,
        recordings=recordings,
        recording_names=[sources.recordings[i] for i in present.tolist()],
    )
    return dump.save(path_for(RUNS_DIR / name, name, cache.TRAIN))


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.arch]
    is_yolo = preset.arch == "yolo"
    hparams = preset.hparams | overrides(args.hp)
    if ARCHITECTURES[preset.arch].needs_db_range:
        hparams["db_low"], hparams["db_high"] = cache.db_range()
    config = replace(preset.config, **overrides(args.cfg))
    name = args.name or f"{args.arch}_kfold{args.folds}"
    run_dir = RUNS_DIR / name
    setup_logging(log_file=run_dir / "kfold.log")
    device = resolve_device(args.device)
    labels = cache.labels()

    # YOLO no lee el caché al entrenar: sólo hace falta para predecir el pliegue retenido
    train_set = (
        SpectrogramDataset(cache.split_path(cache.TRAIN))
        if is_yolo
        else SpectrogramDataset(
            cache.split_path(cache.TRAIN), jitter=config.jitter, augment=config.augment
        )
    )
    n_windows = len(train_set)
    recording_of_window = cache.sources(cache.TRAIN, n_windows).recording_of_window
    fold_of_window = read_or_assign_folds(run_dir, recording_of_window, args.folds)
    val_set: Dataset = SpectrogramDataset(cache.split_path(cache.VAL))
    if args.limit:
        fold_of_window = fold_of_window[: args.limit]
        val_set = Subset(val_set, range(args.limit))

    for fold in args.fold or range(args.folds):
        fold_dir = run_dir / f"fold{fold}"
        if (fold_dir / HELD_OUT).is_file():
            logger.info("pliegue %d ya predicho: %s", fold, fold_dir / HELD_OUT)
            continue
        held_out = [w for w, f in enumerate(fold_of_window) if f == fold]
        seen = [w for w, f in enumerate(fold_of_window) if f != fold]
        logger.info(
            "===== %s pliegue %d: %d ventanas dentro, %d fuera =====",
            name,
            fold,
            len(seen),
            len(held_out),
        )
        set_seed(config.seed)
        if is_yolo:
            yolo.fit(
                fold_dir,
                hparams,
                config,
                device,
                train_images=[yolo.image_path(cache.TRAIN, w) for w in seen],
            )
        else:
            model = build_model(preset.arch, len(labels), hparams).to(device)
            Trainer(
                model,
                preset.arch,
                hparams,
                labels,
                make_loader(
                    Subset(train_set, seen), config.batch_size, config.workers, shuffle=True
                ),
                make_loader(val_set, config.batch_size, config.workers),
                fold_dir,
                config,
                device,
            ).fit()
            del model
            torch.cuda.empty_cache()
        predict_held_out(fold_dir, train_set, held_out, config.batch_size, device)

    if all((run_dir / f"fold{k}" / HELD_OUT).is_file() for k in range(args.folds)):
        logger.info(
            "volcado fuera de muestra: %s", merge(name, args.folds, n_windows, labels.names)
        )
    else:
        logger.info("faltan pliegues; cuando estén todos, cualquier invocación los funde")


if __name__ == "__main__":
    main()
