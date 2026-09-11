import argparse
import logging
from pathlib import Path

import torch

from core.config import settings
from core.runtime import resolve_device, setup_logging
from data import cache
from data.datasets import SpectrogramDataset, make_loader
from evaluation.evaluator import TEST, VAL, RawPredictions, collect_detections, path_for
from models.registry import load_checkpoint
from training.checkpoint import BEST

logger = logging.getLogger("dump_predictions")

BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vuelca las predicciones crudas de un checkpoint, sin umbral ni tope de "
            "detecciones, para que `compare_models.py` las mida a todas con la misma vara."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help=f"nombre de una corrida en runs/; usa su {BEST}")
    source.add_argument("--checkpoint", type=Path, help="ruta a un .pt/.pth cualquiera")
    parser.add_argument("--splits", nargs="+", choices=cache.SPLITS, default=[VAL, TEST])
    parser.add_argument("--name", help="con el que entra a la tabla; por defecto, el de la corrida")
    parser.add_argument("--output", type=Path, help="directorio; por defecto, el del checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, help="usa sólo las primeras N ventanas")
    return parser.parse_args()


def load_split(split: str, limit: int | None) -> tuple[SpectrogramDataset, cache.Sources]:
    dataset = SpectrogramDataset(cache.split_path(split))
    sources = cache.sources(split, len(dataset.boxes))
    if limit:
        dataset.images = dataset.images[:limit]
    return dataset, sources


def main() -> None:
    args = parse_args()
    setup_logging()
    device = resolve_device(args.device)

    path = args.checkpoint or settings.runs_dir / args.run / BEST
    loaded = load_checkpoint(path, device)
    nms_iou = loaded.nms_iou
    name = args.name or args.run or path.stem

    for split in args.splits:
        dataset, sources = load_split(split, args.limit)
        batch_size = args.batch_size
        while True:
            try:
                predictions, truth, n_images = collect_detections(
                    loaded.detect,
                    make_loader(dataset, batch_size),
                    device,
                    nms_iou=nms_iou,
                    max_detections=None,
                    desc=split,
                )
                break
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size //= 2
                logger.warning("%s %s: sin VRAM, reintento con batch %d", name, split, batch_size)
            torch.cuda.empty_cache()  # fuera del `except`: recién ahí se soltó el traceback
        present, recordings = torch.unique(
            torch.tensor(sources.recording_of_window[:n_images]), return_inverse=True
        )
        dump = RawPredictions(
            model=name,
            architecture=loaded.architecture,
            split=split,
            labels=loaded.labels.names,
            predictions=predictions,
            truth=truth,
            n_images=n_images,
            recordings=recordings,
            recording_names=[sources.recordings[index] for index in present.tolist()],
            nms_iou=nms_iou,
        )
        output = dump.save(path_for(args.output or path.parent, name, split))
        logger.info(
            "%s %s: %d detecciones sobre %d ventanas (%.1f por ventana) de %d grabaciones -> %s",
            name,
            split,
            len(predictions.boxes),
            n_images,
            dump.detections_per_window,
            len(present),
            output,
        )


if __name__ == "__main__":
    main()
