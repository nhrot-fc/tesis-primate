import argparse
import logging
from pathlib import Path

import torch

from core.config import RUNS_DIR
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
        description="Vuelca las predicciones crudas de un checkpoint sobre val y test, sin umbral "
        "ni tope, para que `compare_models.py` las mida con la misma vara."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help=f"corrida en runs/; usa su {BEST}")
    source.add_argument("--checkpoint", type=Path, help="ruta a un checkpoint cualquiera")
    parser.add_argument("--splits", nargs="+", choices=cache.SPLITS, default=[VAL, TEST])
    parser.add_argument("--name", help="con el que entra a la tabla; por defecto, el de la corrida")
    parser.add_argument("--output", type=Path, help="directorio; por defecto, el del checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, help="usa sólo las primeras N ventanas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    device = resolve_device(args.device)
    path = args.checkpoint or RUNS_DIR / args.run / BEST
    loaded = load_checkpoint(path, device)
    name = args.name or args.run or path.stem

    for split in args.splits:
        dataset = SpectrogramDataset(cache.split_path(split))
        sources = cache.sources(split, len(dataset.boxes))
        if args.limit:
            dataset.images = dataset.images[: args.limit]

        batch_size = args.batch_size
        while True:
            try:
                predictions, truth, n_images = collect_detections(
                    loaded.model.detect,
                    make_loader(dataset, batch_size),
                    device,
                    loaded.model.nms_iou,
                    desc=split,
                )
                break
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size //= 2
                logger.warning("%s %s: sin VRAM, reintento con batch %d", name, split, batch_size)
            torch.cuda.empty_cache()

        present, recordings = torch.unique(
            torch.tensor(sources.recording_of_window[:n_images]), return_inverse=True
        )
        dump = RawPredictions(
            model=name,
            split=split,
            labels=loaded.labels.names,
            predictions=predictions,
            truth=truth,
            n_images=n_images,
            recordings=recordings,
            recording_names=[sources.recordings[i] for i in present.tolist()],
        )
        output = dump.save(path_for(args.output or path.parent, name, split))
        logger.info(
            "%s %s: %d detecciones sobre %d ventanas -> %s",
            name,
            split,
            len(predictions.boxes),
            n_images,
            output,
        )


if __name__ == "__main__":
    main()
