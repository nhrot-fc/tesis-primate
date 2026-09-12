import argparse
import json
import logging
from dataclasses import replace
from typing import Any, NamedTuple

from torch.utils.data import Subset

from core.config import RUNS_DIR, P
from core.runtime import resolve_device, set_seed, setup_logging
from data import cache
from data.datasets import SpectrogramDataset, make_loader
from models.backbone import TIME_STRIDE
from models.coco_deformable_detr import DETR_CHECKPOINT
from models.deformable_detr import DIM, FRONTEND, N_QUERIES
from models.faster_rcnn import ANCHOR_RATIOS, MAX_SIZE, MIN_SIZE, TRAINABLE_LAYERS
from models.registry import ARCHITECTURES, build_model
from models.yolo import DEFAULT_MODEL, IMAGE_SIZE
from training import yolo
from training.trainer import TrainConfig, Trainer

logger = logging.getLogger("train")


# Una receta de entrenamiento: qué `models.registry.ARCHITECTURES` y con qué hiperparámetros.
# Los valores repiten los defaults de cada modelo para que queden escritos en el checkpoint.
class Preset(NamedTuple):
    arch: str
    config: TrainConfig
    # Hiperparámetros del modelo usado en `build_model`
    hparams: dict[str, Any]


PRESETS: dict[str, Preset] = {
    # AST-Deformable-DETR
    "detr": Preset(
        "ast_deformable_detr",
        TrainConfig(epochs=30, batch_size=8, learning_rate=2e-4),
        {
            "n_frames": P.n_frames,
            "time_stride": TIME_STRIDE,
            "dim": DIM,
            "n_queries": N_QUERIES,
            "frontend": FRONTEND,  # none | logmel | pcen
        },
    ),
    # Deformable DETR de HF con encoder, decoder y propuestas de COCO, sobre el mismo AST
    "detr_coco": Preset(
        "coco_deformable_detr",
        TrainConfig(epochs=30, batch_size=8, learning_rate=2e-4),
        {
            "n_frames": P.n_frames,
            "time_stride": TIME_STRIDE,
            "n_queries": N_QUERIES,
            "frontend": FRONTEND,
            "checkpoint": DETR_CHECKPOINT,
        },
    ),
    # Faster R-CNN con ResNet50-FPN
    "frcnn": Preset(
        "faster_rcnn",
        TrainConfig(epochs=12, batch_size=4, learning_rate=1e-4),
        {
            "min_size": MIN_SIZE,
            "max_size": MAX_SIZE,
            "anchor_ratios": ANCHOR_RATIOS,
            "trainable_layers": TRAINABLE_LAYERS,
            "pretrained": True,
        },
    ),
    # Ultralytics YOLO
    "yolo": Preset(
        "yolo", TrainConfig(epochs=30, batch_size=32), {"model": DEFAULT_MODEL, "imgsz": IMAGE_SIZE}
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un detector sobre las ventanas cacheadas (YOLO: sobre el export).",
        epilog="p. ej. --arch yolo --hp model=yolo26m | --arch detr --cfg epochs=20 batch_size=4",
    )
    parser.add_argument("--arch", choices=tuple(PRESETS), default="detr")
    parser.add_argument("--name", help="nombre de la corrida; por defecto, la ablación")
    parser.add_argument("--device", help="'cuda', 'cuda:1', 'cpu'")
    parser.add_argument("--limit", type=int, help="usa sólo N ventanas (pruebas)")
    parser.add_argument(
        "--hp", nargs="+", default=[], metavar="CLAVE=VALOR", help="pisa `Preset.hparams`"
    )
    parser.add_argument(
        "--cfg", nargs="+", default=[], metavar="CLAVE=VALOR", help="pisa `TrainConfig`"
    )
    return parser.parse_args()


def overrides(pairs: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"esperaba clave=valor, no {pair!r}")
        try:
            parsed[key] = json.loads(value)
        except ValueError:
            parsed[key] = value
    return parsed


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.arch]
    ablation = overrides(args.hp)
    hparams = preset.hparams | ablation
    if ARCHITECTURES[preset.arch].needs_db_range:
        hparams["db_low"], hparams["db_high"] = cache.db_range()

    name = args.name or "_".join([args.arch, *(f"{k}-{v}" for k, v in sorted(ablation.items()))])
    run_dir = RUNS_DIR / name
    setup_logging(log_file=run_dir / "train.log")
    logger.info("===== %s =====", name)

    config = replace(preset.config, **overrides(args.cfg))
    device = resolve_device(args.device)
    set_seed(config.seed)

    if preset.arch == "yolo":
        yolo.fit(run_dir, hparams, config, device, limit=args.limit)
        logger.info("volcá predicciones con: python src/dump_predictions.py --run %s", name)
        return

    labels = cache.labels()
    model = build_model(preset.arch, len(labels), hparams).to(device)

    train_set = SpectrogramDataset(
        cache.split_path(cache.TRAIN), jitter=config.jitter, augment=config.augment
    )
    val_set = SpectrogramDataset(cache.split_path(cache.VAL))
    if args.limit:
        train_set = Subset(train_set, range(min(args.limit, len(train_set))))
        val_set = Subset(val_set, range(min(args.limit, len(val_set))))

    Trainer(
        model,
        preset.arch,
        hparams,
        labels,
        make_loader(train_set, config.batch_size, config.workers, shuffle=True),
        make_loader(val_set, config.batch_size, config.workers),
        run_dir,
        config,
        device,
    ).fit()

    logger.info("Guarda predicciones con: python src/dump_predictions.py --run %s", name)


if __name__ == "__main__":
    main()
