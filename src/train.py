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
from models.faster_rcnn import ANCHOR_RATIOS, MAX_SIZE, MIN_SIZE
from models.registry import build_model
from training import yolo
from training.trainer import TrainConfig, Trainer

logger = logging.getLogger("train")


class Detector(NamedTuple):
    # Key usada para indexar en DETECTORS
    arch: str
    config: TrainConfig
    # Hiperparámetros del modelo usado en `build_model`
    hparams: dict[str, Any]


DETECTORS: dict[str, Detector] = {
    # AST-Deformable-DETR
    "detr": Detector(
        "ast_deformable_detr",
        TrainConfig(epochs=30, batch_size=8, learning_rate=2e-4),
        {"n_frames": P.n_frames, "time_stride": 10, "dim": 128, "n_queries": 100},
    ),
    # Faster R-CNN con ResNet50-FPN
    "frcnn": Detector(
        "faster_rcnn",
        TrainConfig(epochs=12, batch_size=4, learning_rate=1e-4),
        {
            "min_size": MIN_SIZE,
            "max_size": MAX_SIZE,
            "anchor_ratios": ANCHOR_RATIOS,
            "trainable_layers": 3,  # de 5
            "pretrained": True,
        },
    ),
    # Ultralytics YOLO
    "yolo": Detector(
        "yolo", TrainConfig(epochs=30, batch_size=32), {"model": "yolo26s", "imgsz": 512}
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un detector sobre las ventanas cacheadas (YOLO: sobre el export).",
        epilog="p. ej. --arch yolo --hp model=yolo26m | --arch detr --cfg epochs=20 batch_size=4",
    )
    parser.add_argument("--arch", choices=tuple(DETECTORS), default="detr")
    parser.add_argument("--name", help="nombre de la corrida; por defecto, la ablación")
    parser.add_argument("--device", help="'cuda', 'cuda:1', 'cpu'")
    parser.add_argument("--limit", type=int, help="usa sólo N ventanas (pruebas)")
    parser.add_argument(
        "--hp", nargs="+", default=[], metavar="CLAVE=VALOR", help="pisa `Detector.hparams`"
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
    detector = DETECTORS[args.arch]
    ablation = overrides(args.hp)
    hparams = detector.hparams | ablation
    # El FRCNN pinta el mel como imagen: necesita el rango en dB del caché
    if args.arch == "frcnn":
        hparams["db_low"], hparams["db_high"] = cache.db_range()

    name = args.name or "_".join([args.arch, *(f"{k}-{v}" for k, v in sorted(ablation.items()))])
    run_dir = RUNS_DIR / name
    setup_logging(log_file=run_dir / "train.log")
    logger.info("===== %s =====", name)

    config = replace(detector.config, **overrides(args.cfg))
    device = resolve_device(args.device)
    set_seed(config.seed)

    if detector.arch == "yolo":
        yolo.fit(run_dir, hparams, config, device, limit=args.limit)
        logger.info("volcá predicciones con: python src/dump_predictions.py --run %s", name)
        return

    labels = cache.labels()
    model = build_model(detector.arch, len(labels), hparams).to(device)

    train_set = SpectrogramDataset(
        cache.split_path("train"), jitter=config.jitter, augment=config.augment
    )
    val_set = SpectrogramDataset(cache.split_path("val"))
    if args.limit:
        train_set = Subset(train_set, range(min(args.limit, len(train_set))))
        val_set = Subset(val_set, range(min(args.limit, len(val_set))))

    Trainer(
        model,
        detector.arch,
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
