import argparse
import json
import logging
from dataclasses import replace
from typing import Any, NamedTuple

from torch.utils.data import Subset

from core.config import P, settings
from core.runtime import resolve_device, set_seed, setup_logging
from data import cache
from data.datasets import SpectrogramDataset, make_loader
from models.faster_rcnn import ANCHOR_RATIOS, MAX_SIZE, MIN_SIZE
from models.registry import architecture, build_model
from training.trainer import TrainConfig, Trainer

logger = logging.getLogger("train")


class Detector(NamedTuple):
    arch: str  # clave en `models.registry.ARCHITECTURES`
    config: TrainConfig
    # Los hiperparámetros son los que rearman el grafo: viajan en el checkpoint y con ellos
    # `load_checkpoint` reconstruye el modelo sin más contexto.
    hparams: dict[str, Any]


# El AST parchea con solape (time_stride 2) y el EAT, como en su preentrenamiento, sin
# solape (16 x 16). La geometría del DINO es la de la literatura (Zhu & Sato, DCASE 2025):
# 100 queries de 256 dimensiones y pirámide de cuatro niveles {1/32, 1/16, 1/8, 1/4}.
DETECTORS: dict[str, Detector] = {
    "detr": Detector(
        "ast_deformable_detr",
        TrainConfig(epochs=30, batch_size=16, learning_rate=2e-4),
        {
            "n_frames": P.n_frames,
            "frontend": "pcen",
            "freeze": True,
            "time_stride": 2,
            "dim": 128,
            "n_queries": 100,
            "n_levels": 4,
        },
    ),
    # DINO trae encoder deformable además del decodificador: entra menos lote y va con la
    # tasa de aprendizaje de su paper (1e-4) en vez de la del DETR de acá.
    "dino": Detector(
        "eat_dino",
        TrainConfig(epochs=30, batch_size=8, learning_rate=1e-4),
        {
            "n_frames": P.n_frames,
            "frontend": "pcen",
            "freeze": True,
            "time_stride": 16,
            "dim": 256,
            "n_queries": 100,
            "n_levels": 4,
            "n_encoder_layers": 6,  # bajarlo es lo primero si falta VRAM
            "dn_queries": 100,
        },
    ),
    "frcnn": Detector(
        "faster_rcnn",
        TrainConfig(epochs=30, batch_size=8, learning_rate=1e-4),
        {
            "min_size": MIN_SIZE,
            "max_size": MAX_SIZE,
            "anchor_ratios": ANCHOR_RATIOS,
            "trainable_layers": 3,  # de 5; congelar las primeras ahorra memoria y sobreajuste
            "pretrained": True,
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un detector sobre las ventanas cacheadas.",
        epilog="ablaciones: --hp frontend=logmel time_stride=4 freeze=false pretrained=false",
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
    if args.arch == "frcnn":
        # Único que pinta el mel como imagen: necesita el rango en dB con el que se cacheó.
        hparams["db_low"], hparams["db_high"] = cache.db_range()

    name = args.name or "_".join([args.arch, *(f"{k}-{v}" for k, v in sorted(ablation.items()))])
    run_dir = settings.runs_dir / name
    setup_logging(log_file=run_dir / "train.log")
    logger.info("===== %s =====", name)

    config = replace(detector.config, **overrides(args.cfg))
    device = resolve_device(args.device)
    set_seed(config.seed)

    labels = cache.labels()
    model = build_model(detector.arch, len(labels), hparams).to(device)

    train_set = architecture(detector.arch).dataset(
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

    logger.info("volcá predicciones con: python src/dump_predictions.py --run %s", name)


if __name__ == "__main__":
    main()
