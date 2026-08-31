import argparse
import logging
from dataclasses import replace
from itertools import product
from typing import Any

from torch.utils.data import Subset

from core.config import P, settings
from core.runtime import resolve_device, set_seed, setup_logging
from data import cache
from data.datasets import SpectrogramDataset, make_loader
from evaluation.report import format_line
from models.faster_rcnn import ANCHOR_RATIOS, MAX_SIZE, MIN_SIZE
from models.registry import architecture, build_model
from training.trainer import TrainConfig, Trainer

logger = logging.getLogger("train")

# Cada detector entrenable: su nombre en el registro y los valores del baseline. Las
# ablaciones se barren pasando varios `--frontend` y `--time-stride` en un solo comando.
TRAINERS: dict[str, tuple[str, TrainConfig]] = {
    "detr": (
        "ast_deformable_detr",
        TrainConfig(epochs=30, batch_size=16, learning_rate=2e-4, workers=0),
    ),
    "frcnn": (
        "faster_rcnn",
        TrainConfig(epochs=30, batch_size=8, learning_rate=1e-4, workers=4),
    ),
}

MODEL_DIM, N_QUERIES, N_LEVELS = 128, 64, 3
TRAINABLE_BACKBONE_LAYERS = 3  # de 5; congelar las primeras ahorra memoria y sobreajuste


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un detector sobre las ventanas cacheadas."
    )
    parser.add_argument("--arch", choices=tuple(TRAINERS), default="detr")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="'cuda', 'cuda:1', 'cpu'")
    parser.add_argument("--limit", type=int, default=None, help="usa sólo N ventanas (pruebas)")
    parser.add_argument(
        "--name", default=None, help="nombre de la corrida; por defecto, la ablación"
    )

    detr = parser.add_argument_group("Deformable-DETR")
    detr.add_argument("--frontend", nargs="+", choices=("pcen", "logmel"), default=["pcen"])
    detr.add_argument(
        "--time-stride",
        type=int,
        nargs="+",
        default=[2],
        help="paso temporal del parcheo del AST; menos paso, más tokens y más VRAM",
    )
    detr.add_argument("--unfreeze", action="store_true", help="fine-tunea el AST entero")

    frcnn = parser.add_argument_group("Faster R-CNN")
    frcnn.add_argument("--scratch", action="store_true", help="sin los pesos de COCO")
    return parser.parse_args()


def variants(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """Las combinaciones a correr, como (nombre de la corrida, hiperparámetros del modelo).

    Los hiperparámetros son los que rearman el grafo: viajan en el checkpoint y con ellos
    `load_checkpoint` reconstruye el modelo sin más contexto.
    """
    if args.arch == "frcnn":
        db_low, db_high = cache.db_range()
        name = args.name or f"frcnn_{'scratch' if args.scratch else 'coco'}"
        return [
            (
                name,
                {
                    "db_low": db_low,
                    "db_high": db_high,
                    "min_size": MIN_SIZE,
                    "max_size": MAX_SIZE,
                    "anchor_ratios": ANCHOR_RATIOS,
                    "trainable_layers": TRAINABLE_BACKBONE_LAYERS,
                    "pretrained": not args.scratch,
                },
            )
        ]

    state = "ft" if args.unfreeze else "frozen"
    combinations = list(product(args.frontend, args.time_stride))
    if args.name and len(combinations) > 1:
        # Con un solo nombre las corridas se pisarían la carpeta y los checkpoints.
        raise SystemExit("--name sólo vale para una combinación; sacalo para barrer varias.")
    return [
        (
            args.name or f"detr_{frontend}_ts{stride}_{state}",
            {
                "dim": MODEL_DIM,
                "n_queries": N_QUERIES,
                "n_levels": N_LEVELS,
                "n_frames": P.n_frames,
                "frontend": frontend,
                "time_stride": stride,
                "freeze": not args.unfreeze,
            },
        )
        for frontend, stride in combinations
    ]


def train_one(
    name: str, hparams: dict[str, Any], args: argparse.Namespace, config: TrainConfig, device: str
) -> str:
    run_dir = settings.runs_dir / name
    setup_logging(log_file=run_dir / "train.log")
    logger.info("===== %s =====", name)

    labels = cache.labels()
    set_seed(config.seed)
    arch, _ = TRAINERS[args.arch]
    model = build_model(arch, len(labels), hparams).to(device)

    # Train va en el formato que consume el modelo; val siempre en el canónico, que es el
    # espacio en el que se miden las métricas y en el que los tres son comparables.
    train_set = architecture(arch).dataset(cache.split_path("train"), jitter=config.jitter)
    val_set = SpectrogramDataset(cache.split_path("val"))
    if args.limit:
        train_set = Subset(train_set, range(min(args.limit, len(train_set))))
        val_set = Subset(val_set, range(min(args.limit, len(val_set))))

    best = Trainer(
        model,
        arch,
        hparams,
        labels,
        make_loader(train_set, config.batch_size, config.workers, shuffle=True),
        make_loader(val_set, config.batch_size, config.workers),
        run_dir,
        config,
        device,
        dataset_meta=cache.meta(),
    ).fit()

    if best is None:  # no corrió ninguna época: `last.pt` ya estaba en la última
        return "ya completa"
    logger.info("mejor época -> %s", format_line(best))
    return format_line(best)


def main() -> None:
    args = parse_args()
    setup_logging()
    device = resolve_device(args.device)

    config = TRAINERS[args.arch][1]
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "workers": args.workers,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})

    runs = variants(args)
    logger.info("device: %s | %d corrida(s): %s", device, len(runs), [name for name, _ in runs])

    summary: list[str] = []
    for name, hparams in runs:
        try:
            summary.append(f"{name}: {train_one(name, hparams, args, config, device)}")
        except KeyboardInterrupt:
            logger.warning("interrumpido durante %s; `last.pt` quedó guardado", name)
            raise
        except Exception as error:
            # Que una combinación se caiga --típicamente por VRAM-- no puede llevarse
            # puesto el resto del barrido.
            logger.exception("%s falló", name)
            summary.append(f"{name}: falló ({type(error).__name__}: {error})")

    setup_logging()
    logger.info("resumen:\n  %s", "\n  ".join(summary))
    logger.info("evaluá con: python src/evaluate.py --run %s --split test", runs[0][0])


if __name__ == "__main__":
    main()
