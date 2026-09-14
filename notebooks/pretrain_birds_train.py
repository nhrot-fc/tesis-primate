"""Entrena el AST-DETR (`detr`, logmel) con el caché de aves de `notebooks/pretrain_birds.ipynb`
y después lo afina con primates, sin tocar `src/`: sólo redirige el módulo `cache` y la carpeta de
corridas antes de llamar a `train.main()`.

    # 1. preentrenar: caché data/processed_extra, corrida en runs_extra/<name>
    uv run python notebooks/pretrain_birds_train.py --name detr_birds --device cuda:2

    # 2. afinar: caché y runs/ normales; arranca de los pesos del preentrenado salvo las cabezas
    #    de clase (otro número de clases) y entra a la comparación como cualquier corrida
    uv run python notebooks/pretrain_birds_train.py --finetune runs_extra/detr_birds/best.pt \\
        --name detr_t10_logmel_birds --device cuda:2

Lo que siga a las opciones propias se pasa tal cual a train.py (`--cfg epochs=20`, `--limit`…).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import train  # noqa: E402
from data import cache  # noqa: E402

PROCESSED_EXTRA = PROJECT_DIR / "data" / "processed_extra"
RUNS_EXTRA = PROJECT_DIR / "runs_extra"
# El preset `detr` con el frontend logmel: la corrida detr_t10_logmel_v2 de la comparación
HPARAMS = {"frontend": "logmel"}

logger = logging.getLogger("pretrain_birds")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Preentrena el AST-DETR con el caché de aves, o lo afina con primates."
    )
    parser.add_argument("--finetune", type=Path, help="best.pt de runs_extra/: afina con primates")
    parser.add_argument("--name", required=True, help="nombre de la corrida")
    parser.add_argument("--device")
    return parser.parse_known_args()


# Carga en el modelo recién construido todo tensor del preentrenado con el mismo nombre y forma:
# queda fuera lo que depende del número de clases (las cabezas de clase).
def initializer(checkpoint: Path):
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    build = train.build_model

    def build_and_load(architecture: str, n_classes: int, hparams: dict):
        model = build(architecture, n_classes, hparams)
        own = model.state_dict()
        compatible = {
            key: value
            for key, value in stored["state_dict"].items()
            if key in own and own[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        skipped = sorted(set(own) - set(compatible))
        logger.info(
            "%d tensores desde %s; %d nuevos: %s",
            len(compatible),
            checkpoint,
            len(skipped),
            ", ".join(skipped),
        )
        return model

    return build_and_load, stored["hparams"]


def main() -> None:
    args, rest = parse_args()
    hparams = HPARAMS
    if args.finetune is None:
        cache.PROCESSED_DIR = PROCESSED_EXTRA
        train.RUNS_DIR = RUNS_EXTRA
    else:
        # Los mismos hiperparámetros del grafo que el preentrenado, para que los pesos calcen
        train.build_model, hparams = initializer(args.finetune)
    hp = [f"{key}={json.dumps(value)}" for key, value in hparams.items()]
    device = ["--device", args.device] if args.device else []
    sys.argv = [sys.argv[0], "--arch", "detr", "--hp", *hp, "--name", args.name, *device, *rest]
    train.main()


if __name__ == "__main__":
    main()
