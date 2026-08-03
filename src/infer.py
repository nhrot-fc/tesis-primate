import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn

from architectures.deformable_detr import ASTDeformableDETR
from core.setup import setup_logging
from domain.species import LabelSet
from pipelines.inference_pipeline import predict

logger = logging.getLogger("inference")


class LoadedModel(NamedTuple):
    model: nn.Module
    labels: LabelSet
    score_threshold: float  # punto de operación con el que se eligió este checkpoint
    nms_iou: float
    config: dict


def load_model(checkpoint_path: Path, device: str) -> LoadedModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    labels = LabelSet(checkpoint["labels"])
    config = checkpoint.get("config", {})

    # La geometría del backbone (n_frames, time_stride) no viaja en los pesos: el
    # pos-embed se re-interpola al construir. Si no se reconstruye con los mismos
    # valores del entrenamiento, el modelo carga sin protestar y predice peor.
    model = ASTDeformableDETR(
        dim=checkpoint["dim"],
        n_queries=checkpoint["n_queries"],
        n_classes=len(labels),
        n_levels=checkpoint.get("n_levels", 3),
        n_frames=checkpoint.get("n_frames"),
        time_stride=checkpoint.get("time_stride", 5),
    ).to(device)

    # El checkpoint no guarda el AST congelado (`ASTBackbone` ya lo cargó de la copia
    # local), así que faltan sus claves a propósito. Cualquier otra ausencia es un
    # checkpoint incompatible y hay que gritarlo, no cargar pesos a medias.
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    unexpected_keys = list(unexpected) + [k for k in missing if not k.startswith("backbone.model.")]
    if unexpected_keys:
        raise RuntimeError(f"checkpoint incompatible con el modelo: {sorted(unexpected_keys)}")

    model.eval()
    return LoadedModel(
        model=model,
        labels=labels,
        score_threshold=float(config.get("operating_score_threshold", 0.5)),
        nms_iou=float(config.get("nms_iou", 0.3)),
        config=config,
    )


def wav_paths(audio_path: Path) -> list[Path]:
    if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        return [audio_path]
    elif audio_path.is_dir():
        return sorted(p for p in audio_path.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    else:
        raise ValueError(f"'{audio_path}' no es un archivo .wav ni una carpeta con .wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detección de llamadas sobre uno o más .wav.")
    parser.add_argument("audio_path", type=Path, help="archivo .wav o carpeta con .wav")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="por defecto, junto a cada .wav"
    )
    # Por defecto se usa el punto de operación con el que se eligió el checkpoint: si
    # acá se filtrara con otro umbral, el FP/TP que seleccionó ese `.pth` no diría nada
    # sobre la tabla que sale de este comando.
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    args = parser.parse_args()

    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_model(args.checkpoint, device)

    score_threshold = (
        args.score_threshold if args.score_threshold is not None else loaded.score_threshold
    )
    nms_iou = args.nms_iou if args.nms_iou is not None else loaded.nms_iou
    logger.info("score >= %.2f | NMS IoU %.2f", score_threshold, nms_iou)

    paths = wav_paths(args.audio_path)
    logger.info("%d archivo(s) a procesar", len(paths))

    for wav_path in paths:
        table = predict(
            loaded.model,
            wav_path,
            loaded.labels,
            device,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
        )
        output_dir = args.output_dir or wav_path.parent
        output_path = output_dir / f"{wav_path.stem}.selections.txt"
        table.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
        logger.info("%d detecciones -> %s", len(table), output_path)


if __name__ == "__main__":
    main()
