import argparse
import logging
from pathlib import Path

import torch

from architectures.registry import LoadedModel, load_checkpoint
from core.setup import setup_logging
from pipelines.inference_pipeline import predict

logger = logging.getLogger("inference")


def load_model(checkpoint_path: Path, device: str) -> LoadedModel:
    """Reconstruye la arquitectura que dice el checkpoint y le carga sus pesos.

    La geometría del modelo no viaja en los pesos (el pos-embed del AST, por ejemplo,
    se re-interpola al construir): si no se rearma con los mismos hiperparámetros del
    entrenamiento, carga sin protestar y predice peor. Por eso viajan en `hparams`.
    """
    return load_checkpoint(checkpoint_path, device)


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
            loaded,
            wav_path,
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
