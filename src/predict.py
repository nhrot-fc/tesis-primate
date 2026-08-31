import argparse
import logging
from pathlib import Path

from core.runtime import resolve_device, setup_logging
from inference.predictor import predict
from models.registry import load_checkpoint

logger = logging.getLogger("predict")


def wav_paths(audio_path: Path) -> list[Path]:
    if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        return [audio_path]
    if audio_path.is_dir():
        return sorted(p for p in audio_path.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    raise ValueError(f"'{audio_path}' no es un archivo .wav ni una carpeta con .wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detección sobre uno o más .wav, en tabla de selecciones de Raven."
    )
    parser.add_argument("audio_path", type=Path, help="archivo .wav o carpeta con .wav")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="por defecto, junto al .wav")
    parser.add_argument("--device", default=None)
    # Por defecto se usa el punto de operación con el que se eligió el checkpoint: si
    # acá se filtrara con otro umbral, el FP/TP que seleccionó ese `.pt` no diría nada
    # sobre la tabla que sale de este comando.
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    args = parser.parse_args()

    setup_logging()
    device = resolve_device(args.device)
    loaded = load_checkpoint(args.checkpoint, device)

    score_threshold = args.score_threshold or loaded.score_threshold
    nms_iou = args.nms_iou if args.nms_iou is not None else loaded.nms_iou
    logger.info("score >= %.2f | NMS IoU %.2f", score_threshold, nms_iou)

    for wav_path in wav_paths(args.audio_path):
        table = predict(loaded, wav_path, device, score_threshold, nms_iou)
        output_path = (args.output_dir or wav_path.parent) / f"{wav_path.stem}.selections.txt"
        table.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
        logger.info("%d detecciones -> %s", len(table), output_path)


if __name__ == "__main__":
    main()
