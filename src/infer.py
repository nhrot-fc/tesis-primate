import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from architectures.deformable_detr import ASTDeformableDETR
from core.setup import setup_logging
from domain.species import LabelSet
from pipelines.inference_pipeline import predict

logger = logging.getLogger("inference")


def load_model(checkpoint_path: Path, device: str) -> tuple[nn.Module, LabelSet]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    labels = LabelSet(checkpoint["labels"])
    model = ASTDeformableDETR(
        dim=checkpoint["dim"], n_queries=checkpoint["n_queries"], n_classes=len(labels)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels


def wav_paths(audio_path: Path) -> list[Path]:
    return sorted(audio_path.rglob("*.wav")) if audio_path.is_dir() else [audio_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detección de llamadas sobre uno o más .wav.")
    parser.add_argument("audio_path", type=Path, help="archivo .wav o carpeta con .wav")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="por defecto, junto a cada .wav"
    )
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    args = parser.parse_args()

    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels = load_model(args.checkpoint, device)

    paths = wav_paths(args.audio_path)
    logger.info("%d archivo(s) a procesar", len(paths))

    for wav_path in paths:
        table = predict(
            model,
            wav_path,
            labels,
            device,
            score_threshold=args.score_threshold,
            nms_iou=args.nms_iou,
        )
        output_dir = args.output_dir or wav_path.parent
        output_path = output_dir / f"{wav_path.stem}.selections.txt"
        table.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
        logger.info("%d detecciones -> %s", len(table), output_path)


if __name__ == "__main__":
    main()
