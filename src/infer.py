import argparse
import json
import logging
from pathlib import Path

import torch

from architectures.deformable_detr import ASTDeformableDETR
from core.setup import setup_logging
from domain.species import LabelSet
from pipelines.inference_pipeline import predict

logger = logging.getLogger("inference")


def load_model(
    checkpoint_path: Path,
    device: str,
    labels: list[str] | None = None,
    dim: int = 128,
    n_queries: int = 64,
) -> tuple[torch.nn.Module, LabelSet]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict, label_set = checkpoint["state_dict"], LabelSet(checkpoint["labels"])
        dim, n_queries = checkpoint["dim"], checkpoint["n_queries"]
    elif labels is not None:
        state_dict, label_set = checkpoint, LabelSet(labels)
    else:
        raise ValueError(
            f"{checkpoint_path} es un checkpoint sin metadatos (state_dict crudo); "
            "pasa --labels (y --dim/--n-queries si no son los valores por defecto)."
        )

    model = ASTDeformableDETR(dim=dim, n_queries=n_queries, n_classes=len(label_set)).to(device)
    model.load_state_dict(state_dict)
    return model, label_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Detección de llamadas sobre un audio.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model_state.pth"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument(
        "--labels-file",
        type=Path,
        help="JSON {class_id: label} guardado por main.py junto al checkpoint "
        "(solo para checkpoints sin metadatos); tiene prioridad sobre --labels.",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-queries", type=int, default=64)
    args = parser.parse_args()

    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapping = json.loads(args.labels_file.read_text())
    labels = [mapping[str(class_id)] for class_id in range(len(mapping))]
    model, labels = load_model(args.checkpoint, device, labels, args.dim, args.n_queries)

    table = predict(
        model,
        args.audio_path,
        labels,
        device,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
    )

    output_path = args.output or args.audio_path.with_suffix(".selections.txt")
    table.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
    logger.info("%d detecciones -> %s", len(table), output_path)


if __name__ == "__main__":
    main()
