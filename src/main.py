import copy
import json
import logging
from collections import Counter
from pathlib import Path

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from architectures.criterion import HungarianMatcher, SetCriterion
from architectures.deformable_detr import ASTDeformableDETR
from core.config import P, settings
from core.setup import setup_logging, setup_project_path
from domain.annotations import load_annotations
from domain.dataset import (
    LABEL_INDEX,
    CallBoxDataset,
    build_manifest,
    collate_fn,
    split_manifest,
)
from domain.species import LabelSet
from pipelines.training_pipeline import evaluate, train_one_epoch

logger = logging.getLogger("training")

PROJECT_DIR = Path.cwd()

# --- Preprocesado -------------------------------------------------------------
CLIP_PARAMS = {
    "clip_len_s": 3.0,
    "clip_hop_s": 1.5,
    "min_overlap": 0.5,
    "target_sr": 44_100,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 400,
    "n_mels": 128,
    "f_min": 0.0,
    "f_max": 22050.0,
}

SEED = 42
MIN_PAIR_COUNT = 500
LABEL_BY = "species/call_type"
LABEL_COLUMN = {
    "call": lambda df: "call",
    "species": lambda df: df["species"],
    "species/call_type": lambda df: df["species"] + "/" + df["call_type"],
}

# --- Entrenamiento ------------------------------------------------------------
MODEL_DIM, N_QUERIES = 128, 64
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS = 40, 128, 2e-4, 1e-4, 0
EVAL_EVERY = 10
MATCHER_IOU_TYPE = "iou"
METRIC_IOU_TYPE = "iou"
METRIC_IOU_THRESHOLD = 0.25
OPERATING_SCORE_THRESHOLD = 0.5

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"


def save_labels_json(labels: LabelSet, path: Path) -> None:
    """class_id -> label, para inspección rápida y para que `infer.py` los lea
    sin tener que cargar el checkpoint completo."""
    path.write_text(json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False))


def make_loader(dataset: CallBoxDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def preprocess() -> tuple[LabelSet, DataLoader, DataLoader]:
    for name, value in CLIP_PARAMS.items():
        setattr(P, name, value)
    logger.info(
        "clip %ss @ %s Hz -> espectrograma %s x %s",
        P.clip_len_s,
        P.target_sr,
        P.n_mels,
        P.n_frames,
    )

    annotations = load_annotations(settings.data_dir / "cleaned")
    annotations = annotations[
        ~(
            ((annotations["species"] == "lw") & (annotations["call_type"] == "cc"))
            | ((annotations["species"] == "sm") & (annotations["call_type"] == "fc"))
        )
    ]
    annotations.loc[annotations["low_freq_hz"] <= 50.0, "low_freq_hz"] = 50.0
    logger.info("%d anotaciones | %d especies", len(annotations), annotations.species.nunique())

    pairs = annotations[["species", "call_type"]].apply(tuple, axis=1)
    pair_counts = pairs.value_counts()
    valid_pairs = pair_counts[pair_counts >= MIN_PAIR_COUNT].index
    logger.info(
        "%d/%d pares species/call_type con >= %d anotaciones: %s",
        len(valid_pairs),
        len(pair_counts),
        MIN_PAIR_COUNT,
        ", ".join(f"{species}/{call_type}" for species, call_type in valid_pairs),
    )

    experiment_df = annotations[pairs.isin(valid_pairs)].copy()
    experiment_df["label"] = LABEL_COLUMN[LABEL_BY](experiment_df)

    labels = LabelSet(experiment_df["label"])
    logger.info(
        "%d anotaciones del experimento | %d clases: %s",
        len(experiment_df),
        len(labels),
        ", ".join(labels.names),
    )

    manifest = build_manifest(experiment_df, labels)
    train_m, val_m, _ = split_manifest(manifest, seed=SEED)

    class_counts = Counter(
        labels.name(class_id) for window in train_m for class_id in window.boxes[:, LABEL_INDEX]
    )
    logger.info("ventanas -> train %d | val %d", len(train_m), len(val_m))
    logger.info("cajas en train -> %s", dict(class_counts.most_common()))

    train_loader = make_loader(CallBoxDataset(train_m), shuffle=True)
    val_loader = make_loader(CallBoxDataset(val_m), shuffle=False)
    return labels, train_loader, val_loader


def format_confusion(confusion: torch.Tensor, names: list[str]) -> str:
    header = "true\\pred".rjust(10) + "".join(f"{name:>10}" for name in names)
    rows = [header]
    for i, name in enumerate(names):
        row = f"{name:>10}" + "".join(f"{int(confusion[i, j]):>10}" for j in range(len(names)))
        rows.append(row)
    return "\n".join(rows)


def format_ap(ap_per_class: dict[int, float | None], names: list[str]) -> str:
    return ", ".join(
        f"{names[class_id]}={ap:.3f}" if ap is not None else f"{names[class_id]}=n/a"
        for class_id, ap in sorted(ap_per_class.items())
    )


def train(labels: LabelSet, train_loader: DataLoader, val_loader: DataLoader, device: str) -> None:
    torch.manual_seed(SEED)
    n_classes = len(labels)

    model = ASTDeformableDETR(dim=MODEL_DIM, n_queries=N_QUERIES, n_classes=n_classes).to(device)
    matcher = HungarianMatcher(iou_type=MATCHER_IOU_TYPE)
    criterion = SetCriterion(n_classes=n_classes, matcher=matcher, iou_type=MATCHER_IOU_TYPE).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )
    logger.info(
        "%.1fM parámetros | %d clases",
        sum(p.numel() for p in model.parameters()) / 1e6,
        n_classes,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = f"{MODEL_DIM}d_{N_QUERIES}q_{MATCHER_IOU_TYPE}iou_{n_classes}cls"
    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"
    labels_path = CHECKPOINT_DIR / f"{name}_labels.json"

    save_labels_json(labels, labels_path)
    logger.info(
        "class_id -> label -> %s",
        {class_id: label for class_id, label in enumerate(labels.names)},
    )

    # (recall_agn, -fp_per_tp): compara por recall primero, FP/TP como desempate.
    best_score, best_state = (-1.0, float("-inf")), None

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch=epoch,
            epochs=EPOCHS,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            matcher,
            device,
            n_classes=n_classes,
            iou_threshold=METRIC_IOU_THRESHOLD,
            ap_iou_type=METRIC_IOU_TYPE,
            score_threshold=OPERATING_SCORE_THRESHOLD,
            epoch=epoch,
            epochs=EPOCHS,
        )
        logger.info(
            "[%4d/%d] train=%.3f val=%.3f cls_acc=%.3f IoU=%.3f recall_agn@%.2f=%.3f FP/TP=%.2f",
            epoch + 1,
            EPOCHS,
            train_metrics["total"],
            val_metrics.total,
            val_metrics.cls_acc,
            val_metrics.mean_iou,
            METRIC_IOU_THRESHOLD,
            val_metrics.recall_agn,
            val_metrics.fp_per_tp,
        )

        # recall_agn es la métrica objetivo (más alto mejor); fp_per_tp desempata
        # (más bajo mejor) -- ver "Objetivo" arriba y el docstring de `evaluate`.
        score = (val_metrics.recall_agn, -val_metrics.fp_per_tp)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "labels": labels.names,
                    "dim": MODEL_DIM,
                    "n_queries": N_QUERIES,
                    "epoch": epoch,
                    "recall_agn": val_metrics.recall_agn,
                    "fp_per_tp": val_metrics.fp_per_tp,
                    "map": val_metrics.map,
                },
                checkpoint_path,
            )

        if (epoch + 1) % EVAL_EVERY and epoch + 1 != EPOCHS:
            continue

        logger.info("AP por clase -> %s", format_ap(val_metrics.ap_per_class, labels.names))
        logger.info(
            "AP agnóstico de clase -> %s",
            ", ".join(
                f"{threshold}={ap:.3f}" if ap is not None else f"{threshold}=n/a"
                for threshold, ap in sorted(val_metrics.ap_agnostic.items())
            ),
        )
        logger.info(
            "matriz de confusión (queries emparejadas)\n%s",
            format_confusion(val_metrics.confusion, labels.names),
        )

    if best_state is None:
        raise RuntimeError("No se completó ninguna época.")
    best_recall, best_fp_per_tp = best_score[0], -best_score[1]
    logger.info(
        "mejor recall_agn@%.2f de validación: %.3f (FP/TP=%.2f) -> %s",
        METRIC_IOU_THRESHOLD,
        best_recall,
        best_fp_per_tp,
        checkpoint_path,
    )


if __name__ == "__main__":
    setup_logging(settings.LOG_LEVEL)
    setup_project_path(PROJECT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    labels, train_loader, val_loader = preprocess()
    train(labels, train_loader, val_loader, device)
