import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import YOLO_DIR
from data import cache
from data.species import LabelSet
from models.yolo import SpectrogramYOLO, load_ultralytics_weights
from training import checkpoint
from training.trainer import TrainConfig

logger = logging.getLogger(__name__)

PATIENCE = 10
# Lo que deja Ultralytics en la corrida. Coinciden con `checkpoint.BEST`/`LAST` por casualidad:
# son sus nombres, no los del proyecto.
ULTRALYTICS_BEST = Path("weights") / "best.pt"
ULTRALYTICS_LAST = Path("weights") / "last.pt"
RESULTS = "results.csv"
# Fitness de Ultralytics: con ella elige `best.pt`
FITNESS = {"metrics/mAP50(B)": 0.1, "metrics/mAP50-95(B)": 0.9}

# Aumentaciones: espectrograma, no foto.
AUGMENTATION = {
    "hsv_h": 0.0,  # el gris no tiene tono ni saturación
    "hsv_s": 0.0,
    "hsv_v": 0.2,  # brillo == ganancia de grabación
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "translate": 0.05,  # corrimiento chico en tiempo y frecuencia
    "scale": 0.2,
    "fliplr": 0.0,  # invertir el tiempo cambia la llamada
    "flipud": 0.0,  # invertir la frecuencia también
    "bgr": 0.0,
    "mosaic": 0.0,  # pegar 4 clips crea bordes que no existen en el audio
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
}


def check_export() -> dict:
    # El export tiene que ser del caché actual.
    if not (YOLO_DIR / "dataset.yaml").exists():
        raise FileNotFoundError(
            f"no hay dataset YOLO en {YOLO_DIR}; corré `python src/export_yolo.py`"
        )
    meta = json.loads((YOLO_DIR / "meta.json").read_text())
    if meta["dataset"] != cache.meta():
        raise ValueError(
            f"{YOLO_DIR} es de otra versión del caché; corré `python src/export_yolo.py`"
        )
    return meta


def best_epoch(run_dir: Path) -> tuple[int, dict[str, float]]:
    # -> (época de `best.pt`, desde 0, y sus métricas de Ultralytics). Con `patience` la corrida
    # puede parar antes de `epochs`: la época real sale de results.csv, no de la config.
    results = pd.read_csv(run_dir / RESULTS)
    results.columns = [column.strip() for column in results.columns]
    fitness = (results[list(FITNESS)] * pd.Series(FITNESS)).sum(axis=1)
    best = results.iloc[int(fitness.idxmax())]
    metrics = {c: float(best[c]) for c in results.columns if c.startswith(("metrics/", "val/"))}
    return int(best["epoch"]) - 1, metrics


def export_checkpoint(
    run_dir: Path, labels: LabelSet, hparams: dict[str, Any], config: dict
) -> Path:
    # Al formato del proyecto, que es el que `load_checkpoint` abre.
    model = SpectrogramYOLO(n_classes=len(labels), **hparams)
    load_ultralytics_weights(model, run_dir / ULTRALYTICS_BEST)
    epoch, metrics = best_epoch(run_dir)
    path = run_dir / checkpoint.BEST
    checkpoint.save(
        path,
        architecture="yolo",
        model=model,
        hparams=hparams,
        labels=labels,
        config=config,
        epoch=epoch,
        metrics=metrics,
    )
    return path


def fit(
    run_dir: Path,
    hparams: dict[str, Any],  # `model` (yolo26s, yolo26m...), `imgsz`, `db_low`, `db_high`
    config: TrainConfig,
    device: str,
    limit: int | None = None,
) -> Path:
    from ultralytics import YOLO

    meta = check_export()
    labels = cache.labels()
    stem = Path(hparams["model"]).stem
    if meta["image_size"] != hparams["imgsz"]:
        raise ValueError(
            f"el export es de {meta['image_size']} px y el modelo pide {hparams['imgsz']}"
        )

    last = run_dir / ULTRALYTICS_LAST
    resume = last.is_file()
    if resume:
        logger.info("retomando %s", last)
    trainer = YOLO(str(last) if resume else f"{stem}.pt")  # desde los pesos de COCO
    n_train = meta["counts"][cache.TRAIN]["windows"]
    results = trainer.train(
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=hparams["imgsz"],
        workers=config.workers,
        device=device,
        fraction=min(1.0, limit / n_train) if limit else 1.0,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        resume=resume,
        seed=config.seed,
        deterministic=True,
        patience=PATIENCE,
        optimizer="auto",
        cos_lr=True,
        val=True,
        plots=True,
        **AUGMENTATION,
    )
    if results is None:
        raise RuntimeError("Ultralytics no devolvió resultados de entrenamiento.")

    full_hparams = {**hparams, "model": stem}
    run_config = {
        "architecture": "yolo",
        "hparams": full_hparams,
        "dataset": meta["dataset"],
        "nms_iou": SpectrogramYOLO.nms_iou,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "patience": PATIENCE,
        "augmentation": AUGMENTATION,
        "score_threshold": config.score_threshold,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    path = export_checkpoint(run_dir, labels, full_hparams, run_config)
    logger.info("checkpoint -> %s", path)
    return path
