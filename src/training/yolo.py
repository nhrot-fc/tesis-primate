import json
import logging
from pathlib import Path
from typing import Any

from core.config import SEED, settings
from data import cache
from data.species import LabelSet
from models.registry import architecture
from models.yolo import SpectrogramYOLO, load_ultralytics_weights
from training import checkpoint
from training.trainer import TrainConfig

logger = logging.getLogger(__name__)

PATIENCE = 10

# Aumentaciones: espectrograma, no foto.
AUGMENTATION = {
    "hsv_h": 0.0,  # el gris no tiene tono ni saturación
    "hsv_s": 0.0,
    "hsv_v": 0.2,  # brillo == ganancia de grabación: la única que sí varía en el campo
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "translate": 0.05,  # corrimiento chico en tiempo y frecuencia
    "scale": 0.2,
    "fliplr": 0.0,  # invertir el tiempo cambia la llamada
    "flipud": 0.0,  # invertir la frecuencia también
    "bgr": 0.0,
    "mosaic": 0.0,  # pega 4 clips: crea bordes y solapes que no existen en el audio
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
}


def check_export() -> dict:
    # El dataset de YOLO es una reexportación del caché: si el caché se regeneró, entrenar
    # contra el export viejo invalida la comparación.
    data_yaml = settings.yolo_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"no hay dataset YOLO en {settings.yolo_dir}. Corré `python src/export_yolo.py`."
        )
    meta = json.loads((settings.yolo_dir / "meta.json").read_text())
    if meta["dataset"] != cache.meta():
        raise ValueError(
            f"{settings.yolo_dir} se exportó de otra versión del caché. Volvé a correr "
            "`python src/export_yolo.py`."
        )
    return meta


def export_checkpoint(
    run_dir: Path, weights: Path, labels: LabelSet, hparams: dict[str, Any], config: dict
) -> Path:
    # Ultralytics guarda el modelo pickleado; el registro guarda arquitectura, hiperparámetros
    # y `state_dict`, que es lo que `load_checkpoint` abre.
    model = SpectrogramYOLO(n_classes=len(labels), **hparams)
    load_ultralytics_weights(model, weights)
    path = run_dir / checkpoint.BEST
    checkpoint.save(
        path,
        architecture="yolo",
        model=model,
        hparams=hparams,
        labels=labels,
        config=config,
        epoch=config["epochs"] - 1,
        metrics={},  # las de Ultralytics están en results.csv; las comparables, de compare_models.py
    )
    return path


def fit(
    run_dir: Path,
    hparams: dict[str, Any],  # `model` (yolo26s, yolo26m...) e `imgsz`
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

    last = run_dir / "weights" / "last.pt"
    resume = last.is_file()
    if resume:
        logger.info("retomando %s", last)
    trainer = YOLO(str(last) if resume else f"{stem}.pt")  # siempre desde los pesos de COCO
    n_train = meta["counts"]["train"]["windows"]
    results = trainer.train(
        data=str(settings.yolo_dir / "dataset.yaml"),
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

    # YOLO26 es end-to-end (sin NMS), pero el pos-proceso de cajas anidadas se le aplica igual
    # que a los demás para que el punto de operación sea el mismo.
    full_hparams = {
        "model": stem,
        "imgsz": hparams["imgsz"],
        "db_low": meta["db_range"]["low"],
        "db_high": meta["db_range"]["high"],
    }
    run_config = {
        "architecture": "yolo",
        "hparams": full_hparams,
        "dataset": meta["dataset"],
        "nms_iou": architecture("yolo").nms_iou,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": SEED,
        "patience": PATIENCE,
        "augmentation": AUGMENTATION,
        "score_threshold": config.score_threshold,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    path = export_checkpoint(
        run_dir, run_dir / "weights" / "best.pt", labels, full_hparams, run_config
    )
    logger.info("checkpoint -> %s", path)
    return path
