import argparse
import gc
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from architectures.backbone import AST_CHECKPOINT, load_ast_model
from core.config import P, settings
from core.setup import setup_logging, setup_project_path
from pipelines.common import format_metric
from train import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    FREEZE_BACKBONE,
    FRONTEND,
    LEARNING_RATE,
    LOG_DIR,
    NUM_WORKERS,
    PROJECT_DIR,
    load_datasets,
    model_hparams,
    resolve_device,
    run_name,
    train,
    training_config,
)

logger = logging.getLogger("sweep")

STRIDES = (10, 5, 2)
PATCH_SIZE, FREQUENCY_STRIDE = 16, 10  # los del checkpoint del AST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el AST + Deformable-DETR con varios `time_stride` en una corrida."
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=list(STRIDES),
        help="pasos temporales a barrer, en orden (por defecto: 10 5 2, del más barato al más caro)",
    )
    # Se pasan tal cual a `src/train.py` y valen para las tres corridas.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="'cuda', 'cuda:1', 'cpu'")
    parser.add_argument("--frontend", choices=("pcen", "logmel"), default=None)
    parser.add_argument("--unfreeze", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="usa sólo N ventanas (pruebas)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="saltea las combinaciones cuyo checkpoint ya existe (para retomar un barrido)",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="al terminar, mide cada checkpoint sobre el split de test",
    )
    parser.add_argument("--tag", default=None, help="nombre de la carpeta del barrido")
    return parser.parse_args()


def run_args(args: argparse.Namespace, time_stride: int) -> argparse.Namespace:
    """Los argumentos que espera `train.train`, con los defaults de `src/train.py`.

    Los defaults se leen del módulo y no se copian acá: si allá cambia el número de
    épocas, el barrido lo sigue solo.
    """

    def pick(name: str, default: Any) -> Any:
        value = getattr(args, name, None)
        return default if value is None else value

    return argparse.Namespace(
        epochs=pick("epochs", EPOCHS),
        batch=pick("batch", BATCH_SIZE),
        lr=pick("lr", LEARNING_RATE),
        workers=pick("workers", NUM_WORKERS),
        frontend=pick("frontend", FRONTEND),
        unfreeze=args.unfreeze or not FREEZE_BACKBONE,
        time_stride=time_stride,
        device=args.device,
        limit=args.limit,
        name=None,  # que `run_name` lo derive de la ablación
    )


def token_grid(time_stride: int) -> tuple[int, int]:
    freq_out = (P.n_mels - PATCH_SIZE) // FREQUENCY_STRIDE + 1
    time_out = (P.n_frames - PATCH_SIZE) // time_stride + 1
    return freq_out, time_out


@dataclass
class RunResult:
    time_stride: int
    name: str
    status: str = "pendiente"
    tokens: str = ""
    checkpoint: str = ""
    run_dir: str = ""
    minutes: float | None = None
    epoch: int | None = None
    map_50: float | None = None
    map_50_95: float | None = None
    recall_agn: float | None = None
    precision_agn: float | None = None
    test: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def read_checkpoint_metrics(path: Path, result: RunResult) -> None:
    """Las métricas de la mejor época viajan en el checkpoint; no hay que reevaluar."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result.epoch = checkpoint.get("epoch")
    if result.epoch is not None:
        result.epoch += 1  # `save_checkpoint` guarda el índice, no el número de época
    for key in ("map_50", "map_50_95", "recall_agn", "precision_agn"):
        setattr(result, key, checkpoint.get(key))


def summary_table(results: list[RunResult]) -> str:
    header = (
        "| paso | tokens | estado | época | mAP@0.5 | mAP@0.5:0.95 | recall_agn | "
        "precision_agn | minutos |"
    )
    rows = [header, "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(
            f"| {r.time_stride} | {r.tokens} | {r.status} | {r.epoch or '-'} | "
            f"{format_metric(r.map_50)} | {format_metric(r.map_50_95)} | "
            f"{format_metric(r.recall_agn)} | {format_metric(r.precision_agn)} | "
            f"{'-' if r.minutes is None else f'{r.minutes:.1f}'} |"
        )
    return "\n".join(rows)


def write_summary(sweep_dir: Path, results: list[RunResult], args: argparse.Namespace) -> None:
    """Se reescribe después de cada corrida: si el barrido se corta, lo hecho queda."""
    (sweep_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "runs": [asdict(r) for r in results],
                "generated": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    (sweep_dir / "summary.md").write_text(
        f"# Barrido de `time_stride`\n\n{summary_table(results)}\n"
    )


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_on_test(results: list[RunResult], device: str, batch_size: int = 16) -> None:
    """Mide los checkpoints del barrido sobre test, con el mismo código que eval_detector."""
    from torch.utils.data import DataLoader

    from architectures.registry import load_checkpoint
    from domain.dataset import CachedCallBoxDataset, collate_fn
    from eval_detector import IOU_THRESHOLD, format_report
    from pipelines.detection_pipeline import collect_detections
    from pipelines.metrics import AP_THRESHOLDS, detection_metrics
    from train import CACHE_DIR

    done = [r for r in results if r.status == "ok"]
    if not done:
        logger.warning("no hay checkpoints que evaluar sobre test")
        return

    logger.info("cargando el split de test para evaluar %d checkpoint(s)", len(done))
    dataset = CachedCallBoxDataset(CACHE_DIR / "test.pt")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for result in done:
        try:
            loaded = load_checkpoint(Path(result.checkpoint), device)
            predictions, truth = collect_detections(
                loaded.detect, loader, device, loaded.nms_iou, desc=f"test ts{result.time_stride}"
            )
            metrics = detection_metrics(
                predictions,
                truth,
                n_classes=len(loaded.labels),
                iou_threshold=IOU_THRESHOLD,
                score_threshold=loaded.score_threshold,
                ap_thresholds=AP_THRESHOLDS,
            )
            report = format_report(
                Path(result.checkpoint),
                loaded.architecture,
                "test",
                len(dataset),
                loaded.labels.names,
                metrics,
                loaded.score_threshold,
            )
            Path(result.checkpoint).with_name(
                f"{Path(result.checkpoint).stem}_test_metrics.txt"
            ).write_text(report)
            result.test = {
                "map_50": metrics.map_50,
                "map_50_95": metrics.map_50_95,
                "recall_agnostic": metrics.recall_agnostic,
                "precision_agnostic": metrics.precision_agnostic,
            }
            logger.info(
                "test | paso %d -> mAP50=%s mAP50-95=%s recall_agn=%s",
                result.time_stride,
                format_metric(metrics.map_50),
                format_metric(metrics.map_50_95),
                format_metric(metrics.recall_agnostic),
            )
        except Exception as error:  # una evaluación caída no invalida el barrido
            logger.exception("falló la evaluación sobre test del paso %d", result.time_stride)
            result.test = {"error": f"{type(error).__name__}: {error}"}
        finally:
            free_memory()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = LOG_DIR / (args.tag or f"sweep_time_stride_{stamp}")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(settings.LOG_LEVEL, log_file=sweep_dir / "sweep.log")
    setup_project_path(PROJECT_DIR)

    device = resolve_device(args.device)
    logger.info("device: %s | barrido de time_stride %s", device, args.strides)

    # Bajar el AST antes de la primera corrida: si falta la copia local y no hay red, es
    # mejor enterarse ahora que después de cargar 5 GB de caché.
    load_ast_model(AST_CHECKPOINT)

    labels, _meta, cached_train, cached_val = load_datasets()
    train_dataset: Dataset = cached_train
    val_dataset: Dataset = cached_val
    if args.limit:
        train_dataset = Subset(cached_train, range(min(args.limit, len(cached_train))))
        val_dataset = Subset(cached_val, range(min(args.limit, len(cached_val))))

    results: list[RunResult] = []
    for time_stride in args.strides:
        one = run_args(args, time_stride)
        name = run_name(one)
        checkpoint_path = CHECKPOINT_DIR / f"{name}_{len(labels)}cls_best.pth"
        freq_out, time_out = token_grid(time_stride)
        result = RunResult(
            time_stride=time_stride,
            name=name,
            tokens=f"{freq_out}x{time_out}",
            checkpoint=str(checkpoint_path),
        )
        results.append(result)

        if args.skip_existing and checkpoint_path.exists():
            result.status = "salteada"
            logger.info("[%s] ya existe %s: se saltea", name, checkpoint_path)
            read_checkpoint_metrics(checkpoint_path, result)
            write_summary(sweep_dir, results, args)
            continue

        run_dir = sweep_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        result.run_dir = str(run_dir)
        (run_dir / "config.json").write_text(
            json.dumps(training_config(one), indent=2, ensure_ascii=False, default=str)
        )

        # Copia del log de esta corrida en su carpeta, sin perder el log del barrido.
        handler = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.root.handlers[0].formatter)
        logging.root.addHandler(handler)

        logger.info(
            "===== paso %d (%s) | %d x %d tokens | %s =====",
            time_stride,
            name,
            freq_out,
            time_out,
            model_hparams(one),
        )
        started = time.monotonic()
        try:
            train(labels, train_dataset, val_dataset, device, run_dir / "metrics.jsonl", one)
            result.status = "ok"
            read_checkpoint_metrics(checkpoint_path, result)
        except KeyboardInterrupt:
            result.status = "interrumpida"
            logger.warning("barrido interrumpido a mano durante el paso %d", time_stride)
            write_summary(sweep_dir, results, args)
            raise
        except Exception as error:
            # Que una combinación se caiga --típicamente por VRAM con paso 2-- no puede
            # llevarse puesto el resto del barrido.
            result.status = "falló"
            result.error = f"{type(error).__name__}: {error}"
            logger.error("el paso %d falló: %s", time_stride, result.error)
            logger.debug("%s", traceback.format_exc())
        finally:
            result.minutes = (time.monotonic() - started) / 60
            logging.root.removeHandler(handler)
            handler.close()
            free_memory()

        logger.info("[%s] %s en %.1f min", name, result.status, result.minutes)
        write_summary(sweep_dir, results, args)

    if args.eval_test:
        evaluate_on_test(results, device)
        write_summary(sweep_dir, results, args)

    logger.info("resumen del barrido:\n%s", summary_table(results))
    logger.info("resumen -> %s", sweep_dir / "summary.md")


if __name__ == "__main__":
    main()
