"""Pasa un modelo por una lista de grabaciones y deja la tabla de Raven de cada una junto al
audio. Lo comparten el CLI (`detect.py`) y el panel Recordings del visor: un archivo a la vez, la
tabla se escribe al terminarlo y las cajas se descartan, así la memoria no crece con la lista."""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from inference.catalog import output_for
from inference.predictor import BATCH_SIZE, predict
from models.registry import LoadedModel

DONE, SKIPPED, FAILED, STOPPED = "done", "skipped", "failed", "stopped"
# En CPU el lote grande sólo sube el pico de memoria (Faster R-CNN: 4,8 GB con 16)
CPU_BATCH_SIZE = 4


@dataclass(frozen=True)
class Outcome:
    index: int
    path: Path
    status: str
    detections: int = 0
    seconds: float = 0.0  # lo que tardó, sólo en `done`
    message: str = ""  # el error, sólo en `failed`


class StopError(Exception):
    pass


def batch_size_for(device: str | torch.device) -> int:
    return BATCH_SIZE if str(device).startswith("cuda") else CPU_BATCH_SIZE


def run_batch(
    loaded: LoadedModel,
    files: list[Path],
    device: str | torch.device,
    threshold: float,
    overwrite: bool = False,
    on_progress: Callable[[int, int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[Outcome]:
    # `on_progress(índice, ventanas hechas, ventanas totales)` va por archivo; `should_stop`
    # se consulta entre archivos y entre lotes de ventanas, y corta con `stopped`.
    batch_size = batch_size_for(device)
    for index, path in enumerate(files):
        if should_stop is not None and should_stop():
            yield Outcome(index, path, STOPPED)
            return
        output = output_for(path)
        if output.is_file() and not overwrite:
            yield Outcome(index, path, SKIPPED)
            continue

        def report(done: int, total: int, index: int = index) -> None:
            if should_stop is not None and should_stop():
                raise StopError
            if on_progress is not None:
                on_progress(index, done, total)

        started = time.perf_counter()
        try:
            table = predict(
                loaded,
                path,
                device,
                score_threshold=threshold,
                batch_size=batch_size,
                on_progress=report,
            )
            table.to_csv(output, sep="\t", index=False)
        except StopError:
            yield Outcome(index, path, STOPPED)
            return
        except Exception as exc:
            yield Outcome(index, path, FAILED, message=f"{type(exc).__name__}: {exc}")
            continue
        yield Outcome(index, path, DONE, len(table), time.perf_counter() - started)
