import logging
import threading
from collections.abc import Callable
from typing import override

from PyQt6.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal

logger = logging.getLogger("viewer")


class StopError(Exception):
    pass


class UnreadableError(Exception):
    pass


def describe(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, UnreadableError) else f"{type(exc).__name__}: {exc}"


class Worker(QThread):
    ok = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, task, reports: bool = False, what: str = "Background task") -> None:
        # reports=True: la tarea recibe un callback (hechos, total); stop() la corta en el siguiente
        super().__init__()
        self.task = task
        self.reports = reports
        self.what = what
        self.stopping = threading.Event()
        # ok y error salen antes de que isRunning baje
        self.done = False

    def stop(self) -> None:
        self.stopping.set()

    def report(self, done: int, total: int) -> None:
        if self.stopping.is_set():
            raise StopError
        self.progress.emit(done, total)

    @override
    def run(self) -> None:
        try:
            result = self.task(self.report) if self.reports else self.task()
        except StopError:
            self.done = True
        except UnreadableError as exc:
            cause = f" ({exc.__cause__})" if exc.__cause__ is not None else ""
            logger.warning("%s: %s%s", self.what, exc, cause)
            self.done = True
            self.error.emit(describe(exc))
        except Exception as exc:
            logger.exception("%s failed", self.what)
            self.done = True
            self.error.emit(describe(exc))
        else:
            self.done = True
            self.ok.emit(result)


# Un trabajo a la vez: si llega otro mientras calcula, el que esperaba se descarta
class Latest(QThread):
    done = pyqtSignal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self.mutex = QMutex()
        self.waiting = QWaitCondition()
        self.job: tuple[int, Callable[[], object]] | None = None
        self.counter = 0
        self.closing = False
        self.start()

    def submit(self, task: Callable[[], object]) -> int:
        self.mutex.lock()
        self.counter += 1
        job_id = self.counter
        self.job = (job_id, task)
        self.waiting.wakeOne()
        self.mutex.unlock()
        return job_id

    def close(self) -> None:
        self.mutex.lock()
        self.closing = True
        self.waiting.wakeOne()
        self.mutex.unlock()
        self.wait()

    @override
    def run(self) -> None:
        while True:
            self.mutex.lock()
            while self.job is None and not self.closing:
                self.waiting.wait(self.mutex)
            if self.closing or self.job is None:
                self.mutex.unlock()
                if self.closing:
                    return
                continue  # despertar espurio: no hay nada que calcular
            job_id, task = self.job
            self.job = None
            self.mutex.unlock()
            try:
                self.done.emit(job_id, task())
            except Exception:
                logger.warning("Spectrogram render failed", exc_info=True)
                self.done.emit(job_id, None)
