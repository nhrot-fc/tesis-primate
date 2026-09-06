from collections.abc import Callable
from typing import override

from PyQt6.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal


class Worker(QThread):
    ok = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, task, reports: bool = False) -> None:
        # Con `reports=True` la tarea recibe un callback (hechos, total) para el progreso.
        super().__init__()
        self.task = task
        self.reports = reports

    @override
    def run(self) -> None:
        try:
            self.ok.emit(self.task(self.progress.emit) if self.reports else self.task())
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


# Hilo de un solo trabajo a la vez: si llega otro mientras calcula, el que esperaba se
# descarta. Es lo que mantiene fluido el scroll del espectrograma.
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
                self.done.emit(job_id, None)  # una banda fallida no merece una alerta
