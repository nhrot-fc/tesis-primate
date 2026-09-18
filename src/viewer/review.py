"""Revisión caja por caja de las detecciones: aceptar (con el encuadre y la especie que se hayan
retocado) o rechazar. Cada decisión se apunta al momento en un archivo temporal por grabación
—una tabla de Raven con `Score` y `Decision`— y al volver a revisar la misma grabación se
retoma desde ahí: lo aceptado vuelve a Annotations y lo decidido sale de Detections."""

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import override

import pandas as pd
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget

from data.raven import BEGIN, CALL, END, HIGH, LOW, SCORE, SPECIES
from viewer.plot import SpectrogramView
from viewer.session import ANNOTATIONS, DETECTIONS, SOURCES, Row, Session

DECISION = "Decision"
ACCEPTED, REJECTED = "accepted", "rejected"
JOURNAL_COLUMNS = [BEGIN, END, LOW, HIGH, SPECIES, CALL, SCORE, DECISION]
JOURNAL_DIR = Path(tempfile.gettempdir()) / "primate-detector"
SPECIES_WIDTH = 150
# Una decisión del diario se casa con la detección que más se le parece; por debajo de esto
# (la caja se retocó mucho o la tabla es otra) se deja la detección en la cola.
RESUME_IOU = 0.5

Box = tuple[float, float, float, float]


# El diario de una grabación vive en la carpeta temporal del sistema, con nombre fijo por
# ruta de audio: si el visor se cierra, al volver a abrir el audio se sigue sobre el mismo.
def journal_path(audio: Path) -> Path:
    digest = hashlib.md5(str(audio.resolve()).encode()).hexdigest()[:8]
    return JOURNAL_DIR / f"{audio.stem}.{digest}.review.txt"


def iou(a: Box, b: Box) -> float:
    width = min(a[1], b[1]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[2], b[2])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


class Journal:
    def __init__(self, audio: Path) -> None:
        self.path = journal_path(audio)

    def append(self, decision: str, box: Box, label: str, score: float) -> None:
        species, _, call = label.partition("/")
        row = pd.DataFrame([[*box, species, call, score, decision]], columns=JOURNAL_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(self.path, sep="\t", index=False, mode="a", header=not self.path.exists())

    def read(self) -> pd.DataFrame:
        if not self.path.is_file():
            return pd.DataFrame(columns=JOURNAL_COLUMNS)
        return pd.read_csv(self.path, sep="\t", keep_default_na=False)

    def counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.read()[DECISION].value_counts().items()}


# La fila de mandos: dónde va la revisión, la especie de la caja (se puede corregir antes de
# aceptar) y las decisiones. Enter en la especie acepta: se tipea y se sigue.
class ReviewBar(QWidget):
    accepted = pyqtSignal()
    rejected = pyqtSignal()
    done = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.position = QLabel("")
        self.position.setObjectName("hint")
        self.species = QComboBox()
        self.species.setEditable(True)
        self.species.setFixedWidth(SPECIES_WIDTH)
        self.species.setToolTip("Species/call the box is saved with; type to correct it")
        self.species.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        edit = self.species.lineEdit()
        if edit is not None:
            edit.installEventFilter(self)
        self.accept_button = self.button(
            "Accept", "Keep the box as framed and labelled (A, Enter)", self.accepted.emit
        )
        self.accept_button.setObjectName("primary")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.position)
        layout.addSpacing(8)
        layout.addWidget(QLabel("as"))
        layout.addWidget(self.species)
        layout.addWidget(self.button("Reject", "Drop the box (R, Delete)", self.rejected.emit))
        layout.addWidget(self.accept_button)
        layout.addSpacing(8)
        layout.addWidget(self.button("Done", "Leave the review (Esc)", self.done.emit))

    def button(self, text: str, tip: str, action: Callable[[], None]) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda: action())
        return button

    # Enter en la especie acepta una sola vez: se consume acá para que ni el combo ni la
    # ventana (que también acepta con Enter) lo vuelvan a ver.
    @override
    def eventFilter(self, a0, a1) -> bool:
        if (
            isinstance(a1, QKeyEvent)
            and a1.type() == QEvent.Type.KeyPress
            and a1.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            self.accepted.emit()
            return True
        return super().eventFilter(a0, a1)

    def show_row(self, row: Row, index: int, total: int, labels: list[str]) -> None:
        score = "" if pd.isna(row.score) else f" · {row.score:.2f}"
        self.position.setText(f"{index + 1} of {total}{score}")
        self.species.blockSignals(True)
        self.species.clear()
        self.species.addItems(labels)
        self.species.setCurrentText(row.label)
        self.species.blockSignals(False)

    def label(self) -> str:
        return self.species.currentText().strip()


# Recorre las detecciones visibles en orden de tiempo. Aceptar mueve la caja a Annotations con
# el encuadre y la especie que tenga en ese momento; rechazar la quita; las dos quedan en el
# diario. La cola se recorta sola: lo decidido deja de estar en Detections.
class Reviewer(QObject):
    changed = pyqtSignal()

    def __init__(
        self,
        session: Session,
        plot: SpectrogramView,
        bar: ReviewBar,
        frame: Callable[[Row], None],
    ) -> None:
        super().__init__()
        self.session = session
        self.plot = plot
        self.bar = bar
        self.frame = frame
        self.active = False
        self.cursor = 0
        self.journal: Journal | None = None
        self.resumed = 0  # decisiones retomadas del diario al empezar
        bar.accepted.connect(lambda: self.decide(ACCEPTED))
        bar.rejected.connect(lambda: self.decide(REJECTED))
        bar.done.connect(self.stop)
        session.changed.connect(self.refresh)

    def rows(self) -> list[Row]:
        return self.session.rows(DETECTIONS)

    def current(self) -> Row | None:
        rows = self.rows()
        return rows[self.cursor] if self.active and rows else None

    def start(self) -> None:
        if self.session.audio_path is None:
            return
        self.journal = Journal(self.session.audio_path)
        # Antes de activarse: lo que `resume` agrega y quita no debe redibujar la revisión.
        self.resumed = self.resume()
        self.active = True
        self.cursor = 0
        self.show()

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.plot.set_editable(None)
        self.changed.emit()

    # Lo que el diario ya decidió sobre estas detecciones: cada fila se casa con la detección
    # más parecida y la saca de la cola; las aceptadas vuelven a Annotations si no están ya.
    def resume(self) -> int:
        if self.journal is None:
            return 0
        decided = self.journal.read()
        detections = self.session.tables[DETECTIONS]
        if decided.empty or detections is None or detections.empty:
            return 0
        pending = {
            index: (row[BEGIN], row[END], row[LOW], row[HIGH])
            for index, row in detections.iterrows()
        }
        annotations = self.session.tables[ANNOTATIONS]
        present = set()
        if annotations is not None:
            present = {
                tuple(round(v, 3) for v in (row[BEGIN], row[END], row[LOW], row[HIGH]))
                for _, row in annotations.iterrows()
            }
        matched, restored = [], 0
        for _, row in decided.iterrows():
            box = (row[BEGIN], row[END], row[LOW], row[HIGH])
            best = max(pending, key=lambda i: iou(box, pending[i]), default=None)
            if best is None or iou(box, pending[best]) < RESUME_IOU:
                continue
            matched.append((DETECTIONS, best))
            del pending[best]
            restored += 1
            if row[DECISION] == ACCEPTED and tuple(round(v, 3) for v in box) not in present:
                self.session.add(
                    ANNOTATIONS,
                    {
                        BEGIN: box[0],
                        END: box[1],
                        LOW: box[2],
                        HIGH: box[3],
                        SPECIES: row[SPECIES],
                        CALL: row[CALL],
                    },
                )
        if matched:
            self.session.remove(matched)
        return restored

    def refresh(self) -> None:
        if not self.active:
            return
        if self.session.audio_path is None or (
            self.journal is not None and self.journal.path != journal_path(self.session.audio_path)
        ):
            self.stop()  # cambió el audio: la revisión era de otro
            return
        self.show()

    def show(self) -> None:
        rows = self.rows()
        if not rows:
            self.stop()  # cola vacía: la revisión terminó
            return
        self.cursor = min(max(self.cursor, 0), len(rows) - 1)
        row = rows[self.cursor]
        labels = sorted(
            {r.label for source in SOURCES for r in self.session.rows(source) if r.label}
        )
        self.bar.show_row(row, self.cursor, len(rows), labels)
        self.frame(row)
        self.plot.set_editable((row.begin, row.end, row.low, row.high))
        self.changed.emit()

    def step(self, delta: int) -> None:
        if not self.active or not self.rows():
            return
        self.cursor = min(max(self.cursor + delta, 0), len(self.rows()) - 1)
        self.show()

    def decide(self, decision: str) -> None:
        row = self.current()
        if row is None or self.journal is None:
            return
        box = (
            self.plot.editable_box()
            if decision == ACCEPTED
            else (row.begin, row.end, row.low, row.high)
        )
        label = self.bar.label() or row.label
        self.journal.append(decision, box, label, row.score)
        if decision == ACCEPTED:
            species, _, call = label.partition("/")
            self.session.add(
                ANNOTATIONS,
                {
                    BEGIN: box[0],
                    END: box[1],
                    LOW: box[2],
                    HIGH: box[3],
                    SPECIES: species,
                    CALL: call,
                },
            )
        # Quitarla de Detections dispara `changed` -> `refresh` -> `show` con la siguiente
        self.session.remove([(row.source, row.index)])

    def counts(self) -> dict[str, int]:
        return {} if self.journal is None else self.journal.counts()
