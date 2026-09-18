"""Revisión caja por caja de las detecciones: aceptar (con el encuadre y la especie que se hayan
retocado), rechazar o saltar. Cada decisión se apunta al momento en un archivo temporal por
grabación —una tabla de Raven con `Score` y `Decision`— para no perder nada si el visor se
cierra, y `Save → Reviewed boxes…` vuelca las aceptadas como tabla de anotaciones."""

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget

from data.raven import BEGIN, CALL, END, HIGH, LOW, SCORE, SPECIES, raven_table
from viewer.plot import SpectrogramView
from viewer.session import ANNOTATIONS, DETECTIONS, SOURCES, Row, Session

DECISION = "Decision"
ACCEPTED, REJECTED = "accepted", "rejected"
JOURNAL_COLUMNS = [BEGIN, END, LOW, HIGH, SPECIES, CALL, SCORE, DECISION]
JOURNAL_DIR = Path(tempfile.gettempdir()) / "primate-detector"
SPECIES_WIDTH = 150
POSITION_WIDTH = 190


# El diario de una grabación vive en la carpeta temporal del sistema, con nombre fijo por
# ruta de audio: si el visor se cierra, al volver a abrir el audio se sigue sobre el mismo.
def journal_path(audio: Path) -> Path:
    digest = hashlib.md5(str(audio.resolve()).encode()).hexdigest()[:8]
    return JOURNAL_DIR / f"{audio.stem}.{digest}.review.txt"


class Journal:
    def __init__(self, audio: Path) -> None:
        self.path = journal_path(audio)

    def append(
        self, decision: str, box: tuple[float, float, float, float], label: str, score: float
    ) -> None:
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

    # Las aceptadas como tabla de anotaciones de Raven: sin `Score`, que las volvería detecciones.
    def accepted(self) -> pd.DataFrame:
        rows = self.read()
        rows = rows[rows[DECISION] == ACCEPTED]
        table = raven_table(
            rows[BEGIN], rows[END], rows[LOW], rows[HIGH], rows[SPECIES], rows[CALL], rows[SCORE]
        )
        return table.drop(columns=[SCORE])


# La fila de mandos: dónde va la revisión, la especie de la caja (se puede corregir antes de
# aceptar) y las cuatro decisiones.
class ReviewBar(QWidget):
    accepted = pyqtSignal()
    rejected = pyqtSignal()
    skipped = pyqtSignal()
    stepped = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.position = QLabel("")
        self.position.setObjectName("readout")
        self.position.setMinimumWidth(POSITION_WIDTH)
        self.species = QComboBox()
        self.species.setEditable(True)
        self.species.setFixedWidth(SPECIES_WIDTH)
        self.species.setToolTip("Species/call the box is saved with; type to correct it")
        self.species.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.button("◀", "Previous box (P)", lambda: self.stepped.emit(-1)))
        layout.addWidget(self.position)
        layout.addStretch(1)
        layout.addWidget(QLabel("as"))
        layout.addWidget(self.species)
        layout.addWidget(
            self.button("Accept", "Keep the box as framed and labelled (A)", self.accepted.emit)
        )
        layout.addWidget(self.button("Reject", "Drop the box (R)", self.rejected.emit))
        layout.addWidget(self.button("Skip", "Leave it for later (S)", self.skipped.emit))
        layout.addWidget(self.button("▶", "Next box (N)", lambda: self.stepped.emit(1)))

    def button(self, text: str, tip: str, action: Callable[[], None]) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda: action())
        return button

    def show_row(self, row: Row | None, index: int, total: int, labels: list[str]) -> None:
        self.species.blockSignals(True)
        self.species.clear()
        self.species.addItems(labels)
        if row is None:
            self.position.setText("Nothing left to review" if total == 0 else "")
        else:
            score = "" if pd.isna(row.score) else f" · {row.score:.2f}"
            self.position.setText(f"{index + 1} / {total}{score}")
            self.species.setCurrentText(row.label)
        self.species.blockSignals(False)
        self.species.setEnabled(row is not None)

    def label(self) -> str:
        return self.species.currentText().strip()


# Recorre las detecciones visibles en orden de tiempo. Aceptar mueve la caja a Annotations con
# el encuadre y la especie que tenga en ese momento; rechazar la quita; las dos quedan en el
# diario. La cola se recorta sola: lo decidido deja de estar en Detections.
class Reviewer(QObject):
    changed = pyqtSignal()

    def __init__(
        self, session: Session, plot: SpectrogramView, bar: ReviewBar, frame: Callable[[Row], None]
    ) -> None:
        super().__init__()
        self.session = session
        self.plot = plot
        self.bar = bar
        self.frame = frame
        self.active = False
        self.cursor = 0
        self.journal: Journal | None = None
        bar.accepted.connect(lambda: self.decide(ACCEPTED))
        bar.rejected.connect(lambda: self.decide(REJECTED))
        bar.skipped.connect(lambda: self.step(1))
        bar.stepped.connect(self.step)
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
        self.active = True
        self.cursor = 0
        self.show()

    def stop(self) -> None:
        self.active = False
        self.plot.set_editable(None)
        self.changed.emit()

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
        self.cursor = min(max(self.cursor, 0), max(len(rows) - 1, 0))
        row = rows[self.cursor] if rows else None
        labels = sorted(
            {r.label for source in SOURCES for r in self.session.rows(source) if r.label}
        )
        self.bar.show_row(row, self.cursor, len(rows), labels)
        if row is None:
            self.plot.set_editable(None)
        else:
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
