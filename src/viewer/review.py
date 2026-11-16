import hashlib
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import override

import pandas as pd
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from data.raven import BEGIN, CALL, END, HIGH, LOW, SCORE, SPECIES
from viewer.plot import SpectrogramView
from viewer.session import ANNOTATIONS, DETECTIONS, SOURCES, Row, Session

logger = logging.getLogger("viewer")

DECISION = "Decision"
ACCEPTED, REJECTED = "accepted", "rejected"
JOURNAL_COLUMNS = [BEGIN, END, LOW, HIGH, SPECIES, CALL, SCORE, DECISION]
JOURNAL_DIR = Path(tempfile.gettempdir()) / "primate-detector"
SPECIES_WIDTH = 220
# Mínimo para casar una decisión del diario con una detección
RESUME_IOU = 0.5

Box = tuple[float, float, float, float]


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
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row.to_csv(self.path, sep="\t", index=False, mode="a", header=not self.path.exists())
        except OSError:
            logger.warning("Could not write the review journal %s", self.path, exc_info=True)

    def read(self) -> pd.DataFrame:
        empty = pd.DataFrame(columns=JOURNAL_COLUMNS)
        if not self.path.is_file():
            return empty
        try:
            table = pd.read_csv(self.path, sep="\t", keep_default_na=False)
            if absent := set(JOURNAL_COLUMNS) - {SCORE} - set(table.columns):
                raise ValueError(f"missing columns {sorted(absent)}")
            for column in (BEGIN, END, LOW, HIGH):
                table[column] = pd.to_numeric(table[column], errors="raise")
        except (OSError, ValueError) as exc:
            aside = self.path.with_suffix(".unreadable.txt")
            logger.warning(
                "Review journal %s is unreadable (%s); moved to %s", self.path, exc, aside
            )
            try:
                self.path.replace(aside)
            except OSError:
                logger.warning("Could not move %s aside", self.path, exc_info=True)
            return empty
        return table

    def counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.read()[DECISION].value_counts().items()}


class ReviewBar(QWidget):
    accepted = pyqtSignal()
    rejected = pyqtSignal()
    done = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.position = QLabel("")
        self.score = QLabel("")
        self.species = QComboBox()
        self.species.setEditable(True)
        self.species.setMinimumWidth(SPECIES_WIDTH)
        self.species.setToolTip("Species/call the box is saved with; type to correct it")
        self.species.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        edit = self.species.lineEdit()
        if edit is not None:
            edit.installEventFilter(self)
        self.accept_button = self.button(
            "Accept", "Keep the box as framed and labelled (A or Enter)", self.accepted.emit
        )
        label = QLabel("Label")
        label.setToolTip(self.species.toolTip())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.position)
        layout.addWidget(self.score)
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addWidget(self.species)
        layout.addSpacing(8)
        layout.addWidget(self.button("Reject", "Drop the box (R or Delete)", self.rejected.emit))
        layout.addWidget(self.accept_button)
        layout.addSpacing(12)
        layout.addWidget(self.button("Done", "Leave the review (Esc)", self.done.emit))

    def button(self, text: str, tip: str, action: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda: action())
        return button

    # Enter se consume acá: la ventana también acepta con Enter
    @override
    def eventFilter(self, a0, a1) -> bool:
        if (
            isinstance(a1, QKeyEvent)
            and a1.type() == QEvent.Type.KeyPress
            and a1.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            self.accepted.emit()
            self.species.clearFocus()
            return True
        return super().eventFilter(a0, a1)

    def show_row(self, row: Row, index: int, total: int, labels: list[str]) -> None:
        self.position.setText(f"Detection {index + 1} of {total}")
        self.score.setText("" if pd.isna(row.score) else f"score {row.score:.2f}")
        self.species.blockSignals(True)
        self.species.clear()
        self.species.addItems(labels)
        self.species.setCurrentText(row.label)
        self.species.blockSignals(False)

    def label(self) -> str:
        return self.species.currentText().strip()


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
        # Antes de activarse: lo que resume agrega y quita no redibuja la revisión
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
            self.stop()
            return
        self.show()

    def show(self) -> None:
        rows = self.rows()
        if not rows:
            self.stop()
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
        # Quitarla dispara changed -> refresh -> show con la siguiente
        self.session.remove([(row.source, row.index)])

    def counts(self) -> dict[str, int]:
        return {} if self.journal is None else self.journal.counts()
