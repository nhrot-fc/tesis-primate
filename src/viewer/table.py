import math
from typing import override

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QHeaderView,
    QStyle,
    QTableView,
    QTabWidget,
)

from viewer.session import SOURCES, Row, Session

HEADERS = ("Inicio", "Fin", "Hz", "Etiqueta")
LABEL_COLUMN = 3
TRASH_WIDTH = 34
PANEL_WIDTH = 400
ROW_HEIGHT = 22
ROOT = QModelIndex()


class BoxModel(QAbstractTableModel):
    def __init__(self, session: Session, source: str) -> None:
        super().__init__()
        self.session = session
        self.source = source
        self.rows: list[Row] = []
        self.headers = (*HEADERS, "")
        style = QApplication.style()
        self.trash = style.standardIcon(QStyle.StandardPixmap.SP_TrashIcon) if style else QIcon()

    # La columna de score solo aparece si el origen la trae: las anotaciones no.
    def refresh(self) -> int:
        self.beginResetModel()
        self.rows = self.session.rows(self.source)
        scored = any(not math.isnan(row.score) for row in self.rows)
        self.headers = (*HEADERS, "Score", "") if scored else (*HEADERS, "")
        self.endResetModel()
        return len(self.rows)

    @override
    def rowCount(self, parent=ROOT) -> int:
        return 0 if parent.isValid() else len(self.rows)

    @override
    def columnCount(self, parent=ROOT) -> int:
        return 0 if parent.isValid() else len(self.headers)

    @override
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    @override
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        row = self.rows[index.row()]
        column = index.column()
        if column == len(self.headers) - 1:
            if role == Qt.ItemDataRole.DecorationRole:
                return self.trash
            if role == Qt.ItemDataRole.ToolTipRole:
                return "Borrar (Supr)"
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole and column != LABEL_COLUMN:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        values = [
            f"{row.begin:.2f}",
            f"{row.end:.2f}",
            f"{row.low:,.0f}–{row.high:,.0f}",
            row.label,
            f"{row.score:.2f}",
        ]
        return values[column]


class SourceTable(QTableView):
    picked = pyqtSignal(object)  # Row seleccionada, o None

    def __init__(self, session: Session, source: str) -> None:
        super().__init__()
        self.session = session
        self.boxes = BoxModel(session, source)
        self.setModel(self.boxes)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        vertical = self.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
            vertical.setDefaultSectionSize(ROW_HEIGHT)
        self.boxes.modelReset.connect(self.fit_columns)
        self.fit_columns()
        self.clicked.connect(self.on_click)

        selection = self.selectionModel()
        if selection is not None:
            selection.currentRowChanged.connect(self.on_current)

        remove = QShortcut(QKeySequence.StandardKey.Delete, self)
        remove.setContext(Qt.ShortcutContext.WidgetShortcut)
        remove.activated.connect(self.remove_selected)

    # Se rehace en cada reset porque la columna de score aparece y desaparece.
    def fit_columns(self) -> None:
        header = self.horizontalHeader()
        if header is None:
            return
        last = self.boxes.columnCount() - 1
        header.setStretchLastSection(False)
        for column in range(last + 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.resizeSection(last, TRASH_WIDTH)

    def refresh(self) -> int:
        return self.boxes.refresh()

    def on_click(self, index) -> None:
        if index.column() == self.boxes.columnCount() - 1:
            row = self.boxes.rows[index.row()]
            self.session.remove([(row.source, row.index)])

    def on_current(self, current, _previous) -> None:
        self.picked.emit(self.boxes.rows[current.row()] if current.isValid() else None)

    def emit_current(self) -> None:
        index = self.currentIndex()
        self.picked.emit(self.boxes.rows[index.row()] if index.isValid() else None)

    def remove_selected(self) -> None:
        selection = self.selectionModel()
        if selection is None:
            return
        rows = [self.boxes.rows[index.row()] for index in selection.selectedRows()]
        if rows:
            self.session.remove([(row.source, row.index) for row in rows])


# Ventana de revisión: una pestaña por origen, cada caja borrable con su papelera.
class BoxTable(QDockWidget):
    picked = pyqtSignal(object)

    def __init__(self, session: Session) -> None:
        super().__init__("Revisión")
        self.session = session
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.tables = {source: SourceTable(session, source) for source in SOURCES}
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(PANEL_WIDTH)
        self.tabs.setDocumentMode(True)
        for source, table in self.tables.items():
            table.picked.connect(self.picked)
            self.tabs.addTab(table, source)
        self.tabs.currentChanged.connect(self.on_tab)
        self.setWidget(self.tabs)

        session.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        for index, (source, table) in enumerate(self.tables.items()):
            total = self.session.tables[source]
            shown = table.refresh()
            self.tabs.setTabText(index, f"{source}  {shown}/{0 if total is None else len(total)}")

    def on_tab(self, index: int) -> None:
        list(self.tables.values())[index].emit_current()
