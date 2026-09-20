import math
from typing import override

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QHeaderView,
    QLabel,
    QStackedWidget,
    QStyle,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from data.raven import SCORE
from viewer.controls import DockTitle, Panel
from viewer.session import ANNOTATIONS, SOURCES, Row, Session

# La banda en kHz con un decimal: en Hz son once caracteres y se come la etiqueta.
HEADERS = ("Start (s)", "End (s)", "kHz", "Label")
LABEL_COLUMN = 3
TRASH_WIDTH = 34
PANEL_WIDTH, PANEL_LEAST = 360, 260
ROW_HEIGHT = 24
ROOT = QModelIndex()
# Qué decir cuando una pestaña está vacía: lo que falta hacer, no el hecho de que falte.
EMPTY = {
    ANNOTATIONS: "No annotations yet.\nAccept boxes in Review, or open a Raven table.",
}
EMPTY_DEFAULT = "No detections yet.\nPress Detect, or lower the score."


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
        self.headers = (*HEADERS, SCORE, "") if scored else (*HEADERS, "")
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
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        # El encabezado se alinea como su columna: si no, el número queda debajo de nada.
        if role == Qt.ItemDataRole.TextAlignmentRole and section != LABEL_COLUMN:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    @override
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        row = self.rows[index.row()]
        column = index.column()
        if column == len(self.headers) - 1:
            if role == Qt.ItemDataRole.DecorationRole:
                return self.trash
            if role == Qt.ItemDataRole.ToolTipRole:
                return "Remove (Del)"
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole and column != LABEL_COLUMN:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ToolTipRole and column == LABEL_COLUMN:
            return row.label
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        values = [
            f"{row.begin:.2f}",
            f"{row.end:.2f}",
            f"{row.low / 1000:.1f}–{row.high / 1000:.1f}",
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
        # Por el medio: la etiqueta es `especie/llamada` y lo que distingue una fila de otra
        # está al final, así que cortar por la derecha las deja todas iguales.
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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


# Una tabla y, en su lugar cuando no hay filas, una frase que dice cómo se llenan.
class SourcePage(QStackedWidget):
    def __init__(self, session: Session, source: str) -> None:
        super().__init__()
        self.rows = 0
        self.table = SourceTable(session, source)
        empty = QLabel(EMPTY.get(source, EMPTY_DEFAULT))
        empty.setObjectName("placeholder")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setWordWrap(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.addWidget(empty)
        self.addWidget(page)
        self.addWidget(self.table)

    def refresh(self) -> int:
        self.rows = self.table.refresh()
        self.setCurrentIndex(1 if self.rows else 0)
        return self.rows

    def count(self) -> int:
        return self.rows


# Panel de cajas: una pestaña por origen, cada caja borrable con su papelera. Es el inspector
# de la ventana, y el único sitio donde las cajas se cuentan una por una.
class BoxTable(QDockWidget):
    picked = pyqtSignal(object)

    def __init__(self, session: Session) -> None:
        super().__init__("Boxes")
        self.session = session
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        # Un panel lateral, no una ventana flotante: sólo se cierra.
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.header = DockTitle(self, "Boxes")
        self.pages = {source: SourcePage(session, source) for source in SOURCES}
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for source, page in self.pages.items():
            page.table.picked.connect(self.picked)
            self.tabs.addTab(page, source)
        self.pinned = False  # el usuario eligió pestaña: no se le cambia por debajo
        self.tabs.currentChanged.connect(self.on_tab)
        tabbar = self.tabs.tabBar()
        if tabbar is not None:
            tabbar.tabBarClicked.connect(self.pin)
        body = Panel(PANEL_WIDTH, PANEL_LEAST)
        wrapper = QVBoxLayout(body)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(self.tabs)
        self.setWidget(body)

        session.changed.connect(self.refresh)
        self.refresh()

    def pin(self, _index: int) -> None:
        self.pinned = True

    # La pestaña lleva el conteo de lo que se ve; si el score esconde filas, lo dice entero.
    def refresh(self) -> None:
        for index, (source, page) in enumerate(self.pages.items()):
            table = self.session.tables[source]
            total = 0 if table is None else len(table)
            shown = page.refresh()
            label = source if not total else f"{source}  {shown}"
            if shown != total:
                label = f"{source}  {shown} of {total}"
            self.tabs.setTabText(index, label)
        self.show_filled()

    # Abrir el panel en una pestaña vacía teniendo la otra llena es un clic de más; mientras
    # nadie elija pestaña a mano, se muestra la que tiene cajas.
    def show_filled(self) -> None:
        pages = list(self.pages.values())
        if self.pinned or pages[self.tabs.currentIndex()].count():
            return
        filled = next((i for i, page in enumerate(pages) if page.count()), None)
        if filled is not None:
            self.tabs.setCurrentIndex(filled)

    def on_tab(self, index: int) -> None:
        list(self.pages.values())[index].table.emit_current()
