"""Model/view таблиц результатов тестирования (задача 7.2).

Заменяет `QTableWidget` + `QTableWidgetItem`-на-ячейку на `QTableView` +
`QAbstractTableModel`: ячейки рисуются делегатом только для видимой области,
данные хранятся в модели и не превращаются в GUI-элементы. Это позволяет
показывать сотни/тысячи строк без синхронного создания тысяч виджетов.

Модели поддерживают два способа наполнения:

- `set_rows()` / `set_summary()` / `set_comparison()` — полная замена одним
  вызовом (быстрый reset модели);
- `append_rows()` / `append_summary()` — инкрементальное добавление через
  `beginInsertRows`/`endInsertRows` — для потоковой подачи больших
  результатов без блокировки и мерцания.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QTableView

# Колонки сводной таблицы: имя алгоритма + метрики.
SUMMARY_HEADERS = ["Алгоритм", "IoU", "Dice", "F1", "Precision", "Recall"]
SUMMARY_METRICS = ("iou", "dice", "f1", "precision", "recall")

# Колонки таблицы парного сравнения.
COMPARISON_HEADERS = ["Снимок", "Variant", "Победитель"]


class _BaseResultsModel(QAbstractTableModel):
    """Базовая модель строк результатов (ленивое отображение ячеек)."""

    def __init__(self, headers: Sequence[str], parent=None):
        super().__init__(parent)
        self._headers = list(headers)
        self._rows: List[List[str]] = []

    def rowCount(self, parent=QModelIndex()):  # noqa: N802 — API Qt
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802 — API Qt
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        try:
            return self._rows[index.row()][index.column()]
        except (IndexError, TypeError):
            return None

    def headerData(  # noqa: N802 — API Qt
        self, section, orientation, role=Qt.ItemDataRole.DisplayRole
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if (
            orientation == Qt.Orientation.Horizontal
            and 0 <= section < len(self._headers)
        ):
            return self._headers[section]
        return None

    def set_rows(self, rows: List[List[str]]) -> None:
        """Полная замена строк одним вызовом (без виджетов на ячейку)."""
        self.beginResetModel()
        self._rows = [list(row) for row in rows]
        self.endResetModel()

    def append_rows(self, rows: List[List[str]]) -> None:
        """Инкрементальное добавление строк (beginInsertRows — без мерцания)."""
        if not rows:
            return
        first = len(self._rows)
        last = first + len(rows) - 1
        self.beginInsertRows(QModelIndex(), first, last)
        self._rows.extend(list(row) for row in rows)
        self.endInsertRows()

    def clear(self) -> None:
        self.set_rows([])


def _summary_rows(summary: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for entry in summary:
        metrics = entry.get("metrics", {})
        row = [entry.get("name", "")]
        for key in SUMMARY_METRICS:
            row.append(f"{metrics.get(key, 0):.4f}")
        rows.append(row)
    return rows


class SummaryTableModel(_BaseResultsModel):
    """Сводка метрик по алгоритмам (по одной строке на алгоритм)."""

    def __init__(self, parent=None):
        super().__init__(SUMMARY_HEADERS, parent)

    def set_summary(self, summary: List[Dict[str, Any]]) -> None:
        self.set_rows(_summary_rows(summary))

    def append_summary(self, summary: List[Dict[str, Any]]) -> None:
        self.append_rows(_summary_rows(summary))


class ComparisonTableModel(_BaseResultsModel):
    """Результаты парного сравнения алгоритмов."""

    def __init__(self, parent=None):
        super().__init__(COMPARISON_HEADERS, parent)

    @staticmethod
    def flatten(comparison: Dict[str, Any]) -> List[List[str]]:
        rows: List[List[str]] = []
        for metric_name, samples in sorted(comparison.items()):
            for sample_key, variants in sorted(samples.items()):
                for variant, winner in sorted(variants.items()):
                    rows.append([f"{metric_name} · {sample_key}", variant, winner])
        return rows

    def set_comparison(self, comparison: Dict[str, Any]) -> None:
        self.set_rows(self.flatten(comparison))


class ResultsTableView(QTableView):
    """`QTableView` для результатов с совместимым `rowCount()`.

    Ячейки рисуются делегатом только для видимой области — большие таблицы
    не создают GUI-элемент на каждую ячейку.
    """

    def rowCount(self) -> int:  # noqa: N802 — совместимость с QTableWidget API
        model = self.model()
        return model.rowCount() if model is not None else 0
