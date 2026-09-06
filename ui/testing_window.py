"""
Окно тестирования алгоритмов распознавания колоний.
Позволяет выбрать датасет, алгоритмы и модели, запустить прогон в фоне,
просмотреть сводные метрики, парное сравнение и экспортировать HTML-отчёт.
"""

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from testing.dataset import TestDataset
from testing.dashboard import generate_report
from testing.onnx_algorithm import OnnxModelAlgorithm
from testing.registry import (
    list_algorithms,
    register_algorithm_instance,
)
from testing.runner import run_all, compare_algorithms


class _RunWorker(QObject):
    """Выполняет прогон алгоритмов в фоновом потоке."""

    finished = pyqtSignal(dict, dict)
    failed = pyqtSignal(str)

    def __init__(self, data_root: str, algorithm_names: list[str], compare_pair=None):
        super().__init__()
        self.data_root = data_root
        self.algorithm_names = algorithm_names
        self.compare_pair = compare_pair

    def run(self):
        try:
            dataset = TestDataset(root=self.data_root)
            if len(dataset) == 0:
                self.failed.emit(
                    "No test samples found. Пополните датасет парами изображение + маска."
                )
                return

            all_results = run_all(dataset, algorithms=self.algorithm_names)

            comparison = None
            if self.compare_pair:
                name_a, name_b = self.compare_pair
                comparison = compare_algorithms(dataset, name_a, name_b)

            summary = []
            for entry_name in all_results:
                entry = all_results[entry_name]
                means = {}
                for sample_key, variants in entry.items():
                    for variant_key, metrics in variants.items():
                        for metric_key in ("iou", "dice", "f1", "precision", "recall"):
                            means.setdefault(metric_key, []).append(
                                metrics.get(metric_key, 0.0)
                            )
                summary.append(
                    {
                        "name": entry_name,
                        "metrics": {
                            key: (sum(vals) / len(vals)) if vals else 0.0
                            for key, vals in means.items()
                        },
                    }
                )
            self.finished.emit(
                all_results, {"summary": summary, "comparison": comparison}
            )
        except Exception as e:  # noqa: BLE001 — показываем ошибку пользователю
            self.failed.emit(str(e))


class TestingWindow(QMainWindow):
    """Окно тестирования алгоритмов детекции колоний."""

    __test__ = False  # Не коллектировать pytest'ом как тестовый класс

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧪 Тестирование алгоритмов")
        self.resize(1100, 750)

        self.data_root = "test_images"
        self._results = None
        self._worker = None
        self._thread = None

        self._init_ui()
        self._refresh_algorithms()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(12)

        # --- Датасет ---
        dataset_group = QGroupBox("📁 Датасет")
        ds_form = QFormLayout(dataset_group)
        ds_row = QHBoxLayout()
        self.dataset_input = QLineEdit(self.data_root)
        ds_row.addWidget(self.dataset_input, 1)
        btn_browse = QPushButton("📂 Выбрать папку…")
        btn_browse.clicked.connect(self._choose_dataset)
        ds_row.addWidget(btn_browse)
        ds_form.addRow(ds_row)

        hint = QLabel(
            "Датасет — папка с парными изображениями и эталонными масками. "
            "Должна содержать четыре поддиректории:\n"
            "source/ (исходные фото чашек), masks/ (эталонные маски к ним, "
            "имя_файла_mask.png),\n"
            "cropped/ (обрезки чашек), cropped_masks/ (маски обрезков, "
            "имя_файла_cropped_mask.png)."
        )
        hint.setObjectName("info")
        hint.setWordWrap(True)
        ds_form.addRow(hint)
        root_layout.addWidget(dataset_group)

        # --- Алгоритмы ---
        alg_group = QGroupBox("🧠 Алгоритмы")
        alg_layout = QVBoxLayout(alg_group)
        self.alg_widget = QWidget()
        self.alg_list_layout = QVBoxLayout(self.alg_widget)
        self.alg_list_layout.setContentsMargins(0, 0, 0, 0)
        self.alg_list_layout.addStretch()
        alg_layout.addWidget(self.alg_widget)

        btn_load_model = QPushButton("⬇️ Загрузить модель (.onnx)…")
        btn_load_model.clicked.connect(self._load_external_model)
        alg_layout.addWidget(btn_load_model)
        root_layout.addWidget(alg_group)

        # --- Параметры ---
        params_group = QGroupBox("⚙️ Параметры")
        params_layout = QVBoxLayout(params_group)
        self.chk_per_snapshot = QCheckBox("Детализация по снимкам")
        self.chk_per_snapshot.setChecked(True)
        params_layout.addWidget(self.chk_per_snapshot)

        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Парное сравнение:"))
        self.cbx_compare_a = QComboBox()
        self.cbx_compare_b = QComboBox()
        compare_row.addWidget(self.cbx_compare_a)
        compare_row.addWidget(QLabel("vs"))
        compare_row.addWidget(self.cbx_compare_b)
        params_layout.addLayout(compare_row)
        root_layout.addWidget(params_group)

        # --- Запуск / прогресс ---
        run_group = QGroupBox("🚀 Запуск")
        run_layout = QVBoxLayout(run_group)
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶️ Запустить тест")
        self.btn_run.clicked.connect(self._start_run)
        run_row.addWidget(self.btn_run)
        self.btn_export = QPushButton("💾 Экспорт отчёта")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_report)
        run_row.addWidget(self.btn_export)
        run_layout.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        run_layout.addWidget(self.progress)
        self.status_label = QLabel("Готов к работе")
        run_layout.addWidget(self.status_label)
        root_layout.addWidget(run_group)

        # --- Результаты ---
        results_group = QGroupBox("📊 Результаты")
        results_layout = QVBoxLayout(results_group)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Алгоритм", "IoU", "Dice", "F1", "Precision", "Recall"]
        )
        self.table.setColumnWidth(0, 260)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        results_layout.addWidget(self.table)
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Снимок", "Variant", "Победитель"]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.comparison_table.setVisible(False)
        results_layout.addWidget(self.comparison_table)
        root_layout.addWidget(results_group, stretch=1)

    def _refresh_algorithms(self):
        """Обновляет чекбоксы алгоритмов и выпадающие списки парного сравнения."""
        while self.alg_list_layout.count() > 1:
            item = self.alg_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        names = list_algorithms()
        self._algo_checkboxes = {}
        for name in names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self._algo_checkboxes[name] = checkbox
            self.alg_list_layout.insertWidget(
                self.alg_list_layout.count() - 1, checkbox
            )

        self.cbx_compare_a.clear()
        self.cbx_compare_b.clear()
        self.cbx_compare_a.addItems(names)
        self.cbx_compare_b.addItems(names)
        if len(names) > 1:
            self.cbx_compare_b.setCurrentIndex(1)

    def _selected_algorithms(self) -> list[str]:
        return [name for name, cb in self._algo_checkboxes.items() if cb.isChecked()]

    def _choose_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "Выберите папку датасета", str(Path(self.data_root).parent)
        )
        if path:
            self.data_root = path
            self.dataset_input.setText(path)

    def _load_external_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите ONNX-модель", "", "ONNX models (*.onnx)"
        )
        if not path:
            return
        try:
            algo = OnnxModelAlgorithm(model_path=path)
            register_algorithm_instance(algo.name, algo)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки модели", str(e))
            return
        self._refresh_algorithms()
        self.status_label.setText(f"Модель загружена: {algo.name}")

    def _start_run(self):
        names = self._selected_algorithms()
        if not names:
            QMessageBox.warning(
                self, "Нет алгоритмов", "Выберите хотя бы один алгоритм."
            )
            return

        data_root = self.dataset_input.text().strip() or "test_images"
        if not Path(data_root).is_dir():
            QMessageBox.warning(
                self, "Датасет не найден", f"Папка датасета не существует:\n{data_root}"
            )
            return

        compare_pair = None
        if self.cbx_compare_a.currentText() and self.cbx_compare_b.currentText():
            compare_pair = (
                self.cbx_compare_a.currentText(),
                self.cbx_compare_b.currentText(),
            )

        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Запуск прогона…")

        self._thread = QThread(self)
        self._worker = _RunWorker(data_root, names, compare_pair)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.progress.setValue(100))
        self._thread.start()

    def _on_finished(self, results, extra):
        self._results = results
        self._fill_table(extra["summary"])
        if extra.get("comparison"):
            self._fill_comparison(extra["comparison"])
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.status_label.setText("Готово. Прогон завершён.")

    def _on_failed(self, message: str):
        self.btn_run.setEnabled(True)
        self.status_label.setText("Ошибка прогона.")
        QMessageBox.critical(self, "Ошибка прогона", message)

    def _fill_table(self, summary):
        self.table.setRowCount(len(summary))
        for row, entry in enumerate(summary):
            self.table.setItem(row, 0, QTableWidgetItem(entry["name"]))
            m = entry["metrics"]
            for col, key in enumerate(("iou", "dice", "f1", "precision", "recall")):
                self.table.setItem(
                    row, col + 1, QTableWidgetItem(f"{m.get(key, 0):.4f}")
                )

    def _fill_comparison(self, comparison):
        rows = []
        for metric_name, samples in sorted(comparison.items()):
            for sample_key, variants in sorted(samples.items()):
                for variant, winner in sorted(variants.items()):
                    rows.append((f"{metric_name} · {sample_key}", variant, winner))
        self.comparison_table.setRowCount(len(rows))
        for row, (sample, variant, winner) in enumerate(rows):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(sample))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(variant))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(winner))
        self.comparison_table.setVisible(len(rows) > 0)

    def _export_report(self):
        if self._results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "test_report.html", "HTML reports (*.html)"
        )
        if not path:
            return
        try:
            generate_report(
                self._results,
                output_path=path,
                include_per_snapshot=self.chk_per_snapshot.isChecked(),
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка экспорта", str(e))
            return
        self.status_label.setText(f"Отчёт сохранён: {path}")

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
