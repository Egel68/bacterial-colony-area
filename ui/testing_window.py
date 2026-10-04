"""
Окно тестирования алгоритмов распознавания колоний.
Позволяет выбрать датасет, алгоритмы и модели, запустить прогон в фоне,
просмотреть сводные метрики, парное сравнение и экспортировать HTML-отчёт.
"""

from pathlib import Path
import threading

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from testing.baseline import BaselineDataset
from testing.dashboard import generate_report_atomic
from testing.onnx_algorithm import OnnxModelAlgorithm
from testing.registry import (
    list_algorithms,
    register_algorithm_instance,
)
from testing.pipeline_observer import PipelineObserver
from testing.runner import run_all_with_comparison
from testing.scheduler import PipelineDiagnostics
from testing.telemetry import TelemetryCollector
from ui.background import BackgroundOperation, snapshot
from .results_models import (
    ComparisonTableModel,
    ResultsTableView,
    SummaryTableModel,
)
from .responsive import install_application_responsive_sizing


class _DatasetView:
    """Представление выбранных GUI образцов без изменения общего загрузчика."""

    def __init__(self, root: Path, samples):
        self.root = root
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


def _limit_gui_dataset(dataset, sample_limit: int | None):
    if sample_limit is None:
        return dataset

    source_names = sorted(
        sample.name for sample in dataset.samples if sample.variant == "source"
    )[:sample_limit]
    selected_source_names = set(source_names)
    samples = [
        sample
        for sample in dataset.samples
        if (sample.variant == "source" and sample.name in selected_source_names)
        or (
            sample.variant == "cropped"
            and sample.name.removesuffix("_cropped") in selected_source_names
        )
    ]
    return _DatasetView(dataset.root, samples)


class _QtPipelineObserver(PipelineObserver):
    """Мост: события конвейера → Qt-сигналы окна (задача 7.1)."""

    def __init__(self, worker: "_RunWorker"):
        super().__init__()
        self._worker = worker

    def on_phase(self, phase: str, **context) -> None:
        self._worker.phase_changed.emit(phase)

    def on_task_error(self, context: str, error: str) -> None:
        self._worker.task_error.emit(context, error)

    def on_progress(self, completed: int, total: int) -> None:
        self._worker.progress_made.emit(completed, total)


class _RunWorker(QObject):
    """Выполняет прогон алгоритмов в фоновом потоке."""

    finished = pyqtSignal(dict, dict)
    failed = pyqtSignal(str)
    # Фазы конвейера и контекстные per-task ошибки (задача 7.1).
    phase_changed = pyqtSignal(str)
    task_error = pyqtSignal(str, str)
    progress_made = pyqtSignal(int, int)

    def __init__(
        self,
        data_root: str,
        algorithm_names: list[str],
        compare_pair=None,
        sample_limit: int | None = None,
        batch_size: int = 8,
        workers: int | None = None,
        telemetry: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.algorithm_names = algorithm_names
        self.compare_pair = compare_pair
        self.sample_limit = sample_limit
        self.batch_size = batch_size
        self.workers = workers
        self.telemetry_enabled = telemetry
        # Кооперативная отмена (задача 7.4): новые задачи не ставятся
        # после cancel() на ближайшей safe-границе конвейера.
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self):
        try:
            try:
                dataset = BaselineDataset(
                    root=self.data_root,
                )
            except RuntimeError as exc:
                if "No valid image-mask pairs found" not in str(exc):
                    raise
                self.failed.emit(
                    "No test samples found. Проверьте, что папка содержит "
                    "пары изображение + эталонная маска."
                )
                return
            except Exception as exc:  # noqa: BLE001 — ошибка структуры или манифеста
                self.failed.emit(
                    "Не удалось загрузить датасет или прочитать манифест "
                    f"dataset.json: {exc}"
                )
                return
            dataset = _limit_gui_dataset(dataset, self.sample_limit)
            if len(dataset) == 0:
                self.failed.emit(
                    "No test samples found. Проверьте, что папка содержит "
                    "пары изображение + эталонная маска."
                )
                return

            performance_output = (
                str(Path(self.data_root) / "performance.json")
                if self.telemetry_enabled
                else None
            )
            telemetry = TelemetryCollector(
                enabled=self.telemetry_enabled,
                output_path=performance_output,
            )
            diagnostics = PipelineDiagnostics()
            all_results, comparison = run_all_with_comparison(
                dataset,
                algorithms=self.algorithm_names,
                compare_pair=self.compare_pair,
                workers=self.workers,
                batch_size=self.batch_size,
                telemetry=telemetry,
                diagnostics=diagnostics,
                observer=_QtPipelineObserver(self),
                cancel_event=self._cancel_event,
            )
            if not any(
                sample_results
                for algorithm_results in all_results.values()
                for sample_results in algorithm_results.values()
            ):
                if diagnostics.loaded_pairs == 0:
                    details = "\n".join(diagnostics.load_failure_details)
                    suffix = f"\nПримеры проблем:\n{details}" if details else ""
                    self.failed.emit(
                        "Не удалось прочитать ни одной пары изображения и маски. "
                        f"Неуспешных пар: {diagnostics.load_failure_count}. "
                        f"Проверьте файлы и пути датасета.{suffix}"
                    )
                else:
                    details = "\n".join(diagnostics.algorithm_failure_details)
                    suffix = f"\nПримеры ошибок:\n{details}" if details else ""
                    self.failed.emit(
                        "Все запуски выбранных алгоритмов завершились с ошибкой "
                        f"для {diagnostics.loaded_pairs} прочитанных пар. "
                        f"Неуспешных задач: {diagnostics.algorithm_failure_count}. "
                        f"Проверьте модели и параметры алгоритмов.{suffix}"
                    )
                return

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
        self._export_op = None
        self._closing = False

        self._init_ui()
        self._responsive_sizer = install_application_responsive_sizing(
            self, minimum_scale=0.75
        )
        self._refresh_algorithms()

    def _init_ui(self):
        self.content_widget = QWidget()
        content = self.content_widget
        content.setMinimumWidth(0)
        root_layout = QVBoxLayout(content)
        root_layout.setSpacing(12)

        # --- Датасет ---
        dataset_group = QGroupBox("📁 Датасет")
        ds_form = QFormLayout(dataset_group)
        ds_row = QHBoxLayout()
        self.dataset_input = QLineEdit(self.data_root)
        ds_row.addWidget(self.dataset_input, 1)
        self.btn_browse_dataset = QPushButton("📂 Выбрать папку…")
        self.btn_browse_dataset.clicked.connect(self._choose_dataset)
        ds_row.addWidget(self.btn_browse_dataset)
        ds_form.addRow(ds_row)

        hint = QLabel(
            "Датасет — папка с парными изображениями и эталонными масками. "
            "Поддерживаются два формата:<br>"
            "<b>Legacy:</b> source/ (исходные фото), masks/ (маски с суффиксом "
            "_mask), cropped/ (обрезки), cropped_masks/ (маски обрезков с "
            "суффиксом _cropped_mask).<br>"
            "<b>Manifest:</b> корневая папка с dataset.json, содержащим "
            "относительные пути к изображениям и маскам; расширения файлов "
            "могут различаться. Для импорта 22022540 укажите "
            "datasets/22022540_imported.<br>"
            '<a href="https://github.com/Egel68/bacterial-colony-area/blob/main/docs/datasets.md">'
            "Каталог источников, версий и цитирования датасетов</a>."
        )
        hint.setObjectName("info")
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        ds_form.addRow(hint)
        root_layout.addWidget(dataset_group)

        # --- Алгоритмы ---
        alg_group = QGroupBox("🧠 Алгоритмы")
        alg_layout = QVBoxLayout(alg_group)
        self.alg_widget = QWidget()
        self.alg_widget.setMinimumWidth(0)
        self.alg_list_layout = QVBoxLayout(self.alg_widget)
        self.alg_list_layout.setContentsMargins(0, 0, 0, 0)
        self.alg_list_layout.addStretch()
        alg_layout.addWidget(self.alg_widget)

        self.btn_load_model = QPushButton("⬇️ Загрузить модель (.onnx)…")
        self.btn_load_model.clicked.connect(self._load_external_model)
        alg_layout.addWidget(self.btn_load_model)
        root_layout.addWidget(alg_group)

        # --- Параметры ---
        params_group = QGroupBox("⚙️ Параметры")
        params_layout = QVBoxLayout(params_group)
        self.chk_per_snapshot = QCheckBox("Детализация по снимкам")
        self.chk_per_snapshot.setChecked(True)
        params_layout.addWidget(self.chk_per_snapshot)

        tuning_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.tuning_row = tuning_row
        tuning_row.addWidget(QLabel("Batch:"))
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 512)
        self.batch_size_input.setValue(8)
        tuning_row.addWidget(self.batch_size_input)
        tuning_row.addWidget(QLabel("Объектов:"))
        self.sample_limit_input = QSpinBox()
        self.sample_limit_input.setRange(0, 1000000)
        self.sample_limit_input.setSpecialValueText("все")
        tuning_row.addWidget(self.sample_limit_input)
        self.chk_telemetry = QCheckBox("Telemetry")
        tuning_row.addWidget(self.chk_telemetry)
        tuning_row.addStretch()
        params_layout.addLayout(tuning_row)

        compare_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.compare_row = compare_row
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
        run_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.run_row = run_row
        self.btn_run = QPushButton("▶️ Запустить тест")
        self.btn_run.clicked.connect(self._start_run)
        run_row.addWidget(self.btn_run)
        self.btn_export = QPushButton("💾 Экспорт отчёта")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_report)
        run_row.addWidget(self.btn_export)
        self.btn_cancel = QPushButton("⏹ Отменить")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_run)
        run_row.addWidget(self.btn_cancel)
        run_layout.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        run_layout.addWidget(self.progress)
        self.status_label = QLabel("Готов к работе")
        run_layout.addWidget(self.status_label)
        root_layout.addWidget(run_group)

        # --- Результаты (model/view: ячейки не создают GUI-элементы) ---
        results_group = QGroupBox("📊 Результаты")
        results_layout = QVBoxLayout(results_group)
        self.table = ResultsTableView()
        self.table.setMinimumSize(0, 0)
        self.summary_model = SummaryTableModel(self)
        self.table.setModel(self.summary_model)
        self.table.setColumnWidth(0, 260)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        results_layout.addWidget(self.table)
        self.comparison_table = ResultsTableView()
        self.comparison_table.setMinimumSize(0, 0)
        self.comparison_model = ComparisonTableModel(self)
        self.comparison_table.setModel(self.comparison_model)
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.comparison_table.setVisible(False)
        results_layout.addWidget(self.comparison_table)
        root_layout.addWidget(results_group, stretch=1)

        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("testingContentScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setMinimumSize(0, 0)
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.content_scroll.setWidget(content)
        self.setCentralWidget(self.content_scroll)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = self.contentsRect().width() < 760
        direction = (
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        self.tuning_row.setDirection(direction)
        self.compare_row.setDirection(direction)
        self.run_row.setDirection(direction)

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
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Запуск прогона…")

        self._thread = QThread(self)
        sample_limit = self.sample_limit_input.value() or None
        self._worker = _RunWorker(
            data_root,
            names,
            compare_pair,
            sample_limit=sample_limit,
            batch_size=self.batch_size_input.value(),
            telemetry=self.chk_telemetry.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.phase_changed.connect(self._on_phase_changed)
        self._worker.task_error.connect(self._on_task_error)
        self._worker.progress_made.connect(self._on_progress_made)
        self._run_errors = []
        self._thread.finished.connect(lambda: self.progress.setValue(100))
        self._thread.start()

    def _cancel_run(self):
        """Кооперативная отмена прогона/экспорта (задача 7.4).

        Новые задачи прекращают ставиться на ближайшей safe-границе;
        уже выполняющийся native-вызов не прерывается и не блокирует
        GUI-поток.
        """
        if self._worker is not None:
            self._worker.cancel()
        if self._export_op is not None and self._export_op.is_running():
            self._export_op.cancel()
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("⏳ Отмена… Ожидание безопасной точки.")

    # Человекочитаемые названия фаз конвейера (задача 7.1).
    _PHASE_LABELS = {
        "dataset_scan": "Сканирование датасета…",
        "read_decode": "Чтение и декодирование пар…",
        "algorithm": "Вычисление алгоритмов…",
        "comparison": "Парное сравнение алгоритмов…",
        "result_preparation": "Подготовка результатов…",
    }

    def _on_phase_changed(self, phase: str):
        self.status_label.setText(self._PHASE_LABELS.get(phase, phase))

    def _on_task_error(self, context: str, error: str):
        # Ошибка отдельной задачи: успешные задачи остаются доступны.
        self._run_errors.append(f"{context}: {error}")
        self.status_label.setText(f"Ошибка задачи ({len(self._run_errors)}): {context}")

    def _on_progress_made(self, completed: int, total: int):
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(min(completed, total))

    def _on_finished(self, results, extra):
        if self._closing:
            # Поздние результаты не применяются после запроса закрытия (7.4).
            return
        self._results = results
        self._fill_table(extra["summary"])
        if extra.get("comparison"):
            self._fill_comparison(extra["comparison"])
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        error_count = len(getattr(self, "_run_errors", []))
        if error_count:
            self.status_label.setText(
                f"Готово. Прогон завершён (ошибок задач: {error_count})."
            )
        else:
            self.status_label.setText("Готово. Прогон завершён.")

    def _on_failed(self, message: str):
        if self._closing:
            return
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("Ошибка прогона.")
        QMessageBox.critical(self, "Ошибка прогона", message)

    def _fill_table(self, summary):
        # Model/view: данные уходят в модель, ячейки рисует делегат —
        # без создания GUI-элемента на каждую ячейку (задача 7.2).
        self.summary_model.set_summary(summary)

    def _fill_comparison(self, comparison):
        self.comparison_model.set_comparison(comparison)
        self.comparison_table.setVisible(self.comparison_model.rowCount() > 0)

    def _export_report(self):
        if self._results is None:
            return
        if self._export_op is not None and self._export_op.is_running():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "test_report.html", "HTML reports (*.html)"
        )
        if not path:
            return
        # Стабильный снимок результатов: генерация идёт вне GUI-потока (7.3),
        # публикация по финальному пути — только при успехе (атомарно).
        results_snapshot = snapshot(self._results)
        include_per_snapshot = self.chk_per_snapshot.isChecked()

        def export_work(ctx, results, output_path, per_snapshot):
            ctx.status("Генерация HTML-отчёта…")
            return generate_report_atomic(
                results,
                output_path,
                include_per_snapshot=per_snapshot,
                publish_check=ctx.checkpoint,
            )

        self._export_op = BackgroundOperation(export_work, parent=self)
        self._export_op.status_changed.connect(self.status_label.setText)
        self._export_op.result_ready.connect(self._on_export_finished)
        self._export_op.error_raised.connect(self._on_export_failed)
        self._export_op.start(
            results_snapshot,
            path,
            include_per_snapshot,
            copy_inputs=False,  # снимок уже скопирован выше
        )
        self.btn_export.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.status_label.setText("Генерация HTML-отчёта…")

    def _on_export_finished(self, published_path, generation):
        self.btn_export.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"Отчёт сохранён: {published_path}")

    def _on_export_failed(self, message, generation):
        self.btn_export.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.status_label.setText("Ошибка экспорта.")
        QMessageBox.critical(self, "Ошибка экспорта", message)

    def closeEvent(self, event):
        """Deferred close (задача 7.4): окно показывает «Завершение…» и
        закрывается после safe-точки worker без блокировки GUI-потока."""
        running = (self._thread is not None and self._thread.isRunning()) or (
            self._export_op is not None and self._export_op.is_running()
        )
        if not running:
            super().closeEvent(event)
            return
        event.ignore()
        if self._closing:
            return  # уже ждём безопасную точку
        self._closing = True
        self.status_label.setText(
            "⏳ Завершение… Окно закроется после завершения операций."
        )
        # Кооперативная отмена; native-вызовы доработают сами.
        if self._worker is not None:
            self._worker.cancel()
        if self._export_op is not None and self._export_op.is_running():
            self._export_op.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.finished.connect(self._maybe_close_after_ops)
        if self._export_op is not None:
            self._export_op.close_reached.connect(self._maybe_close_after_ops)
            self._export_op.request_close()
        if self._thread is None or not self._thread.isRunning():
            self._maybe_close_after_ops()

    def _maybe_close_after_ops(self):
        if not self._closing:
            return
        running = (self._thread is not None and self._thread.isRunning()) or (
            self._export_op is not None and self._export_op.is_running()
        )
        if running:
            return
        self._closing = False
        self.close()
