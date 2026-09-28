"""
Окно анализа изображения.
Отображает изображение и результаты анализа с расширенными настройками.
"""

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from analysis.params import AnalysisParams
from utils.image_loader import load_image
from PyQt6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.geometry import PetriInfo
from .controllers.analysis_controller import AnalysisController
from .responsive import ResponsiveMetrics, install_application_responsive_sizing


class _AlgorithmWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, controller, image, petri_mask, petri_info, algorithm, margin):
        super().__init__()
        self.controller = controller
        self.image = image
        self.petri_mask = petri_mask
        self.petri_info = petri_info
        self.algorithm = algorithm
        self.margin = margin

    def run(self):
        try:
            result = self.controller.calculate_algorithm_result(
                self.image,
                self.petri_mask,
                self.algorithm,
                petri_info=self.petri_info,
                margin_percent=self.margin,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._original_pixmap = None

    def set_image(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._original_pixmap:
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


class AnalysisWindow(QMainWindow):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path

        self.controller = AnalysisController()

        self.original_image = None
        self.display_image = None

        self.petri_mask = None
        self.petri_info: PetriInfo | None = None
        self.analysis_results = None
        self._analysis_thread = None
        self._analysis_worker = None
        self._analysis_busy = False

        self._updating_ui = False
        self.setMinimumSize(0, 0)

        self._load_image()
        self._init_ui()
        self._responsive_sizer = install_application_responsive_sizing(
            self, minimum_scale=0.75
        )
        self._run_full_analysis()

    def _load_image(self):
        self.original_image = load_image(self.image_path)
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

    def _init_ui(self):
        self.setWindowTitle("Анализ бактериальных колоний")
        self.resize(1400, 950)  # Немного увеличим высоту

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, central_widget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        self.controls_scroll_area = QScrollArea()
        scroll = self.controls_scroll_area
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(450)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustIgnored)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll.setStyleSheet("QScrollArea {border: none; background-color: #1e1e2e;}")
        self.controls_widget = QWidget()
        self.controls_widget.setMinimumSize(0, 0)
        self.controls_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setSpacing(15)

        self._create_view_controls(self.controls_layout)
        self._create_geometry_controls(self.controls_layout)
        self._create_algorithm_controls(self.controls_layout)
        self._create_results_panel(self.controls_layout)
        self._create_action_buttons(self.controls_layout)
        self.controls_layout.addStretch()

        controls_widget = self.controls_widget
        scroll.setWidget(controls_widget)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._create_image_panel(self.main_layout)
        self.main_layout.addWidget(scroll, stretch=1)
        self._controls_reflowed = False
        self._update_controls_width()
        self.statusBar().showMessage("Готов к работе")

    def _update_controls_width(self):
        if not hasattr(self, "controls_scroll_area"):
            return
        if getattr(self, "_controls_reflowed", False):
            self.controls_scroll_area.setMaximumWidth(16_777_215)
            return

        sizer = getattr(self, "_responsive_sizer", None)
        metrics = getattr(sizer, "metrics", None) or ResponsiveMetrics(1.0)
        available_width = self.contentsRect().width() or self.width()
        minimum_width = metrics.dimension(280, minimum=1)
        maximum_width = metrics.dimension(450, minimum=minimum_width)
        target_width = max(
            minimum_width,
            min(maximum_width, round(available_width * 0.34)),
        )
        self.controls_scroll_area.setMaximumWidth(target_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_controls_width()
        compact = self.width() < 920 or self.height() < 680
        if compact != self._controls_reflowed:
            self._controls_reflowed = compact
            self.main_layout.setDirection(
                QBoxLayout.Direction.TopToBottom
                if compact
                else QBoxLayout.Direction.LeftToRight
            )
            sizer = getattr(self, "_responsive_sizer", None)
            metrics = getattr(sizer, "metrics", None) or ResponsiveMetrics(1.0)
            self.controls_scroll_area.setMaximumHeight(
                metrics.dimension(350, minimum=1) if compact else 16_777_215
            )
            self._update_controls_width()

    def _create_image_panel(self, layout: QHBoxLayout):
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: #11111b; border-radius: 15px;")
        image_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        vl = QVBoxLayout(image_frame)

        self.image_header = QLabel("Оригинальное изображение")
        self.image_header.setStyleSheet("color: #89b4fa; font-weight: bold;")
        self.image_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.image_header)

        self.image_label = ImageLabel()
        vl.addWidget(self.image_label, stretch=1)

        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("color: #a6adc8;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        vl.addWidget(self.progress_bar)

        layout.addWidget(image_frame, stretch=2)

    def _create_view_controls(self, layout: QVBoxLayout):
        group = QGroupBox("👁️ Режим просмотра")
        vl = QVBoxLayout(group)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            [
                "🔍 Результат (С наложением)",
                "📷 Оригинал",
                "🌗 Предобработка (Контраст)",
                "🏁 Бинарная маска (Ч/Б)",
            ]
        )
        self.view_mode_combo.currentIndexChanged.connect(self._update_display)
        vl.addWidget(self.view_mode_combo)

        self.show_petri_contour = QCheckBox("Показать контур чашки")
        self.show_petri_contour.setChecked(True)
        self.show_petri_contour.stateChanged.connect(self._update_display)
        vl.addWidget(self.show_petri_contour)

        self.show_area_overlay = QCheckBox("Закрасить колонии")
        self.show_area_overlay.setChecked(True)
        self.show_area_overlay.stateChanged.connect(self._update_display)
        vl.addWidget(self.show_area_overlay)

        self.algorithm_combo = QComboBox()
        self._load_analysis_algorithms()
        vl.addWidget(QLabel("Алгоритм распознавания:"))
        vl.addWidget(self.algorithm_combo)
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)

        layout.addWidget(group)

    def _create_geometry_controls(self, layout: QVBoxLayout):
        group = QGroupBox("📏 Геометрия чашки")
        vl = QFormLayout(group)

        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 5000)
        self.spin_x.setSuffix(" px")
        self.spin_x.valueChanged.connect(self._on_geometry_changed)
        vl.addRow("Центр X:", self.spin_x)

        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 5000)
        self.spin_y.setSuffix(" px")
        self.spin_y.valueChanged.connect(self._on_geometry_changed)
        vl.addRow("Центр Y:", self.spin_y)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(10, 3000)
        self.spin_radius.setSuffix(" px")
        self.spin_radius.valueChanged.connect(self._on_geometry_changed)
        vl.addRow("Радиус:", self.spin_radius)

        self.btn_reset_geometry = QPushButton("Сбросить к авто-поиску")
        self.btn_reset_geometry.setStyleSheet("background-color: #45475a;")
        self.btn_reset_geometry.clicked.connect(self._reset_geometry)
        vl.addRow(self.btn_reset_geometry)

        layout.addWidget(group)

    def _create_algorithm_controls(self, layout: QVBoxLayout):
        group = QGroupBox("⚙️ Параметры алгоритма")
        vl = QVBoxLayout(group)

        # Чувствительность
        vl.addWidget(QLabel("Чувствительность:"))
        h_sens = QHBoxLayout()
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(1, 100)
        self.slider_sens.setValue(50)
        self.slider_sens.valueChanged.connect(
            lambda v: self.label_sens.setText(f"{v}%")
        )
        self.label_sens = QLabel("50%")
        h_sens.addWidget(self.slider_sens)
        h_sens.addWidget(self.label_sens)
        vl.addLayout(h_sens)

        # Контраст
        vl.addWidget(QLabel("Усиление контраста:"))
        h_cont = QHBoxLayout()
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(5, 30)
        self.slider_contrast.setValue(10)
        self.slider_contrast.valueChanged.connect(
            lambda v: self.label_contrast.setText(f"{v / 10:.1f}x")
        )
        self.label_contrast = QLabel("1.0x")
        h_cont.addWidget(self.slider_contrast)
        h_cont.addWidget(self.label_contrast)
        vl.addLayout(h_cont)

        # Отступ
        vl.addWidget(QLabel("Отступ от края (%):"))
        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0, 30)
        self.spin_margin.setValue(8.0)
        vl.addWidget(self.spin_margin)

        # Мин размер
        vl.addWidget(QLabel("Мин. размер колонии (px):"))
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 1000)
        self.spin_min_size.setValue(50)
        vl.addWidget(self.spin_min_size)

        # --- Секция сплошных зон (НОВАЯ) ---
        fill_frame = QFrame()
        fill_frame.setStyleSheet("background-color: #313244; border-radius: 6px;")
        fill_layout = QVBoxLayout(fill_frame)

        self.chk_solid_fill = QCheckBox("💧 Заполнять сплошные зоны")
        self.chk_solid_fill.setToolTip(
            "Включите для сплошных мазков бактерий. Выключите для отдельных мелких колоний."
        )
        fill_layout.addWidget(self.chk_solid_fill)

        fill_h = QHBoxLayout()
        fill_h.addWidget(QLabel("Сила заполнения:"))
        self.spin_fill_strength = QSpinBox()
        self.spin_fill_strength.setRange(1, 100)
        self.spin_fill_strength.setValue(15)
        self.spin_fill_strength.setToolTip(
            "Радиус объединения. Увеличьте, если остаются черные дыры внутри пятен."
        )
        fill_h.addWidget(self.spin_fill_strength)
        fill_layout.addLayout(fill_h)

        vl.addWidget(fill_frame)
        # ------------------------------------

        btn_apply = QPushButton("🔄 Пересчитать")
        btn_apply.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; margin-top: 10px;"
        )
        btn_apply.clicked.connect(self._run_colony_analysis_only)
        self.btn_apply = btn_apply
        vl.addWidget(btn_apply)

        layout.addWidget(group)

    def _create_results_panel(self, layout: QVBoxLayout):
        group = QGroupBox("📊 Результаты")
        vl = QVBoxLayout(group)
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setMinimumHeight(58)
        vl.addWidget(self.text_results)
        layout.addWidget(group)

    def _create_action_buttons(self, layout: QVBoxLayout):
        h = QHBoxLayout()
        self.btn_save = QPushButton("💾 Сохранить")
        self.btn_save.clicked.connect(self._save_result)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.setObjectName("secondary")
        self.btn_close.clicked.connect(self.close)
        h.addWidget(self.btn_save)
        h.addWidget(self.btn_close)
        layout.addLayout(h)

    # --- ЛОГИКА ---

    def _run_full_analysis(self):
        self.status_label.setText("Поиск чашки Петри...")
        self.petri_mask, self.petri_info = self.controller.find_petri_dish(
            self.original_image
        )

        if self.petri_info:
            self._updating_ui = True
            self.spin_x.setValue(self.petri_info.cx)
            self.spin_y.setValue(self.petri_info.cy)
            self.spin_radius.setValue(self.petri_info.radius)
            self._updating_ui = False
            self._run_colony_analysis_only()
        else:
            QMessageBox.warning(
                self, "Ошибка", "Чашка Петри не найдена. Настройте вручную."
            )
            h, w = self.original_image.shape[:2]
            self.petri_info = PetriInfo(
                cx=w // 2,
                cy=h // 2,
                radius=int(min(w, h) * 0.4),
                image_shape=(h, w),
            )
            self._updating_ui = True
            self.spin_x.setValue(w // 2)
            self.spin_y.setValue(h // 2)
            self.spin_radius.setValue(int(min(w, h) * 0.4))
            self._updating_ui = False

    def _on_geometry_changed(self):
        if self._updating_ui:
            return
        cx = self.spin_x.value()
        cy = self.spin_y.value()
        r = self.spin_radius.value()
        h, w = self.original_image.shape[:2]
        self.petri_info = PetriInfo(
            cx=cx,
            cy=cy,
            radius=r,
            image_shape=(h, w),
        )
        self.petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.petri_mask, (cx, cy), r, 255, -1)
        self._update_display()

    def _reset_geometry(self):
        self._run_full_analysis()

    def _run_colony_analysis_only(self):
        if not self.petri_info:
            return

        if self._analysis_busy:
            return

        algorithm = self._selected_algorithm()
        if algorithm is not None and algorithm.name.startswith("NN:"):
            self._run_algorithm_async(algorithm)
            return
        if algorithm is not None:
            self._run_algorithm_synchronously(algorithm)
            return

        self.status_label.setText("Анализ колоний...")
        self.statusBar().showMessage("Анализ колоний...")

        params = AnalysisParams(
            sensitivity=self.slider_sens.value() / 100.0,
            contrast=self.slider_contrast.value() / 10.0,
            margin_percent=self.spin_margin.value(),
            min_colony_size=self.spin_min_size.value(),
            solid_fill=self.chk_solid_fill.isChecked(),
            fill_strength=self.spin_fill_strength.value(),
        )

        self.analysis_results = self.controller.analyze(
            self.original_image,
            self.petri_mask,
            params=params,
            petri_info=self.petri_info,
            blur_size=5,
        )

        res = self.analysis_results
        text = (
            f"Количество колоний: {res.colony_count}\n"
            f"Покрытие (рабочей зоны): {res.coverage_percent:.2f}%\n"
            f"Площадь колоний: {res.colony_area_px} px"
        )
        self.text_results.setText(text)
        self.status_label.setText(f"Готово. Найдено: {res.colony_count}")
        self.statusBar().showMessage(f"Готово. Найдено: {res.colony_count}")

        self._update_display()

    def _load_analysis_algorithms(self):
        from testing.onnx_algorithm import register_bundled_models
        from testing.registry import list_algorithms

        register_bundled_models()
        self.algorithm_combo.addItem("Классический (параметры ниже)", "__classic__")
        for name in list_algorithms():
            self.algorithm_combo.addItem(name, name)

    def _selected_algorithm(self):
        name = self.algorithm_combo.currentData()
        if not name or name == "__classic__":
            return None
        from testing.registry import get_algorithm

        return get_algorithm(name)

    def _on_algorithm_changed(self, _index):
        name = self.algorithm_combo.currentData()
        uses_classic_controls = name == "__classic__"
        for widget in (
            self.slider_sens,
            self.slider_contrast,
            self.spin_min_size,
            self.chk_solid_fill,
            self.spin_fill_strength,
        ):
            widget.setEnabled(uses_classic_controls)

    def _run_algorithm_async(self, algorithm):
        self._analysis_busy = True
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.statusBar().showMessage(f"Запуск {algorithm.name}…")
        self.status_label.setText(f"Запуск {algorithm.name}…")
        self.btn_apply.setEnabled(False)
        self.algorithm_combo.setEnabled(False)
        for widget in (self.spin_x, self.spin_y, self.spin_radius, self.spin_margin):
            widget.setEnabled(False)

        self._analysis_thread = QThread(self)
        self._analysis_worker = _AlgorithmWorker(
            self.controller,
            self.original_image.copy(),
            self.petri_mask.copy(),
            self.petri_info,
            algorithm,
            self.spin_margin.value(),
        )
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.progress.connect(self._on_algorithm_progress)
        self._analysis_worker.finished.connect(self._on_algorithm_finished)
        self._analysis_worker.failed.connect(self._on_algorithm_failed)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.failed.connect(self._analysis_thread.quit)
        self._analysis_thread.finished.connect(self._cleanup_algorithm_worker)
        self._analysis_thread.start()

    def _on_algorithm_progress(self, completed, total):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(completed)
        message = f"Обработано тайлов {completed}/{total}"
        self.statusBar().showMessage(message)
        self.status_label.setText(message)

    def _on_algorithm_finished(self, result_and_mask):
        result, mask = result_and_mask
        self.controller.set_algorithm_mask(mask)
        self.analysis_results = result
        self.text_results.setText(
            f"Количество колоний: {result.colony_count}\n"
            f"Покрытие (рабочей зоны): {result.coverage_percent:.2f}%\n"
            f"Площадь колоний: {result.colony_area_px} px"
        )
        self.statusBar().showMessage(f"Готово. Найдено: {result.colony_count}")
        self.status_label.setText(f"Готово. Найдено: {result.colony_count}")
        self._update_display()

    def _on_algorithm_failed(self, message):
        self.statusBar().showMessage("Ошибка нейросетевого анализа")
        self.status_label.setText("Ошибка анализа")
        QMessageBox.critical(self, "Ошибка модели", message)

    def _cleanup_algorithm_worker(self):
        self._analysis_busy = False
        self.btn_apply.setEnabled(True)
        self.algorithm_combo.setEnabled(True)
        for widget in (self.spin_x, self.spin_y, self.spin_radius, self.spin_margin):
            widget.setEnabled(True)
        self.progress_bar.hide()
        if self._analysis_worker is not None:
            self._analysis_worker.deleteLater()
        self._analysis_worker = None
        self._analysis_thread = None

    def _run_algorithm_synchronously(self, algorithm):
        self.statusBar().showMessage(f"Анализ: {algorithm.name}…")
        try:
            result_and_mask = self.controller.analyze_with_algorithm(
                self.original_image,
                self.petri_mask,
                algorithm,
                petri_info=self.petri_info,
                margin_percent=self.spin_margin.value(),
            )
        except Exception as error:
            QMessageBox.critical(self, "Ошибка алгоритма", str(error))
            return
        self._on_algorithm_finished((result_and_mask, self.controller.colony_mask))

    def closeEvent(self, event):
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self.statusBar().showMessage(
                "Дождитесь завершения анализа перед закрытием окна"
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _update_display(self):
        if self.original_image is None:
            return

        mode = self.view_mode_combo.currentIndex()
        final_img = None

        if mode == 1:  # Original
            final_img = self.original_image.copy()
            if self.show_petri_contour.isChecked() and self.petri_info:
                cv2.circle(
                    final_img,
                    self.petri_info.center,
                    self.petri_info.radius,
                    (255, 0, 0),
                    2,
                )
            self.image_header.setText("Оригинальное изображение")

        elif mode == 2:  # Preprocessed
            dbg = self.controller.debug_images
            if "preprocessed" in dbg:
                gray = dbg["preprocessed"]
                final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.image_header.setText("Усиленный контраст")
            else:
                final_img = self.original_image.copy()

        elif mode == 3:  # Binary
            dbg = self.controller.debug_images
            if "binary" in dbg:
                mask = dbg["binary"]
                final_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                self.image_header.setText("Бинарная маска")
            else:
                final_img = np.zeros_like(self.original_image)

        else:  # Result
            final_img = self.original_image.copy()
            self.image_header.setText("Результат анализа")

            if self.show_petri_contour.isChecked() and self.petri_info:
                cv2.circle(
                    final_img,
                    self.petri_info.center,
                    self.petri_info.radius,
                    (100, 100, 255),
                    2,
                )
                margin = self.spin_margin.value()
                r_inner = int(self.petri_info.radius * (100 - margin) / 100)
                cv2.circle(final_img, self.petri_info.center, r_inner, (255, 255, 0), 1)

            colony_mask = self.controller.colony_mask
            if self.show_area_overlay.isChecked() and colony_mask is not None:
                overlay = final_img.copy()
                overlay[colony_mask > 0] = [0, 255, 0]
                cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0, final_img)
                cnts, _ = cv2.findContours(
                    colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(final_img, cnts, -1, (0, 255, 0), 1)

        h, w, ch = final_img.shape
        bytes_per_line = ch * w
        rgb_image = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        q_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        self.image_label.set_image(QPixmap.fromImage(q_image))

    def _save_result(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить", "result.png", "Images (*.png *.jpg)"
        )
        if file_path:
            self.image_label.pixmap().save(file_path)
