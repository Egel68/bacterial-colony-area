"""
Окно анализа изображения.
Отображает изображение и результаты анализа с расширенными настройками.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
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
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.colony_detector import ColonyDetector
from analysis.image_processor import ImageProcessor
from utils.calculations import AreaCalculator


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
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

        self.processor = ImageProcessor()
        self.detector = ColonyDetector()
        self.calculator = AreaCalculator()

        self.original_image = None
        self.display_image = None

        self.petri_mask = None
        self.colony_mask = None
        self.petri_info = None
        self.analysis_results = None
        self.debug_images = {}

        self._updating_ui = False

        self._load_image()
        self._init_ui()
        self._run_full_analysis()

    def _load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {self.image_path}")
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

    def _init_ui(self):
        self.setWindowTitle("Анализ бактериальных колоний")
        self.resize(1400, 950)  # Немного увеличим высоту

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        self._create_image_panel(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(450)
        scroll.setStyleSheet("QScrollArea {border: none; background-color: #1e1e2e;}")

        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(controls_widget)
        self.controls_layout.setSpacing(15)

        self._create_view_controls(self.controls_layout)
        self._create_geometry_controls(self.controls_layout)
        self._create_algorithm_controls(self.controls_layout)
        self._create_results_panel(self.controls_layout)
        self._create_action_buttons(self.controls_layout)

        self.controls_layout.addStretch()
        scroll.setWidget(controls_widget)
        main_layout.addWidget(scroll)

    def _create_image_panel(self, layout: QHBoxLayout):
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: #11111b; border-radius: 15px;")
        image_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        l = QVBoxLayout(image_frame)

        self.image_header = QLabel("Оригинальное изображение")
        self.image_header.setStyleSheet("color: #89b4fa; font-weight: bold;")
        self.image_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.image_header)

        self.image_label = ImageLabel()
        l.addWidget(self.image_label)

        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("color: #a6adc8;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.status_label)

        layout.addWidget(image_frame, stretch=2)

    def _create_view_controls(self, layout: QVBoxLayout):
        group = QGroupBox("👁️ Режим просмотра")
        l = QVBoxLayout(group)

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
        l.addWidget(self.view_mode_combo)

        self.show_petri_contour = QCheckBox("Показать контур чашки")
        self.show_petri_contour.setChecked(True)
        self.show_petri_contour.stateChanged.connect(self._update_display)
        l.addWidget(self.show_petri_contour)

        self.show_area_overlay = QCheckBox("Закрасить колонии")
        self.show_area_overlay.setChecked(True)
        self.show_area_overlay.stateChanged.connect(self._update_display)
        l.addWidget(self.show_area_overlay)

        layout.addWidget(group)

    def _create_geometry_controls(self, layout: QVBoxLayout):
        group = QGroupBox("📏 Геометрия чашки")
        l = QFormLayout(group)

        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 5000)
        self.spin_x.setSuffix(" px")
        self.spin_x.valueChanged.connect(self._on_geometry_changed)
        l.addRow("Центр X:", self.spin_x)

        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 5000)
        self.spin_y.setSuffix(" px")
        self.spin_y.valueChanged.connect(self._on_geometry_changed)
        l.addRow("Центр Y:", self.spin_y)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(10, 3000)
        self.spin_radius.setSuffix(" px")
        self.spin_radius.valueChanged.connect(self._on_geometry_changed)
        l.addRow("Радиус:", self.spin_radius)

        btn_reset = QPushButton("Сбросить к авто-поиску")
        btn_reset.setStyleSheet(
            "background-color: #45475a; font-size: 11px; padding: 5px;"
        )
        btn_reset.clicked.connect(self._reset_geometry)
        l.addRow(btn_reset)

        layout.addWidget(group)

    def _create_algorithm_controls(self, layout: QVBoxLayout):
        group = QGroupBox("⚙️ Параметры алгоритма")
        l = QVBoxLayout(group)

        # Чувствительность
        l.addWidget(QLabel("Чувствительность:"))
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
        l.addLayout(h_sens)

        # Контраст
        l.addWidget(QLabel("Усиление контраста:"))
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
        l.addLayout(h_cont)

        # Отступ
        l.addWidget(QLabel("Отступ от края (%):"))
        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0, 30)
        self.spin_margin.setValue(8.0)
        l.addWidget(self.spin_margin)

        # Мин размер
        l.addWidget(QLabel("Мин. размер колонии (px):"))
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 1000)
        self.spin_min_size.setValue(50)
        l.addWidget(self.spin_min_size)

        # --- Секция сплошных зон (НОВАЯ) ---
        l.addSpacing(10)
        fill_frame = QFrame()
        fill_frame.setStyleSheet(
            "background-color: #313244; border-radius: 6px; padding: 5px;"
        )
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

        l.addWidget(fill_frame)
        # ------------------------------------

        btn_apply = QPushButton("🔄 Пересчитать")
        btn_apply.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; margin-top: 10px;"
        )
        btn_apply.clicked.connect(self._run_colony_analysis_only)
        l.addWidget(btn_apply)

        layout.addWidget(group)

    def _create_results_panel(self, layout: QVBoxLayout):
        group = QGroupBox("📊 Результаты")
        l = QVBoxLayout(group)
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setMaximumHeight(150)
        l.addWidget(self.text_results)
        layout.addWidget(group)

    def _create_action_buttons(self, layout: QVBoxLayout):
        h = QHBoxLayout()
        btn_save = QPushButton("💾 Сохранить")
        btn_save.clicked.connect(self._save_result)
        btn_close = QPushButton("Закрыть")
        btn_close.setObjectName("secondary")
        btn_close.clicked.connect(self.close)
        h.addWidget(btn_save)
        h.addWidget(btn_close)
        layout.addLayout(h)

    # --- ЛОГИКА ---

    def _run_full_analysis(self):
        self.status_label.setText("Поиск чашки Петри...")
        self.petri_mask, self.petri_info = self.detector.detect_petri_dish(
            self.original_image
        )

        if self.petri_info:
            self._updating_ui = True
            self.spin_x.setValue(self.petri_info["center"][0])
            self.spin_y.setValue(self.petri_info["center"][1])
            self.spin_radius.setValue(self.petri_info["radius"])
            self._updating_ui = False
            self._run_colony_analysis_only()
        else:
            QMessageBox.warning(
                self, "Ошибка", "Чашка Петри не найдена. Настройте вручную."
            )
            h, w = self.original_image.shape[:2]
            self.petri_info = {
                "center": (w // 2, h // 2),
                "radius": int(min(w, h) * 0.4),
                "area_px": 0,
            }
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
        self.petri_info = {
            "center": (cx, cy),
            "radius": r,
            "area_px": int(np.pi * r**2),
        }
        h, w = self.original_image.shape[:2]
        self.petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.petri_mask, (cx, cy), r, 255, -1)
        self._update_display()

    def _reset_geometry(self):
        self._run_full_analysis()

    def _run_colony_analysis_only(self):
        if not self.petri_info:
            return

        self.status_label.setText("Анализ колоний...")

        sensitivity = self.slider_sens.value() / 100.0
        contrast = self.slider_contrast.value() / 10.0
        min_size = self.spin_min_size.value()
        margin = self.spin_margin.value()

        # Новые параметры
        use_solid_fill = self.chk_solid_fill.isChecked()
        fill_strength = self.spin_fill_strength.value()

        self.colony_mask, self.debug_images = self.detector.detect_colonies(
            self.original_image,
            self.petri_mask,
            petri_info=self.petri_info,
            sensitivity=sensitivity,
            min_colony_size=min_size,
            edge_margin_percent=margin,
            contrast_level=contrast,
            blur_size=5,
            # Передача новых параметров
            use_solid_fill=use_solid_fill,
            fill_strength=fill_strength,
        )

        self.analysis_results = self.calculator.calculate_areas(
            self.petri_mask, self.colony_mask, self.petri_info
        )

        res = self.analysis_results
        text = (
            f"Количество колоний: {res['colony_count']}\n"
            f"Покрытие чашки: {res['coverage_percent']:.2f}%\n"
            f"Площадь колоний: {res['colony_area_px']} px"
        )
        self.text_results.setText(text)
        self.status_label.setText(f"Готово. Найдено: {res['colony_count']}")

        self._update_display()

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
                    self.petri_info["center"],
                    self.petri_info["radius"],
                    (255, 0, 0),
                    2,
                )
            self.image_header.setText("Оригинальное изображение")

        elif mode == 2:  # Preprocessed
            if "preprocessed" in self.debug_images:
                gray = self.debug_images["preprocessed"]
                final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.image_header.setText("Усиленный контраст")
            else:
                final_img = self.original_image.copy()

        elif mode == 3:  # Binary
            if "binary" in self.debug_images:
                mask = self.debug_images["binary"]
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
                    self.petri_info["center"],
                    self.petri_info["radius"],
                    (100, 100, 255),
                    2,
                )
                margin = self.spin_margin.value()
                r_inner = int(self.petri_info["radius"] * (100 - margin) / 100)
                cv2.circle(
                    final_img, self.petri_info["center"], r_inner, (255, 255, 0), 1
                )

            if self.show_area_overlay.isChecked() and self.colony_mask is not None:
                overlay = final_img.copy()
                overlay[self.colony_mask > 0] = [0, 255, 0]
                cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0, final_img)
                cnts, _ = cv2.findContours(
                    self.colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
