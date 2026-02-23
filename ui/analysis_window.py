"""
Окно анализа изображения.
Отображает изображение и результаты анализа с расширенными настройками.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QImage, QPixmap
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.colony_detector import ColonyDetector
from analysis.image_processor import ImageProcessor
from utils.calculations import AreaCalculator


class ImageLabel(QLabel):
    """Виджет для отображения изображения с масштабированием."""

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
    """Окно для анализа изображения бактериальных колоний."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path

        # Инициализация анализаторов
        self.processor = ImageProcessor()
        self.detector = ColonyDetector()
        self.calculator = AreaCalculator()

        # Данные изображения
        self.original_image = None
        self.display_image = None  # То, что сейчас на экране (RGB)

        # Результаты анализа
        self.petri_mask = None
        self.colony_mask = None
        self.petri_info = None  # {center: (x,y), radius: r}
        self.analysis_results = None
        self.debug_images = {}  # Словарь для хранения промежуточных этапов

        # Флаги для блокировки повторных вызовов при обновлении UI
        self._updating_ui = False

        self._load_image()
        self._init_ui()

        # Первый запуск - полный анализ
        self._run_full_analysis()

    def _load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {self.image_path}")
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

    def _init_ui(self):
        self.setWindowTitle("Анализ бактериальных колоний - Расширенный режим")
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Левая часть - Изображение
        self._create_image_panel(main_layout)

        # Правая часть - Управление (Scroll Area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(450)
        scroll.setStyleSheet("QScrollArea {border: none; background-color: #1e1e2e;}")

        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(controls_widget)
        self.controls_layout.setSpacing(15)

        # Секции управления
        self._create_view_controls(self.controls_layout)  # Режимы просмотра
        self._create_geometry_controls(self.controls_layout)  # Ручная коррекция круга
        self._create_algorithm_controls(self.controls_layout)  # Параметры детекции
        self._create_results_panel(self.controls_layout)  # Текст результатов
        self._create_action_buttons(self.controls_layout)  # Кнопки

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
        """Панель выбора режима просмотра."""
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

        # Чекбоксы оверлеев
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
        """Панель ручной коррекции геометрии чашки."""
        group = QGroupBox("📏 Геометрия чашки")
        l = QFormLayout(group)

        # X Center
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 5000)
        self.spin_x.setSuffix(" px")
        self.spin_x.valueChanged.connect(self._on_geometry_changed)
        l.addRow("Центр X:", self.spin_x)

        # Y Center
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 5000)
        self.spin_y.setSuffix(" px")
        self.spin_y.valueChanged.connect(self._on_geometry_changed)
        l.addRow("Центр Y:", self.spin_y)

        # Radius
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(10, 3000)
        self.spin_radius.setSuffix(" px")
        self.spin_radius.valueChanged.connect(self._on_geometry_changed)
        l.addRow("Радиус:", self.spin_radius)

        # Кнопка сброса к авто-детекции
        btn_reset = QPushButton("Сбросить к авто-поиску")
        btn_reset.setStyleSheet(
            "background-color: #45475a; font-size: 11px; padding: 5px;"
        )
        btn_reset.clicked.connect(self._reset_geometry)
        l.addRow(btn_reset)

        layout.addWidget(group)

    def _create_algorithm_controls(self, layout: QVBoxLayout):
        """Панель настроек алгоритма."""
        group = QGroupBox("⚙️ Параметры алгоритма")
        l = QVBoxLayout(group)

        # Чувствительность
        l.addWidget(QLabel("Чувствительность обнаружения:"))
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
        self.slider_contrast.setRange(5, 30)  # 0.5x to 3.0x
        self.slider_contrast.setValue(10)  # 1.0x
        self.slider_contrast.valueChanged.connect(
            lambda v: self.label_contrast.setText(f"{v / 10:.1f}x")
        )
        self.label_contrast = QLabel("1.0x")
        h_cont.addWidget(self.slider_contrast)
        h_cont.addWidget(self.label_contrast)
        l.addLayout(h_cont)

        # Мин размер
        l.addWidget(QLabel("Мин. размер колонии (px):"))
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 1000)
        self.spin_min_size.setValue(50)
        l.addWidget(self.spin_min_size)

        # Отступ
        l.addWidget(QLabel("Отступ от края (%):"))
        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0, 20)
        self.spin_margin.setValue(8.0)
        l.addWidget(self.spin_margin)

        # Кнопка ПРИМЕНИТЬ
        btn_apply = QPushButton("🔄 Пересчитать")
        btn_apply.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold;"
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
        """Полный цикл: поиск чашки -> поиск колоний."""
        self.status_label.setText("Поиск чашки Петри...")

        # 1. Ищем чашку
        self.petri_mask, self.petri_info = self.detector.detect_petri_dish(
            self.original_image
        )

        if self.petri_info:
            # Обновляем спинбоксы геометрии (без вызова сигнала changed)
            self._updating_ui = True
            self.spin_x.setValue(self.petri_info["center"][0])
            self.spin_y.setValue(self.petri_info["center"][1])
            self.spin_radius.setValue(self.petri_info["radius"])
            self._updating_ui = False

            # 2. Ищем колонии
            self._run_colony_analysis_only()
        else:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Чашка Петри не найдена. Попробуйте настроить параметры вручную.",
            )
            # Даже если не нашли, инициализируем дефолтные значения в центре
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
        """Вызывается когда юзер крутит спинбоксы координат."""
        if self._updating_ui:
            return

        # Обновляем инфо о чашке из спинбоксов
        cx = self.spin_x.value()
        cy = self.spin_y.value()
        r = self.spin_radius.value()

        self.petri_info = {
            "center": (cx, cy),
            "radius": r,
            "area_px": int(np.pi * r**2),
        }

        # Пересоздаем маску чашки
        h, w = self.original_image.shape[:2]
        self.petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.petri_mask, (cx, cy), r, 255, -1)

        # Перезапускаем анализ колоний (не полный, чашку искать не надо)
        # Делаем с небольшой задержкой или просто обновляем UI,
        # но для простоты здесь вызовем анализ
        # (В идеале здесь нужен QTimer, чтобы не лагало при прокрутке, но пока так)
        pass
        # Примечание: лучше пусть пользователь нажмет "Пересчитать",
        # иначе при прокрутке спинбокса будет лаг.
        # Однако контур чашки обновить стоит.
        self._update_display()

    def _reset_geometry(self):
        self._run_full_analysis()

    def _run_colony_analysis_only(self):
        """Запуск только этапа поиска колоний (используя текущую маску чашки)."""
        if not self.petri_info:
            return

        self.status_label.setText("Анализ колоний...")

        # Считываем параметры
        sensitivity = self.slider_sens.value() / 100.0
        contrast = self.slider_contrast.value() / 10.0
        min_size = self.spin_min_size.value()
        margin = self.spin_margin.value()

        # Запускаем детектор
        # ВНИМАНИЕ: detect_colonies теперь возвращает tuple (mask, debug_dict)
        self.colony_mask, self.debug_images = self.detector.detect_colonies(
            self.original_image,
            self.petri_mask,
            petri_info=self.petri_info,
            sensitivity=sensitivity,
            min_colony_size=min_size,
            edge_margin_percent=margin,
            contrast_level=contrast,
            blur_size=5,
        )

        # Считаем статистику
        self.analysis_results = self.calculator.calculate_areas(
            self.petri_mask, self.colony_mask, self.petri_info
        )

        # Выводим текст
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
        """Отрисовка в зависимости от выбранного режима."""
        if self.original_image is None:
            return

        mode = self.view_mode_combo.currentIndex()
        # 0: Result, 1: Original, 2: Preprocessed, 3: Binary

        final_img = None

        if mode == 1:  # Original
            final_img = self.original_image.copy()
            # Контуры рисуем только если попросили
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
                # Это grayscale изображение
                gray = self.debug_images["preprocessed"]
                final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.image_header.setText("Усиленный контраст + Вычитание фона")
            else:
                final_img = self.original_image.copy()

        elif mode == 3:  # Binary
            if "binary" in self.debug_images:
                # Маска
                mask = self.debug_images["binary"]
                final_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                self.image_header.setText("Бинарная маска (без фильтрации по размеру)")
            else:
                final_img = np.zeros_like(self.original_image)

        else:  # 0: Result (Overlay)
            final_img = self.original_image.copy()
            self.image_header.setText("Результат анализа")

            # Рисуем чашку
            if self.show_petri_contour.isChecked() and self.petri_info:
                # Внешний контур
                cv2.circle(
                    final_img,
                    self.petri_info["center"],
                    self.petri_info["radius"],
                    (100, 100, 255),
                    2,
                )
                # Внутренний (ROI)
                margin = self.spin_margin.value()
                r_inner = int(self.petri_info["radius"] * (100 - margin) / 100)
                cv2.circle(
                    final_img, self.petri_info["center"], r_inner, (255, 255, 0), 1
                )

            # Рисуем колонии
            if self.show_area_overlay.isChecked() and self.colony_mask is not None:
                # Зеленая заливка
                overlay = final_img.copy()
                overlay[self.colony_mask > 0] = [0, 255, 0]  # BGR
                cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0, final_img)

                # Контуры колоний
                cnts, _ = cv2.findContours(
                    self.colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(final_img, cnts, -1, (0, 255, 0), 1)

        # Конвертация в QPixmap
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
            pixmap = self.image_label.pixmap()
            if pixmap:
                pixmap.save(file_path)
                QMessageBox.information(self, "Успех", "Изображение сохранено")
