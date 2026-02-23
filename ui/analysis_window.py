"""
Окно анализа изображения.
Отображает изображение и результаты анализа.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
    """Виджет для отображения изображения с масштабированием."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self._original_pixmap = None

    def set_image(self, pixmap: QPixmap):
        """Установить изображение."""
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        """Обработка изменения размера."""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        """Обновить масштабированное изображение."""
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
        self.display_image = None
        self.petri_mask = None
        self.colony_mask = None
        self.petri_info = None
        self.inner_mask = None
        self.analysis_results = None

        # Флаги состояния
        self.show_colonies_overlay = False
        self.analysis_done = False

        self._load_image()
        self._init_ui()
        self._run_initial_analysis()

    def _load_image(self):
        """Загрузка изображения."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {self.image_path}")

        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

    def _init_ui(self):
        """Инициализация интерфейса."""
        self.setWindowTitle("Анализ бактериальных колоний")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        self._create_image_panel(main_layout)
        self._create_control_panel(main_layout)

    def _create_image_panel(self, layout: QHBoxLayout):
        """Создание панели с изображением."""
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border: 3px solid #45475a;
                border-radius: 15px;
            }
        """)
        image_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("📸 Изображение чашки Петри")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #89b4fa;
            background-color: transparent;
            border: none;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(title)

        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: #11111b; border: none;")
        image_layout.addWidget(self.image_label)

        self.status_label = QLabel("Статус: Ожидание анализа...")
        self.status_label.setStyleSheet("""
            font-size: 12px;
            color: #a6adc8;
            background-color: transparent;
            border: none;
            padding: 5px;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.status_label)

        layout.addWidget(image_frame, stretch=2)

    def _create_control_panel(self, layout: QHBoxLayout):
        """Создание панели управления."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(420)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e2e;
            }
        """)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(15)

        title = QLabel("⚙️ Панель управления")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title)

        self._create_detection_settings(control_layout)
        self._create_display_options(control_layout)
        self._create_action_buttons(control_layout)
        self._create_results_section(control_layout)

        control_layout.addStretch()

        close_button = QPushButton("✖️ Закрыть")
        close_button.setObjectName("secondary")
        close_button.clicked.connect(self.close)
        control_layout.addWidget(close_button)

        scroll_area.setWidget(control_widget)
        layout.addWidget(scroll_area)

    def _create_detection_settings(self, layout: QVBoxLayout):
        """Создание секции настроек обнаружения."""
        group = QGroupBox("🎯 Настройки обнаружения")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(10)

        # Чувствительность
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Чувствительность:")
        sensitivity_label.setMinimumWidth(120)
        sensitivity_layout.addWidget(sensitivity_label)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        sensitivity_layout.addWidget(self.sensitivity_slider)

        self.sensitivity_value = QLabel("50")
        self.sensitivity_value.setMinimumWidth(35)
        self.sensitivity_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        sensitivity_layout.addWidget(self.sensitivity_value)

        group_layout.addLayout(sensitivity_layout)

        # Минимальный размер колонии
        min_size_layout = QHBoxLayout()
        min_size_label = QLabel("Мин. размер (px):")
        min_size_label.setMinimumWidth(120)
        min_size_layout.addWidget(min_size_label)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 2000)
        self.min_size_spin.setValue(100)
        self.min_size_spin.setSingleStep(10)
        min_size_layout.addWidget(self.min_size_spin)

        group_layout.addLayout(min_size_layout)

        # Отступ от края
        margin_layout = QHBoxLayout()
        margin_label = QLabel("Отступ от края (%):")
        margin_label.setMinimumWidth(120)
        margin_layout.addWidget(margin_label)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 25)
        self.margin_spin.setValue(5)
        margin_layout.addWidget(self.margin_spin)

        group_layout.addLayout(margin_layout)

        # Кнопка переанализа
        self.reanalyze_button = QPushButton("🔄 Переанализировать")
        self.reanalyze_button.setStyleSheet("""
            QPushButton {
                background-color: #74c7ec;
                color: #1e1e2e;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #89dceb;
            }
        """)
        self.reanalyze_button.clicked.connect(self._run_analysis)
        group_layout.addWidget(self.reanalyze_button)

        layout.addWidget(group)

    def _create_display_options(self, layout: QVBoxLayout):
        """Создание опций отображения."""
        group = QGroupBox("🎨 Отображение")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        self.overlay_checkbox = QCheckBox("🟢 Подсветить колонии бактерий")
        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.stateChanged.connect(self._toggle_overlay)
        group_layout.addWidget(self.overlay_checkbox)

        self.petri_checkbox = QCheckBox("🔵 Показать контур чашки Петри")
        self.petri_checkbox.setChecked(True)
        self.petri_checkbox.stateChanged.connect(self._update_display)
        group_layout.addWidget(self.petri_checkbox)

        self.inner_checkbox = QCheckBox("🟠 Показать область анализа")
        self.inner_checkbox.setChecked(False)
        self.inner_checkbox.stateChanged.connect(self._update_display)
        group_layout.addWidget(self.inner_checkbox)

        layout.addWidget(group)

    def _create_action_buttons(self, layout: QVBoxLayout):
        """Создание кнопок действий."""
        group = QGroupBox("📊 Анализ")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(10)

        self.area_button = QPushButton("📏 Вывести площадь колонии")
        self.area_button.setObjectName("success")
        self.area_button.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        self.area_button.clicked.connect(self._show_area_results)
        group_layout.addWidget(self.area_button)

        self.save_button = QPushButton("💾 Сохранить результат")
        self.save_button.setObjectName("warning")
        self.save_button.clicked.connect(self._save_result)
        group_layout.addWidget(self.save_button)

        layout.addWidget(group)

    def _create_results_section(self, layout: QVBoxLayout):
        """Создание секции результатов."""
        group = QGroupBox("📈 Результаты анализа")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(5)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(250)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: #cdd6f4;
            }
        """)
        self.result_text.setPlaceholderText(
            "Нажмите 'Вывести площадь колонии'\nдля получения результатов анализа..."
        )
        group_layout.addWidget(self.result_text)

        layout.addWidget(group)

    def _run_initial_analysis(self):
        """Запуск начального анализа."""
        self._run_analysis()

    def _run_analysis(self):
        """Запуск анализа изображения."""
        try:
            self.status_label.setText("⏳ Анализ изображения...")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                color: #f9e2af;
                background-color: transparent;
                border: none;
                padding: 5px;
            """)

            # Получаем параметры
            sensitivity = self.sensitivity_slider.value() / 100.0
            min_size = self.min_size_spin.value()
            edge_margin = self.margin_spin.value()

            # Обнаружение чашки Петри
            self.petri_mask, self.petri_info = self.detector.detect_petri_dish(
                self.original_image
            )

            if self.petri_info is None:
                self.status_label.setText("❌ Чашка Петри не обнаружена!")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    color: #f38ba8;
                    background-color: transparent;
                    border: none;
                    padding: 5px;
                """)
                self._show_warning(
                    "Чашка Петри не обнаружена",
                    "Не удалось обнаружить чашку Петри на изображении.\n"
                    "Убедитесь, что изображение содержит чашку Петри на тёмном фоне.",
                )
                self._update_display()
                return

            # Создаём внутреннюю маску
            self.inner_mask = self.detector.create_inner_mask(
                self.petri_mask, self.petri_info, edge_margin
            )

            # Обнаружение колоний (теперь возвращает только маску)
            self.colony_mask = self.detector.detect_colonies(
                self.original_image,
                self.petri_mask,
                petri_info=self.petri_info,
                sensitivity=sensitivity,
                min_colony_size=min_size,
                edge_margin_percent=edge_margin,
            )

            # Вычисляем результаты
            self.analysis_results = self.calculator.calculate_areas(
                self.petri_mask, self.colony_mask, self.petri_info
            )

            self.analysis_done = True

            # Обновляем статус
            colony_count = self.analysis_results["colony_count"]
            coverage = self.analysis_results["coverage_percent"]

            self.status_label.setText(
                f"✅ Найдено колоний: {colony_count} | Покрытие: {coverage:.2f}%"
            )
            self.status_label.setStyleSheet("""
                font-size: 12px;
                color: #a6e3a1;
                background-color: transparent;
                border: none;
                padding: 5px;
            """)

            self._update_display()

        except Exception as e:
            self.status_label.setText(f"❌ Ошибка: {str(e)}")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                color: #f38ba8;
                background-color: transparent;
                border: none;
                padding: 5px;
            """)
            self._show_error(
                "Ошибка анализа", f"Произошла ошибка при анализе:\n{str(e)}"
            )

    def _find_contours(self, mask: np.ndarray):
        """Обёртка для cv2.findContours, совместимая с разными версиями OpenCV."""
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]
        return contours

    def _update_display(self):
        """Обновить отображение изображения."""
        display = self.display_image.copy()

        # Отображение контура чашки Петри
        if self.petri_checkbox.isChecked() and self.petri_info is not None:
            center = self.petri_info["center"]
            radius = self.petri_info["radius"]
            cv2.circle(display, center, radius, (137, 180, 250), 3)

        # Отображение внутренней границы
        if self.inner_checkbox.isChecked() and self.petri_info is not None:
            center = self.petri_info["center"]
            radius = self.petri_info["radius"]
            margin = self.margin_spin.value()
            inner_radius = int(radius * (100 - margin) / 100)
            cv2.circle(display, center, inner_radius, (250, 180, 137), 2)

        # Отображение колоний
        if self.overlay_checkbox.isChecked() and self.colony_mask is not None:
            if np.count_nonzero(self.colony_mask) > 0:
                overlay = display.copy()
                overlay[self.colony_mask > 0] = [166, 227, 161]

                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                contours = self._find_contours(self.colony_mask)
                cv2.drawContours(display, contours, -1, (100, 200, 100), 2)

        self._set_display_pixmap(display)

    def _set_display_pixmap(self, image: np.ndarray):
        """Установить изображение для отображения."""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(
            image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.set_image(pixmap)

    def _toggle_overlay(self, state: int):
        """Переключение отображения колоний."""
        self.show_colonies_overlay = state == Qt.CheckState.Checked.value
        self._update_display()

    def _on_sensitivity_changed(self, value: int):
        """Обработка изменения чувствительности."""
        self.sensitivity_value.setText(str(value))

    def _show_area_results(self):
        """Показать результаты анализа площади."""
        if not self.analysis_done or self.analysis_results is None:
            self._show_warning(
                "Анализ не выполнен",
                "Сначала выполните анализ изображения, нажав кнопку 'Переанализировать'.",
            )
            return

        results = self.analysis_results

        # Форматируем результаты
        text_lines = [
            "╔══════════════════════════════════════╗",
            "║       📊 РЕЗУЛЬТАТЫ АНАЛИЗА          ║",
            "╚══════════════════════════════════════╝",
            "",
            "┌─────────────────────────────────────┐",
            "│ 🔵 ЧАШКА ПЕТРИ                      │",
            "├─────────────────────────────────────┤",
            f"│ Площадь: {results['petri_area_px']:>15,} px    │",
            f"│ Площадь: {results['petri_area_mm2']:>15.2f} мм²   │",
            "└─────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────┐",
            "│ 🟢 КОЛОНИИ БАКТЕРИЙ                 │",
            "├─────────────────────────────────────┤",
            f"│ Площадь: {results['colony_area_px']:>15,} px    │",
            f"│ Площадь: {results['colony_area_mm2']:>15.2f} мм²   │",
            f"│ Количество: {results['colony_count']:>9} колоний   │",
            "└─────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────┐",
            "│ 📐 ОТНОСИТЕЛЬНАЯ ПЛОЩАДЬ            │",
            "├─────────────────────────────────────┤",
            f"│ Покрытие: {results['coverage_percent']:>14.2f} %     │",
            f"│ Доля: {results['area_ratio']:>18.4f}      │",
            "└─────────────────────────────────────┘",
        ]

        if results["colony_count"] > 0:
            text_lines.extend(
                [
                    "",
                    "┌─────────────────────────────────────┐",
                    "│ 📏 СРЕДНИЙ РАЗМЕР КОЛОНИИ           │",
                    "├─────────────────────────────────────┤",
                    f"│ Площадь: {results['avg_colony_area_px']:>12.1f} px      │",
                    f"│ Средняя площадь: {results['avg_colony_area_mm2']:>10.4f} мм²  │",
                    "└─────────────────────────────────────┘",
                ]
            )

        if results["colonies"] and len(results["colonies"]) > 0:
            text_lines.extend(
                [
                    "",
                    "┌─────────────────────────────────────┐",
                    "│ 🏆 ТОП-5 КРУПНЕЙШИХ КОЛОНИЙ         │",
                    "├─────────────────────────────────────┤",
                ]
            )

            for i, colony in enumerate(results["colonies"][:5]):
                text_lines.append(
                    f"│ {i + 1}. {colony['area_px']:>8} px ({colony['area_mm2']:.3f} мм²)   │"
                )

            text_lines.append("└─────────────────────────────────────┘")

        result_text = "\n".join(text_lines)

        self.result_text.setText(result_text)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #313244;
                border: 2px solid #a6e3a1;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: #cdd6f4;
            }
        """)

        self.overlay_checkbox.setChecked(True)

    def _save_result(self):
        """Сохранение результата анализа."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "analysis_result.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )

        if file_path:
            result = self.display_image.copy()

            if self.petri_info is not None:
                center = self.petri_info["center"]
                radius = self.petri_info["radius"]
                cv2.circle(result, center, radius, (137, 180, 250), 3)

            if self.colony_mask is not None and np.count_nonzero(self.colony_mask) > 0:
                overlay = result.copy()
                overlay[self.colony_mask > 0] = [166, 227, 161]
                result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
                contours = self._find_contours(self.colony_mask)
                cv2.drawContours(result, contours, -1, (100, 200, 100), 2)

            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, result_bgr)

            self._show_info("Сохранено", f"Результат сохранён:\n{file_path}")

    def _show_error(self, title: str, message: str):
        """Показать сообщение об ошибке."""
        QMessageBox.critical(self, title, message)

    def _show_warning(self, title: str, message: str):
        """Показать предупреждение."""
        QMessageBox.warning(self, title, message)

    def _show_info(self, title: str, message: str):
        """Показать информационное сообщение."""
        QMessageBox.information(self, title, message)
