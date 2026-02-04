```python
"""
Окно анализа изображения.
Отображает изображение и результаты анализа.
"""

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QFrame, QScrollArea,
    QMessageBox, QGroupBox, QSlider, QSpinBox, QSplitter,
    QSizePolicy, QFileDialog
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap

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
                Qt.TransformationMode.SmoothTransformation
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
        self.processed_image = None
        self.display_image = None
        self.petri_mask = None
        self.colony_mask = None
        self.petri_info = None
        self.inner_mask = None
        
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
        
        # Конвертируем BGR -> RGB для отображения
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
    
    def _init_ui(self):
        """Инициализация интерфейса."""
        self.setWindowTitle("Анализ изображения")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Левая панель - изображение
        self._create_image_panel(main_layout)
        
        # Правая панель - управление
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
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(10, 10, 10, 10)
        
        # Заголовок
        title = QLabel("📸 Изображение чашки Петри")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(title)
        
        # Виджет изображения
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: #11111b; border: none;")
        image_layout.addWidget(self.image_label)
        
        layout.addWidget(image_frame, stretch=2)
    
    def _create_control_panel(self, layout: QHBoxLayout):
        """Создание панели управления."""
        control_frame = QFrame()
        control_frame.setFixedWidth(380)
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(15)
        
        # Заголовок
        title = QLabel("⚙️ Панель управления")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title)
        
        # Настройки обнаружения
        self._create_detection_settings(control_layout)
        
        # Опции отображения
        self._create_display_options(control_layout)
        
        # Кнопки действий
        self._create_action_buttons(control_layout)
        
        # Результаты
        self._create_results_section(control_layout)
        
        control_layout.addStretch()
        
        # Кнопка закрытия
        close_button = QPushButton("✖️ Закрыть")
        close_button.setObjectName("secondary")
        close_button.clicked.connect(self.close)
        control_layout.addWidget(close_button)
        
        layout.addWidget(control_frame)
    
    def _create_detection_settings(self, layout: QVBoxLayout):
        """Создание секции настроек обнаружения."""
        group = QGroupBox("🎯 Настройки обнаружения")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(10)
        
        # Чувствительность обнаружения колоний
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Чувствительность:")
        sensitivity_layout.addWidget(sensitivity_label)
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        
        self.sensitivity_value = QLabel("50")
        self.sensitivity_value.setMinimumWidth(30)
        sensitivity_layout.addWidget(self.sensitivity_value)
        
        group_layout.addLayout(sensitivity_layout)
        
        # Минимальный размер колонии
        min_size_layout = QHBoxLayout()
        min_size_label = QLabel("Мин. размер (px):")
        min_size_layout.addWidget(min_size_label)
        
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 1000)
        self.min_size_spin.setValue(100)
        self.min_size_spin.valueChanged.connect(self._on_settings_changed)
        min_size_layout.addWidget(self.min_size_spin)
        
        group_layout.addLayout(min_size_layout)
        
        # Отступ от края чашки
        margin_layout = QHBoxLayout()
        margin_label = QLabel("Отступ от края (%):")
        margin_layout.addWidget(margin_label)
        
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 20)
        self.margin_spin.setValue(5)
        self.margin_spin.valueChanged.connect(self._on_settings_changed)
        margin_layout.addWidget(self.margin_spin)
        
        group_layout.addLayout(margin_layout)
        
        # Кнопка переанализа
        self.reanalyze_button = QPushButton("🔄 Переанализировать")
        self.reanalyze_button.clicked.connect(self._run_analysis)
        group_layout.addWidget(self.reanalyze_button)
        
        layout.addWidget(group)
    
    def _create_display_options(self, layout: QVBoxLayout):
        """Создание опций отображения."""
        group = QGroupBox("🎨 Отображение")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(10)
        
        # Переключатель отображения колоний
        self.overlay_checkbox = QCheckBox("Подсветить колонии бактерий")
        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.stateChanged.connect(self._toggle_overlay)
        group_layout.addWidget(self.overlay_checkbox)
        
        # Переключатель отображения контура чашки
        self.petri_checkbox = QCheckBox("Показать контур чашки Петри")
        self.petri_checkbox.setChecked(True)
        self.petri_checkbox.stateChanged.connect(self._update_display)
        group_layout.addWidget(self.petri_checkbox)
        
        # Переключатель отображения внутренней границы
        self.inner_checkbox = QCheckBox("Показать область анализа")
        self.inner_checkbox.setChecked(False)
        self.inner_checkbox.stateChanged.connect(self._update_display)
        group_layout.addWidget(self.inner_checkbox)
        
        layout.addWidget(group)
    
    def _create_action_buttons(self, layout: QVBoxLayout):
        """Создание кнопок действий."""
        group = QGroupBox("📊 Анализ")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(10)
        
        # Кнопка вывода площади
        self.area_button = QPushButton("📏 Вывести площадь колонии")
        self.area_button.setObjectName("success")
        self.area_button.clicked.connect(self._show_area_results)
        group_layout.addWidget(self.area_button)
        
        # Кнопка сохранения
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
        
        self.result_label = QLabel("Нажмите 'Вывести площадь колонии'\nдля получения результатов")
        self.result_label.setObjectName("info")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(180)
        group_layout.addWidget(self.result_label)
        
        layout.addWidget(group)
    
    def _run_initial_analysis(self):
        """Запуск начального анализа."""
        self._update_display_image()
        self._run_analysis()
    
    def _run_analysis(self):
        """Запуск анализа изображения."""
        try:
            # Получаем параметры
            sensitivity = self.sensitivity_slider.value() / 100.0
            min_size = self.min_size_spin.value()
            edge_margin = self.margin_spin.value()
            
            # Обнаружение чашки Петри
            self.petri_mask, self.petri_info = self.detector.detect_petri_dish(
                self.original_image
            )
            
            if self.petri_info is None:
                self._show_warning(
                    "Чашка Петри не обнаружена",
                    "Не удалось обнаружить чашку Петри на изображении.\n"
                    "Убедитесь, что изображение содержит чашку Петри на тёмном фоне."
                )
                return
            
            # Создаём внутреннюю маску (для отображения)
            self.inner_mask = self.detector.create_inner_mask(
                self.petri_mask, self.petri_info, edge_margin
            )
            
            # Обнаружение колоний (с исключением краёв)
            self.colony_mask = self.detector.detect_colonies(
                self.original_image,
                self.petri_mask,
                petri_info=self.petri_info,
                sensitivity=sensitivity,
                min_colony_size=min_size,
                edge_margin_percent=edge_margin
            )
            
            self.analysis_done = True
            self._update_display()
            
        except Exception as e:
            self._show_error("Ошибка анализа", f"Произошла ошибка при анализе:\n{str(e)}")
    
    def _update_display_image(self):
        """Обновление отображаемого изображения."""
        self._update_display()
    
    def _update_display(self):
        """Обновить отображение изображения."""
        display = self.display_image.copy()
        
        # Отображение контура чашки Петри
        if self.petri_checkbox.isChecked() and self.petri_info is not None:
            center = self.petri_info['center']
            radius = self.petri_info['radius']
            cv2.circle(display, center, radius, (137, 180, 250), 3)  # Голубой контур
        
        # Отображение внутренней границы (область анализа)
        if self.inner_checkbox.isChecked() and self.petri_info is not None:
            center = self.petri_info['center']
            radius = self.petri_info['radius']
            margin = self.margin_spin.value()
            inner_radius = int(radius * (100 - margin) / 100)
            cv2.circle(display, center, inner_radius, (250, 180, 137), 2)  # Оранжевый контур
        
        # Отображение колоний
        if self.overlay_checkbox.isChecked() and self.colony_mask is not None:
            # Создаём цветное наложение для колоний
            overlay = display.copy()
            overlay[self.colony_mask > 0] = [166, 227, 161]  # Зелёный цвет
            
            # Смешиваем с оригиналом
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            
            # Добавляем контуры колоний
            contours, _ = cv2.findContours(
                self.colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display, contours, -1, (166, 227, 161), 2)
        
        self._set_display_pixmap(display)
    
    def _set_display_pixmap(self, image: np.ndarray):
        """Установить изображение для отображения."""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
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
    
    def _on_settings_changed(self):
        """Обработка изменения настроек."""
        pass  # Настройки применяются при нажатии "Переанализировать"
    
    def _show_area_results(self):
        """Показать результаты анализа площади."""
        if not self.analysis_done:
            self._show_warning(
                "Анализ не выполнен",
                "Сначала выполните анализ изображения."
            )
            return
        
        if self.petri_mask is None or self.colony_mask is None:
            self._show_warning(
                "Данные отсутствуют",
                "Не удалось получить данные для расчёта площади."
            )
            return
        
        # Вычисляем площади
        results = self.calculator.calculate_areas(
            self.petri_mask,
            self.colony_mask,
            self.petri_info
        )
        
        # Формируем текст результатов
        result_text = (
            f"📊 Результаты анализа:\n\n"
            f"🔵 Площадь чашки Петри:\n"
            f"   {results['petri_area_px']:,} пикселей\n"
            f"   ≈ {results['petri_area_mm2']:.2f} мм²\n\n"
            f"🟢 Площадь колоний:\n"
            f"   {results['colony_area_px']:,} пикселей\n"
            f"   ≈ {results['colony_area_mm2']:.2f} мм²\n\n"
            f"📐 Относительная площадь:\n"
            f"   {results['coverage_percent']:.2f}% от чашки\n"
            f"   (соотношение: 1:{1/results['area_ratio']:.1f})\n\n"
            f"🔢 Количество колоний:\n"
            f"   {results['colony_count']}\n\n"
            f"📏 Средний размер колонии:\n"
            f"   ≈ {results['avg_colony_area_mm2']:.3f} мм²"
        )
        
        self.result_label.setText(result_text)
        self.result_label.setStyleSheet("""
            background-color: #313244;
            border: 2px solid #a6e3a1;
            border-radius: 8px;
            padding: 10px;
            font-size: 11px;
        """)
        
        # Включаем отображение колоний для наглядности
        self.overlay_checkbox.setChecked(True)
    
    def _save_result(self):
        """Сохранение результата анализа."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "analysis_result.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        
        if file_path:
            # Создаём изображение с наложением
            result = self.display_image.copy()
            
            if self.petri_info is not None:
                center = self.petri_info['center']
                radius = self.petri_info['radius']
                cv2.circle(result, center, radius, (137, 180, 250), 3)
            
            if self.colony_mask is not None:
                overlay = result.copy()
                overlay[self.colony_mask > 0] = [166, 227, 161]
                result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
                contours, _ = cv2.findContours(
                    self.colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result, contours, -1, (166, 227, 161), 2)
            
            # Конвертируем RGB -> BGR для сохранения
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
