"""
Модуль стилей для графического интерфейса.
Содержит CSS-стили для PyQt виджетов.
"""


def get_application_style() -> str:
    """Возвращает основной стиль приложения."""
    return """
        /* Основное окно */
        QMainWindow, QDialog {
            background-color: #1e1e2e;
        }

        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
        }

        /* Заголовки */
        QLabel#title {
            font-size: 24px;
            font-weight: bold;
            color: #89b4fa;
            padding: 10px;
        }

        QLabel#subtitle {
            font-size: 14px;
            color: #a6adc8;
            padding: 5px;
        }

        /* Поле ввода пути */
        QLineEdit {
            background-color: #313244;
            border: 2px solid #45475a;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 13px;
            color: #cdd6f4;
            selection-background-color: #89b4fa;
        }

        QLineEdit:focus {
            border-color: #89b4fa;
        }

        QLineEdit:disabled {
            background-color: #11111b;
            color: #6c7086;
        }

        /* Основные кнопки */
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: bold;
            min-width: 120px;
        }

        QPushButton:hover {
            background-color: #b4befe;
        }

        QPushButton:pressed {
            background-color: #74c7ec;
        }

        QPushButton:disabled {
            background-color: #45475a;
            color: #6c7086;
        }

        /* Вторичные кнопки */
        QPushButton#secondary {
            background-color: #45475a;
            color: #cdd6f4;
        }

        QPushButton#secondary:hover {
            background-color: #585b70;
        }

        /* Кнопка успеха */
        QPushButton#success {
            background-color: #a6e3a1;
            color: #1e1e2e;
        }

        QPushButton#success:hover {
            background-color: #94e2d5;
        }

        /* Кнопка предупреждения */
        QPushButton#warning {
            background-color: #fab387;
            color: #1e1e2e;
        }

        QPushButton#warning:hover {
            background-color: #f9e2af;
        }

        /* Переключатель */
        QCheckBox {
            font-size: 14px;
            color: #cdd6f4;
            spacing: 10px;
        }

        QCheckBox::indicator {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 2px solid #45475a;
            background-color: #313244;
        }

        QCheckBox::indicator:checked {
            background-color: #89b4fa;
            border-color: #89b4fa;
        }

        QCheckBox::indicator:hover {
            border-color: #89b4fa;
        }

        /* Группа виджетов */
        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            color: #89b4fa;
            border: 2px solid #45475a;
            border-radius: 10px;
            margin-top: 15px;
            padding-top: 15px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px;
        }

        /* Область прокрутки */
        QScrollArea {
            border: none;
            background-color: transparent;
        }

        /* Полоса прокрутки */
        QScrollBar:vertical {
            background-color: #313244;
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }

        QScrollBar::handle:vertical {
            background-color: #45475a;
            border-radius: 6px;
            min-height: 30px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #585b70;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        QScrollBar:horizontal {
            background-color: #313244;
            height: 12px;
            border-radius: 6px;
            margin: 0;
        }

        QScrollBar::handle:horizontal {
            background-color: #45475a;
            border-radius: 6px;
            min-width: 30px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #585b70;
        }

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }

        /* Информационные метки */
        QLabel#info {
            background-color: #313244;
            border: 1px solid #45475a;
            border-radius: 8px;
            padding: 15px;
            font-size: 14px;
        }

        QLabel#result {
            background-color: #313244;
            border: 2px solid #a6e3a1;
            border-radius: 10px;
            padding: 20px;
            font-size: 16px;
            color: #a6e3a1;
        }

        /* Слайдер */
        QSlider::groove:horizontal {
            height: 8px;
            background-color: #313244;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background-color: #89b4fa;
            width: 20px;
            height: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }

        QSlider::handle:horizontal:hover {
            background-color: #b4befe;
        }

        QSlider::sub-page:horizontal {
            background-color: #89b4fa;
            border-radius: 4px;
        }

        /* Спинбокс */
        QSpinBox, QDoubleSpinBox {
            background-color: #313244;
            border: 2px solid #45475a;
            border-radius: 6px;
            padding: 8px;
            color: #cdd6f4;
        }

        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #89b4fa;
        }

        /* Сообщения */
        QMessageBox {
            background-color: #1e1e2e;
        }

        QMessageBox QLabel {
            color: #cdd6f4;
            font-size: 14px;
        }

        QMessageBox QPushButton {
            min-width: 80px;
            padding: 8px 16px;
        }

        /* Прогресс бар */
        QProgressBar {
            background-color: #313244;
            border: none;
            border-radius: 8px;
            height: 20px;
            text-align: center;
            color: #cdd6f4;
        }

        QProgressBar::chunk {
            background-color: #89b4fa;
            border-radius: 8px;
        }
    """


def get_image_frame_style() -> str:
    """Возвращает стиль для рамки изображения."""
    return """
        QFrame {
            background-color: #11111b;
            border: 3px solid #45475a;
            border-radius: 15px;
        }
    """


def get_result_panel_style() -> str:
    """Возвращает стиль для панели результатов."""
    return """
        QFrame {
            background-color: #313244;
            border: 2px solid #89b4fa;
            border-radius: 12px;
            padding: 10px;
        }
    """
