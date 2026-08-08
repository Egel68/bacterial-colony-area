"""
Точка входа приложения для анализа бактериальных колоний.
Запуск: python main.py
Сборка: pyinstaller --onefile --windowed --name BacteriaAnalyzer main.py
"""

import os
import sys

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.styles import get_application_style
from utils.logging import setup_logging


def main():
    """Главная функция запуска приложения."""
    setup_logging()

    # Включаем поддержку High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Bacteria Colony Analyzer")
    app.setApplicationVersion("1.1.3")

    # Устанавливаем шрифт приложения
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Применяем стили
    app.setStyleSheet(get_application_style())

    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
