"""
Точка входа приложения для анализа бактериальных колоний.
Запуск: python main.py
Сборка: pyinstaller --onefile --windowed --name BacteriaAnalyzer main.py
"""

import os
import sys
import json

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.styles import get_application_style
from utils.logging import setup_logging
from testing.onnx_algorithm import register_bundled_models


def main():
    """Главная функция запуска приложения."""
    setup_logging()
    register_bundled_models()

    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test-model":
        if len(sys.argv) < 3:
            raise SystemExit(
                "Usage: BacteriaAnalyzer --smoke-test-model <algorithm-name>"
            )
        _smoke_test_model(sys.argv[2])
        return

    # Включаем поддержку High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Bacteria Colony Analyzer")
    app.setApplicationVersion("0.2.0")

    # Устанавливаем шрифт приложения
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Применяем стили
    app.setStyleSheet(get_application_style())

    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


def _smoke_test_model(name: str) -> None:
    """Проверить bundled CPU-модель без создания GUI (для packaged build)."""
    import numpy as np

    from testing.registry import get_algorithm, list_algorithms

    if name not in list_algorithms():
        available = ", ".join(list_algorithms())
        raise SystemExit(f"Bundled model {name!r} not found. Available: {available}")
    algorithm = get_algorithm(name)
    image = np.zeros((97, 83, 3), dtype=np.uint8)
    mask = algorithm.detect(image)
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise RuntimeError(
            f"Invalid smoke output from {name}: shape={mask.shape}, dtype={mask.dtype}"
        )
    if not np.isin(mask, (0, 255)).all():
        raise RuntimeError(f"Invalid non-binary mask values from {name}")
    get_session = getattr(algorithm, "_get_session", None)
    if get_session is None:
        raise RuntimeError(f"Algorithm {name!r} does not expose an ONNX session")
    session = get_session()
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Expected CPU-only execution, got providers={providers}")
    print(
        json.dumps(
            {
                "model": name,
                "providers": providers,
                "input_shape": list(image.shape),
                "output_shape": list(mask.shape),
                "output_dtype": str(mask.dtype),
                "output_values": np.unique(mask).tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
