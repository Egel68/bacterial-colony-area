"""GUI-проверки выбора и асинхронного запуска алгоритма."""

import threading
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import QSize

from analysis.geometry import PetriInfo
from testing.interface import BaseDetectionAlgorithm
from testing.registry import register_algorithm_instance
from ui.analysis_window import AnalysisWindow
from ui.controllers.analysis_controller import AnalysisController

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


class _SlowTiledAlgorithm(BaseDetectionAlgorithm):
    name = "NN:fake_tiled"
    description = "Тестовая медленная модель"

    def __init__(self):
        self.thread_ids = []

    def detect(self, image, is_cropped=False):
        return self.detect_with_progress(image, is_cropped)

    def detect_with_progress(self, image, is_cropped=False, progress_callback=None):
        self.thread_ids.append(threading.get_ident())
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for index in range(2):
            time.sleep(0.04)
            if progress_callback:
                progress_callback(index + 1, 2)
        cv2.circle(mask, (40, 40), 15, 255, -1)
        return mask


def test_analysis_runs_selected_model_asynchronously_with_tile_progress(
    qtbot, tmp_path, monkeypatch
):
    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((100, 100, 3), 80, dtype=np.uint8))
    dish_info = PetriInfo(50, 50, 45, (100, 100))
    dish_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(dish_mask, dish_info.center, dish_info.radius, 255, -1)
    circle_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(circle_mask, dish_info.center, dish_info.radius, 255, -1)
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: (circle_mask.copy(), dish_info),
    )

    algorithm = _SlowTiledAlgorithm()
    register_algorithm_instance(algorithm.name, algorithm)
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.resize(800, 600)
    qtbot.waitUntil(lambda: window.size() == QSize(800, 600), timeout=3000)
    index = window.algorithm_combo.findData(algorithm.name)
    assert index >= 0
    window.algorithm_combo.setCurrentIndex(index)
    assert not window._analysis_busy  # Выбор алгоритма не запускает тяжёлый инференс.
    assert algorithm.thread_ids == []

    ticks = []
    timer = QTimer(window)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(10)
    resize_events = []

    def resize_during_analysis(width, height):
        window.resize(width, height)
        resize_events.append((width, height))

    QTimer.singleShot(10, lambda: resize_during_analysis(640, 480))
    QTimer.singleShot(35, lambda: resize_during_analysis(1280, 800))
    gui_thread_id = threading.get_ident()

    window._run_colony_analysis_only()
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)
    timer.stop()

    assert ticks
    assert resize_events == [(640, 480), (1280, 800)]
    assert window.size() == QSize(1280, 800)
    assert algorithm.thread_ids and algorithm.thread_ids[0] != gui_thread_id
    assert window.controller.colony_mask is not None
    assert window.analysis_results.colony_count == 1
    assert "Найдено: 1" in window.statusBar().currentMessage()
    assert window.progress_bar.isHidden()
    from testing.registry import _INSTANCES

    _INSTANCES.pop(algorithm.name, None)
    window.close()
