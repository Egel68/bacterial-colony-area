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
    # Начальная загрузка/поиск чашки/первый анализ асинхронны (задача 5.1).
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)
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


# --- Задачи 5.1 / 5.2: асинхронные загрузка и анализ, ошибки, отмена, delayed close ---


def _petri_stub(size=120):
    dish_info = PetriInfo(size // 2, size // 2, size // 2 - 10, (size, size))
    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle_mask, dish_info.center, dish_info.radius, 255, -1)
    return circle_mask, dish_info


def _patch_dialogs(monkeypatch, criticals=None):
    import ui.analysis_window as aw

    monkeypatch.setattr(
        aw.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        aw.QMessageBox,
        "critical",
        staticmethod(
            lambda *a, **k: criticals.append(a[2]) if criticals is not None else None
        ),
    )


def _make_window(qtbot, image_path, monkeypatch, dish_delay=0.0):
    circle_mask, dish_info = _petri_stub()

    def find_dish(self, image):
        if dish_delay:
            time.sleep(dish_delay)
        return circle_mask.copy(), dish_info

    monkeypatch.setattr(AnalysisController, "find_petri_dish", find_dish)
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_initial_load_and_detection_keep_gui_responsive(
    qtbot, tmp_path, monkeypatch
):
    """5.1: heartbeat и действия пользователя живут во время загрузки/поиска."""
    import ui.analysis_window as aw

    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))
    circle_mask, dish_info = _petri_stub()

    real_loader = aw.load_image_worker_safe

    def slow_loader(path):
        time.sleep(0.15)
        return real_loader(path)

    def slow_find(self, image):
        time.sleep(0.25)
        return circle_mask.copy(), dish_info

    monkeypatch.setattr(aw, "load_image_worker_safe", slow_loader)
    monkeypatch.setattr(AnalysisController, "find_petri_dish", slow_find)
    _patch_dialogs(monkeypatch)

    ticks = []
    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    observed = {}

    def user_action():
        observed["petri_during_action"] = window.petri_info
        window.slider_sens.setValue(77)
        observed["slider"] = window.slider_sens.value()

    QTimer.singleShot(30, user_action)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=10000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=10000)
    heartbeat.stop()

    assert observed.get("petri_during_action") is None, (
        "действие пользователя обработано только после завершения загрузки"
    )
    assert observed.get("slider") == 77
    assert len(ticks) >= 2, "heartbeat не работал во время загрузки/поиска"
    assert window.analysis_results is not None


def test_load_error_is_recoverable(qtbot, tmp_path, monkeypatch):
    """5.1: ошибка загрузки показывается и не роняет окно."""
    import ui.analysis_window as aw

    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))
    criticals = []
    _patch_dialogs(monkeypatch, criticals)

    def broken_loader(path):
        raise ValueError("битый файл")

    monkeypatch.setattr(aw, "load_image_worker_safe", broken_loader)
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(
        lambda: window._init_operation is not None
        and not window._init_operation.is_running(),
        timeout=5000,
    )
    assert any("битый файл" in message for message in criticals)
    assert "Ошибка" in window.status_label.text()
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_classic_analysis_runs_in_background(qtbot, tmp_path, monkeypatch):
    """5.2: классический пересчёт идёт в worker с phase/progress состоянием."""
    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))
    _patch_dialogs(monkeypatch)

    threads = []
    real_analyze = AnalysisController.analyze

    def slow_analyze(self, *args, **kwargs):
        threads.append(threading.get_ident())
        time.sleep(0.2)
        return real_analyze(self, *args, **kwargs)

    monkeypatch.setattr(AnalysisController, "analyze", slow_analyze)
    window = _make_window(qtbot, image_path, monkeypatch)

    gui_thread = threading.get_ident()
    qtbot.waitUntil(lambda: window._analysis_busy, timeout=5000)
    assert not window.btn_apply.isEnabled(), "кнопки не заблокированы при анализе"
    assert "Анализ колоний" in window.status_label.text()
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)

    assert threads and threads[0] != gui_thread, (
        "классический анализ выполнялся не в worker-потоке"
    )
    assert window.analysis_results is not None
    assert window.btn_apply.isEnabled()
    assert window.progress_bar.isHidden()


def test_classic_analysis_error_is_reported(qtbot, tmp_path, monkeypatch):
    """5.2: ошибка анализа показывается, UI восстанавливается."""
    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))
    criticals = []
    _patch_dialogs(monkeypatch, criticals)

    def broken_analyze(self, *args, **kwargs):
        raise RuntimeError("сбой анализа")

    monkeypatch.setattr(AnalysisController, "analyze", broken_analyze)
    window = _make_window(qtbot, image_path, monkeypatch)
    qtbot.waitUntil(
        lambda: window._analysis_operation is not None
        and not window._analysis_operation.is_running(),
        timeout=5000,
    )
    assert any("сбой анализа" in message for message in criticals)
    assert not window._analysis_busy
    assert window.btn_apply.isEnabled()


class _CancellableTiledAlgorithm(BaseDetectionAlgorithm):
    name = "NN:fake_cancellable"
    description = "Тестовая модель с множеством тайлов"

    def detect(self, image, is_cropped=False):
        return self.detect_with_progress(image, is_cropped)

    def detect_with_progress(self, image, is_cropped=False, progress_callback=None):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for index in range(30):
            time.sleep(0.03)
            if progress_callback:
                progress_callback(index + 1, 30)
        cv2.circle(mask, (60, 60), 15, 255, -1)
        return mask


def test_analysis_cancellation_stops_between_tiles(qtbot, tmp_path, monkeypatch):
    """5.2: кооперативная отмена останавливает инференс между тайлами."""
    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))
    _patch_dialogs(monkeypatch)

    algorithm = _CancellableTiledAlgorithm()
    register_algorithm_instance(algorithm.name, algorithm)
    window = _make_window(qtbot, image_path, monkeypatch)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)

    index = window.algorithm_combo.findData(algorithm.name)
    assert index >= 0
    window.algorithm_combo.setCurrentIndex(index)
    window._run_colony_analysis_only()
    qtbot.waitUntil(lambda: window._analysis_busy, timeout=5000)
    window._analysis_operation.cancel()
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)

    assert "отменён" in window.status_label.text()
    assert window.btn_apply.isEnabled()
    from testing.registry import _INSTANCES

    _INSTANCES.pop(algorithm.name, None)
    window.close()


def test_close_during_analysis_defers_and_closes(qtbot, tmp_path, monkeypatch):
    """5.2: delayed close — окно закрывается после safe-точки анализа."""
    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))
    _patch_dialogs(monkeypatch)

    real_analyze = AnalysisController.analyze

    def slow_analyze(self, *args, **kwargs):
        time.sleep(0.3)
        return real_analyze(self, *args, **kwargs)

    monkeypatch.setattr(AnalysisController, "analyze", slow_analyze)
    window = _make_window(qtbot, image_path, monkeypatch)

    ticks = []
    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    qtbot.waitUntil(lambda: window._analysis_busy, timeout=5000)
    window.close()
    assert window.isVisible(), "окно закрылось до safe-точки"
    assert "Завершение" in window.status_label.text()

    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
    heartbeat.stop()
    assert len(ticks) >= 2, "event loop блокировался при закрытии"
