"""GUI-проверки фиксированной модели и асинхронного анализа."""

import threading
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QTimer

from analysis.geometry import PetriInfo
from testing.interface import BaseDetectionAlgorithm
from testing.registry import register_algorithm_instance
from ui.analysis_window import AnalysisWindow
from ui.controllers.analysis_controller import AnalysisController

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


class _SlowTiledAlgorithm(BaseDetectionAlgorithm):
    name = "NN:colony_seg"
    description = "Тестовая медленная bundled-модель"

    def __init__(self, delay=0.04, tile_count=2):
        self.thread_ids = []
        self.delay = delay
        self.tile_count = tile_count

    def detect(self, image, is_cropped=False):
        return self.detect_with_progress(image, is_cropped)

    def detect_with_progress(self, image, is_cropped=False, progress_callback=None):
        self.thread_ids.append(threading.get_ident())
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for index in range(self.tile_count):
            time.sleep(self.delay)
            if progress_callback:
                progress_callback(index + 1, self.tile_count)
        cv2.circle(mask, (image.shape[1] // 2, image.shape[0] // 2), 15, 255, -1)
        return mask


def _petri_stub(size=120):
    dish_info = PetriInfo(size // 2, size // 2, size // 2 - 10, (size, size))
    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle_mask, dish_info.center, dish_info.radius, 255, -1)
    return circle_mask, dish_info


def _patch_dialogs(monkeypatch, criticals=None, warnings=None):
    import ui.analysis_window as aw

    monkeypatch.setattr(
        aw.QMessageBox,
        "warning",
        staticmethod(
            lambda *a, **k: warnings.append(a[2]) if warnings is not None else None
        ),
    )
    monkeypatch.setattr(
        aw.QMessageBox,
        "critical",
        staticmethod(
            lambda *a, **k: criticals.append(a[2]) if criticals is not None else None
        ),
    )


def _install_model(monkeypatch, algorithm=None):
    algorithm = algorithm or _SlowTiledAlgorithm()
    monkeypatch.setattr(
        "testing.onnx_algorithm.register_bundled_models", lambda: [algorithm]
    )
    register_algorithm_instance("NN:colony_seg", algorithm)
    return algorithm


def _make_window(qtbot, image_path, monkeypatch, dish=("ok", 0.0)):
    circle_mask, dish_info = _petri_stub()

    def find_dish(self, image):
        if dish[1]:
            time.sleep(dish[1])
        if dish[0] == "missing":
            return None, None
        return circle_mask.copy(), dish_info

    monkeypatch.setattr(AnalysisController, "find_petri_dish", find_dish)
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def _write_image(tmp_path, name="dish.png", size=120):
    image_path = tmp_path / name
    assert cv2.imwrite(str(image_path), np.full((size, size, 3), 80, dtype=np.uint8))
    return image_path


def test_analysis_always_uses_approved_bundled_model_without_tuning_controls(
    qtbot, tmp_path, monkeypatch
):
    image_path = _write_image(tmp_path)
    algorithm = _install_model(monkeypatch)
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: _petri_stub(image.shape[0]),
    )
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.resize(800, 600)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)

    assert not hasattr(window, "algorithm_combo")
    assert not hasattr(window, "slider_sens")
    assert not hasattr(window, "slider_contrast")
    assert not hasattr(window, "spin_margin")
    assert not hasattr(window, "spin_min_size")
    assert not hasattr(window, "chk_solid_fill")
    assert not hasattr(window, "spin_fill_strength")
    assert not hasattr(window, "spin_x")
    assert not hasattr(window, "spin_y")
    assert not hasattr(window, "spin_radius")
    assert algorithm.thread_ids
    assert window.analysis_results is not None
    assert window.analysis_results.colony_count == 1

    window.close()
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)


def test_analysis_fails_closed_when_approved_bundled_model_is_missing(
    qtbot, tmp_path, monkeypatch
):
    image_path = _write_image(tmp_path)
    criticals = []
    _patch_dialogs(monkeypatch, criticals=criticals)
    monkeypatch.setattr("testing.onnx_algorithm.register_bundled_models", lambda: [])
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: _petri_stub(image.shape[0]),
    )
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(
        lambda: (
            window._analysis_operation is not None
            and not window._analysis_operation.is_running()
        ),
        timeout=5000,
    )

    assert any("NN:colony_seg" in message for message in criticals)
    assert window.analysis_results is None
    assert not window._analysis_busy
    assert "Ошибка нейросетевого анализа" in window.statusBar().currentMessage()
    window.close()


def test_analysis_errors_do_not_fallback_to_classic(qtbot, tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    criticals = []
    _patch_dialogs(monkeypatch, criticals=criticals)
    _install_model(monkeypatch)
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: _petri_stub(image.shape[0]),
    )
    classic_calls = []
    monkeypatch.setattr(
        AnalysisController,
        "analyze",
        lambda *args, **kwargs: classic_calls.append(True),
    )
    monkeypatch.setattr(
        AnalysisController,
        "calculate_algorithm_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("onnx сбой")),
    )
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(
        lambda: (
            window._analysis_operation is not None
            and not window._analysis_operation.is_running()
        ),
        timeout=5000,
    )

    assert any("onnx сбой" in message for message in criticals)
    assert classic_calls == []
    assert window.analysis_results is None
    assert not window._analysis_busy
    window.close()
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)


def test_dish_autodetection_failure_does_not_create_fallback_or_manual_geometry(
    qtbot, tmp_path, monkeypatch
):
    image_path = _write_image(tmp_path)
    algorithm = _install_model(monkeypatch)
    warnings = []
    _patch_dialogs(monkeypatch, warnings=warnings)
    monkeypatch.setattr(
        AnalysisController, "find_petri_dish", lambda self, image: (None, None)
    )
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(
        lambda: (
            window._init_operation is not None
            and not window._init_operation.is_running()
        ),
        timeout=5000,
    )

    assert any("автоматически определить" in message for message in warnings)
    assert window.petri_info is None
    assert window.petri_mask is None
    assert window.analysis_results is None
    assert not algorithm.thread_ids
    assert not hasattr(window, "spin_x")
    assert not hasattr(window, "spin_y")
    assert not hasattr(window, "spin_radius")
    window.close()


def test_initial_load_and_detection_keep_gui_responsive(qtbot, tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    algorithm = _install_model(monkeypatch)
    circle_mask, dish_info = _petri_stub()
    real_loader = __import__(
        "ui.analysis_window", fromlist=["load_image_worker_safe"]
    ).load_image_worker_safe

    def slow_loader(path):
        time.sleep(0.15)
        return real_loader(path)

    def slow_find(self, image):
        time.sleep(0.25)
        return circle_mask.copy(), dish_info

    monkeypatch.setattr("ui.analysis_window.load_image_worker_safe", slow_loader)
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
        observed["button_enabled"] = window.btn_apply.isEnabled()

    QTimer.singleShot(30, user_action)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=10000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=10000)
    heartbeat.stop()

    assert observed.get("petri_during_action") is None
    assert observed.get("button_enabled") is True
    assert len(ticks) >= 2
    assert window.analysis_results is not None
    assert algorithm.thread_ids
    window.close()
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)


def test_load_error_is_recoverable(qtbot, tmp_path, monkeypatch):
    import ui.analysis_window as aw

    image_path = _write_image(tmp_path)
    criticals = []
    _patch_dialogs(monkeypatch, criticals)
    monkeypatch.setattr(
        aw,
        "load_image_worker_safe",
        lambda path: (_ for _ in ()).throw(ValueError("битый файл")),
    )
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(
        lambda: (
            window._init_operation is not None
            and not window._init_operation.is_running()
        ),
        timeout=5000,
    )
    assert any("битый файл" in message for message in criticals)
    assert "Ошибка" in window.status_label.text()
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_bundled_analysis_runs_in_background(qtbot, tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    algorithm = _install_model(monkeypatch, _SlowTiledAlgorithm(delay=0.2))
    _patch_dialogs(monkeypatch)
    gui_thread = threading.get_ident()
    window = _make_window(qtbot, image_path, monkeypatch)

    qtbot.waitUntil(lambda: window._analysis_busy, timeout=5000)
    assert not window.btn_apply.isEnabled()
    assert "NN:colony_seg" in window.status_label.text()
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)

    assert algorithm.thread_ids and algorithm.thread_ids[0] != gui_thread
    assert window.analysis_results is not None
    assert window.btn_apply.isEnabled()
    assert window.progress_bar.isHidden()
    window.close()
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)


def test_bundled_analysis_error_is_reported(qtbot, tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    criticals = []
    _patch_dialogs(monkeypatch, criticals)
    _install_model(monkeypatch)
    monkeypatch.setattr(
        AnalysisController,
        "calculate_algorithm_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("сбой анализа")),
    )
    window = _make_window(qtbot, image_path, monkeypatch)
    qtbot.waitUntil(
        lambda: (
            window._analysis_operation is not None
            and not window._analysis_operation.is_running()
        ),
        timeout=5000,
    )
    assert any("сбой анализа" in message for message in criticals)
    assert not window._analysis_busy
    assert window.btn_apply.isEnabled()
    window.close()
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)


class _CancellableTiledAlgorithm(_SlowTiledAlgorithm):
    def __init__(self):
        super().__init__(delay=0.03, tile_count=30)


def test_analysis_cancellation_stops_between_tiles(qtbot, tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    _install_model(monkeypatch, _CancellableTiledAlgorithm())
    _patch_dialogs(monkeypatch)
    window = _make_window(qtbot, image_path, monkeypatch)
    qtbot.waitUntil(lambda: window._analysis_busy, timeout=5000)
    window._analysis_operation.cancel()
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=5000)

    assert "отменён" in window.status_label.text()
    assert window.btn_apply.isEnabled()
    assert window.controller.colony_mask is None
    window.close()
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)


def test_close_during_analysis_defers_and_closes(qtbot, tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    algorithm = _install_model(monkeypatch, _SlowTiledAlgorithm(delay=0.3))
    _patch_dialogs(monkeypatch)
    window = _make_window(qtbot, image_path, monkeypatch)
    ticks = []
    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    qtbot.waitUntil(lambda: window._analysis_busy, timeout=5000)
    window.close()
    assert window.isVisible()
    assert "Завершение" in window.status_label.text()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
    heartbeat.stop()
    assert len(ticks) >= 2
    assert algorithm.thread_ids
    from testing.registry import _INSTANCES

    _INSTANCES.pop("NN:colony_seg", None)
