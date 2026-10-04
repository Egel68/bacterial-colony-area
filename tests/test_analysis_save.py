"""Задача 5.4: сохранение результата анализа из snapshot'а вне GUI-потока.

Покрывает: snapshot отображённого вида/режима на момент команды, сохранение
содержимого и размеров `QPixmap.save`, кодирование по выбранному расширению,
отзывчивость GUI при большом изображении и ошибки сохранения.
"""

import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage

from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams
from ui.analysis_window import AnalysisWindow
from ui.controllers.analysis_controller import AnalysisController

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


def _petri_stub(size):
    dish_info = PetriInfo(size // 2, size // 2, size // 2 - 10, (size, size))
    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle_mask, dish_info.center, dish_info.radius, 255, -1)
    return circle_mask, dish_info


def _patch_dialogs(monkeypatch, criticals=None):
    import ui.analysis_window as aw

    monkeypatch.setattr(aw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        aw.QMessageBox,
        "critical",
        staticmethod(
            lambda *a, **k: criticals.append(a[2]) if criticals is not None else None
        ),
    )


def _make_window(qtbot, tmp_path, monkeypatch, size=120):
    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((size, size, 3), 80, dtype=np.uint8))
    circle_mask, dish_info = _petri_stub(size)
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: (circle_mask.copy(), dish_info),
    )
    _patch_dialogs(monkeypatch)
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=30000)
    # Представление подготавливается асинхронно (задача 5.3).
    qtbot.waitUntil(
        lambda: window.image_label.pixmap() is not None
        and not window.image_label.pixmap().isNull(),
        timeout=5000,
    )
    return window


def _run_save(window, monkeypatch, target):
    import ui.analysis_window as aw

    monkeypatch.setattr(
        aw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "Images (*.png *.jpg)")),
    )
    window._save_result()


def test_save_preserves_snapshot_content_and_dimensions(
    qtbot, tmp_path, monkeypatch
):
    window = _make_window(qtbot, tmp_path, monkeypatch)
    target = tmp_path / "result.png"

    # Snapshot отображённого вида на момент команды.
    expected = window.image_label.pixmap().toImage().copy()
    _run_save(window, monkeypatch, target)
    # Пользователь переключает режим сразу после команды «Сохранить».
    window.view_mode_combo.setCurrentIndex(1)

    qtbot.waitUntil(
        lambda: window._save_operation is not None
        and not window._save_operation.is_running(),
        timeout=5000,
    )
    saved = QImage(str(target))
    assert not saved.isNull(), "файл результата не создан"
    assert (saved.width(), saved.height()) == (expected.width(), expected.height())
    assert saved.convertToFormat(QImage.Format.Format_RGB888) == expected.convertToFormat(
        QImage.Format.Format_RGB888
    ), "содержимое сохранённого файла не совпало с отображённым видом"


def test_save_uses_extension_selected_encoding(qtbot, tmp_path, monkeypatch):
    window = _make_window(qtbot, tmp_path, monkeypatch)
    for name, magic in (("result.png", b"\x89PNG"), ("result.jpg", b"\xff\xd8")):
        target = tmp_path / name
        _run_save(window, monkeypatch, target)
        qtbot.waitUntil(
            lambda: window._save_operation is not None
            and not window._save_operation.is_running(),
            timeout=5000,
        )
        header = target.read_bytes()[:2]
        assert header == magic[:2], f"{name}: неверное кодирование ({header!r})"


def test_large_image_save_keeps_heartbeat(qtbot, tmp_path, monkeypatch):
    """Сохранение результата по большому изображению не блокирует event loop."""
    size = 2500  # ~6 MP: достаточно крупное изображение, без экстремальной нагрузки
    # Шумное содержимое: PNG-кодирование большого отображения реально долго.
    rng = np.random.RandomState(3)
    noisy = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
    image_path = tmp_path / "dish.png"
    assert cv2.imwrite(str(image_path), noisy)
    circle_mask, dish_info = _petri_stub(size)
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: (circle_mask.copy(), dish_info),
    )
    _patch_dialogs(monkeypatch)
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=10000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=30000)

    window.resize(2400, 1600)
    qtbot.waitUntil(lambda: window.size().width() == 2400, timeout=5000)
    qtbot.wait(50)
    target = tmp_path / "big_result.png"

    ticks = []
    heartbeat = QTimer(window)
    heartbeat.setInterval(50)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    _run_save(window, monkeypatch, target)
    qtbot.waitUntil(
        lambda: window._save_operation is not None
        and not window._save_operation.is_running(),
        timeout=30000,
    )
    heartbeat.stop()

    assert target.exists()
    assert len(ticks) >= 2, "heartbeat timer почти не срабатывал"
    intervals = [(b - a) * 1000.0 for a, b in zip(ticks, ticks[1:])]
    intervals_sorted = sorted(intervals)
    p95 = intervals_sorted[int(0.95 * (len(intervals_sorted) - 1))]
    assert p95 <= 100, f"p95 heartbeat {p95:.0f} мс"
    assert intervals_sorted[-1] <= 250, (
        f"максимальный интервал {intervals_sorted[-1]:.0f} мс"
    )


def test_save_failure_is_reported(qtbot, tmp_path, monkeypatch):
    window = _make_window(qtbot, tmp_path, monkeypatch)
    criticals = []
    _patch_dialogs(monkeypatch, criticals)
    target = "/nonexistent-dir-xyz/result.png"
    _run_save(window, monkeypatch, target)
    qtbot.waitUntil(
        lambda: window._save_operation is not None
        and not window._save_operation.is_running(),
        timeout=5000,
    )
    assert criticals, "ошибка сохранения не показана"
    assert "Ошибка" in window.status_label.text()
