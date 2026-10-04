"""Задача 6.7: deferred close и безопасная отмена в окне разметки.

Покрывает: закрытие во время непрерываемой операции (экспорт/детектор),
состояние «Завершение…», живой event loop, прекращение новой работы и
закрытие окна в safe-точке.
"""

import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QTimer

from labeling.labeling_window import LabelingWindow

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    win = LabelingWindow(tmp_path / "close-session")
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    import labeling.labeling_window as lw

    monkeypatch.setattr(
        lw.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(lw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        lw.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: lw.QMessageBox.StandardButton.Yes),
    )
    return win


def _start_export(window, monkeypatch, tmp_path):
    import labeling.labeling_window as lw

    target = tmp_path / "out.zip"
    monkeypatch.setattr(
        lw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "ZIP архивы (*.zip)")),
    )
    window._on_export_zip()
    return target


def test_close_when_idle_closes_immediately(qtbot, window):
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_close_during_uninterruptible_export_defers_and_closes_later(
    qtbot, window, monkeypatch, tmp_path
):
    """Непрерываемая операция завершается сама, окно закрывается после неё."""
    window.controller.export_session_to_zip = (
        lambda *a, **k: time.sleep(0.4)  # «native-вызов» без checkpoint
    )
    target = _start_export(window, monkeypatch, tmp_path)
    qtbot.waitUntil(
        lambda: (
            window._export_operation is not None
            and window._export_operation.is_running()
        ),
        timeout=5000,
    )

    ticks = []
    heartbeat = QTimer(window)
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    window.close()
    assert window.isVisible(), "окно закрылось до safe-точки"
    assert "Завершение" in window.status_label.text()

    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
    heartbeat.stop()
    assert len(ticks) >= 2, "event loop блокировался во время завершения"
    assert not target.exists(), "отменённый экспорт опубликовал архив"


def test_new_work_stops_during_closing(qtbot, window, monkeypatch, tmp_path):
    window.controller.export_session_to_zip = lambda *a, **k: time.sleep(0.3)
    _start_export(window, monkeypatch, tmp_path)
    qtbot.waitUntil(
        lambda: (
            window._export_operation is not None
            and window._export_operation.is_running()
        ),
        timeout=5000,
    )
    window.close()
    assert window.isVisible()

    # Новая работа во время завершения не начинается.
    list_generation = window._list_generation
    window._load_file_list()
    assert window._list_generation == list_generation

    source = tmp_path / "new.png"
    cv2.imwrite(str(source), np.full((8, 8, 3), 9, dtype=np.uint8))
    import labeling.labeling_window as lw

    monkeypatch.setattr(
        lw.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: ([str(source)], "")),
    )
    window._on_add_images()
    assert window._copy_operation is None

    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_close_during_cooperative_copy_cancels_and_closes(
    qtbot, window, monkeypatch, tmp_path
):
    """Кооперативная отмена останавливает копирование между файлами."""
    import labeling.labeling_window as lw

    files = []
    for i in range(20):
        path = tmp_path / f"f_{i}.png"
        cv2.imwrite(str(path), np.full((8, 8, 3), i + 1, dtype=np.uint8))
        files.append(str(path))

    original_copy2 = lw.shutil.copy2

    def slow_copy2(src, dst):
        time.sleep(0.05)
        return original_copy2(src, dst)

    monkeypatch.setattr(lw.shutil, "copy2", slow_copy2)
    monkeypatch.setattr(
        lw.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: (files, "")),
    )
    window._on_add_images()
    qtbot.waitUntil(
        lambda: (
            window._copy_operation is not None and window._copy_operation.is_running()
        ),
        timeout=5000,
    )
    qtbot.wait(120)
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
    copied = list(window.source_dir.glob("*.png"))
    assert len(copied) < 20, "отмена не остановила копирование"
