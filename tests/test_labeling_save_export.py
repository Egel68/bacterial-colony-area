"""Задача 6.6: сохранение маски из snapshot'а и атомарный ZIP-экспорт.

Покрывает: точность snapshot'а маски на момент действия, отзывчивость GUI
при большом сохранении (heartbeat), публикацию ZIP только после успеха и
сохранность ранее существовавшего целевого файла при ошибке/отмене.
"""

import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QTimer

from labeling.labeling_window import LabelingWindow

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

HEARTBEAT_PERIOD_MS = 50
HEARTBEAT_P95_LIMIT_MS = 100
HEARTBEAT_MAX_LIMIT_MS = 250


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    win = LabelingWindow(tmp_path / "save-session")
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


def _prepare_image_with_mask(window, size=(60, 80)):
    h, w = size
    image = np.full((h, w, 3), 50, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 10, 255, -1)
    window.paint_label.set_image(image, mask)
    window.current_stem = "sample"
    window.current_path = window.source_dir / "sample.png"
    return mask


def test_mask_saved_from_immutable_snapshot(qtbot, window):
    mask = _prepare_image_with_mask(window)
    expected = mask.copy()

    window._on_save()
    # Пользователь продолжает рисовать сразу после действия «Сохранить».
    cv2.circle(window.paint_label.mask, (10, 10), 5, 255, -1)

    saved_path = window.masks_dir / "sample_mask.png"
    qtbot.waitUntil(lambda: saved_path.exists(), timeout=5000)
    qtbot.waitUntil(
        lambda: (
            window._save_operation is not None
            and not window._save_operation.is_running()
        ),
        timeout=5000,
    )
    saved = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(saved, expected), (
        "сохранённая маска не совпадает с snapshot'ом на момент действия"
    )


def test_large_mask_save_keeps_heartbeat(qtbot, window):
    rng = np.random.RandomState(7)
    size = 6000  # «большая» маска: кодирование PNG занимает сотни мс
    big_mask = (rng.randint(0, 2, (size, size), dtype=np.uint8) * 255).astype(np.uint8)
    window.paint_label.set_image(np.zeros((size, size, 3), dtype=np.uint8), big_mask)
    window.current_stem = "big"
    window.current_path = window.source_dir / "big.png"

    ticks = []
    heartbeat = QTimer(window)
    heartbeat.setInterval(HEARTBEAT_PERIOD_MS)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    window._on_save()
    qtbot.waitUntil(
        lambda: (
            window._save_operation is not None
            and not window._save_operation.is_running()
        ),
        timeout=30000,
    )
    heartbeat.stop()

    saved_path = window.masks_dir / "big_mask.png"
    assert saved_path.exists()
    assert len(ticks) >= 2, "heartbeat timer почти не срабатывал"
    intervals = [(b - a) * 1000.0 for a, b in zip(ticks, ticks[1:])]
    intervals_sorted = sorted(intervals)
    p95 = intervals_sorted[int(0.95 * (len(intervals_sorted) - 1))]
    assert p95 <= HEARTBEAT_P95_LIMIT_MS, f"p95 heartbeat {p95:.0f} мс"
    assert intervals_sorted[-1] <= HEARTBEAT_MAX_LIMIT_MS, (
        f"максимальный интервал {intervals_sorted[-1]:.0f} мс"
    )


def test_zip_export_publishes_only_after_success(qtbot, window, tmp_path, monkeypatch):
    import labeling.labeling_window as lw

    (window.source_dir / "img.png").touch()
    target = tmp_path / "out.zip"
    monkeypatch.setattr(
        lw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "ZIP архивы (*.zip)")),
    )

    window._on_export_zip()
    qtbot.waitUntil(
        lambda: (
            window._export_operation is not None
            and not window._export_operation.is_running()
        ),
        timeout=10000,
    )
    assert target.exists(), "ZIP не опубликован после успеха"
    assert not (target.parent / f".{target.name}.partial").exists(), (
        "остался частичный временный архив"
    )
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
    assert any(name.endswith("img.png") for name in names)


def test_zip_failure_preserves_previous_target(qtbot, window, tmp_path, monkeypatch):
    import labeling.labeling_window as lw

    target = tmp_path / "out.zip"
    target.write_bytes(b"previous-archive-content")
    monkeypatch.setattr(
        lw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "ZIP архивы (*.zip)")),
    )

    def broken_export(session_dir, output_path, checkpoint=None, progress=None):
        # Частичная запись во временный файл, затем ошибка (как при сбое диска).
        Path(output_path).write_bytes(b"partial")
        raise OSError("диск недоступен")

    monkeypatch.setattr(window.controller, "export_session_to_zip", broken_export)
    window._on_export_zip()
    qtbot.waitUntil(
        lambda: (
            window._export_operation is not None
            and not window._export_operation.is_running()
        ),
        timeout=10000,
    )
    assert target.read_bytes() == b"previous-archive-content", (
        "целевой файл изменён при ошибке экспорта"
    )
    assert not (target.parent / f".{target.name}.partial").exists(), (
        "частичный архив оставлен под именем"
    )
    qtbot.waitUntil(lambda: "Ошибка" in window.status_label.text(), timeout=5000)


def test_zip_cancel_preserves_previous_target(qtbot, window, tmp_path, monkeypatch):
    import labeling.labeling_window as lw
    import time as _time

    target = tmp_path / "out.zip"
    target.write_bytes(b"previous-archive-content")
    monkeypatch.setattr(
        lw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "ZIP архивы (*.zip)")),
    )

    def slow_export(session_dir, output_path, checkpoint=None, progress=None):
        with open(output_path, "wb") as fh:
            for i in range(100):
                if checkpoint is not None:
                    checkpoint()  # кооперативная отмена между файлами
                fh.write(b"x" * 1024)
                if progress is not None:
                    progress(i + 1)
                _time.sleep(0.02)

    monkeypatch.setattr(window.controller, "export_session_to_zip", slow_export)
    window._on_export_zip()
    qtbot.waitUntil(
        lambda: (
            window._export_operation is not None
            and window._export_operation.is_running()
        ),
        timeout=5000,
    )
    qtbot.wait(150)
    window._export_operation.cancel()
    qtbot.waitUntil(lambda: not window._export_operation.is_running(), timeout=5000)
    qtbot.waitUntil(lambda: "отменён" in window.status_label.text(), timeout=5000)
    assert target.read_bytes() == b"previous-archive-content", (
        "целевой файл изменён при отмене экспорта"
    )
    assert not (target.parent / f".{target.name}.partial").exists(), (
        "частичный архив оставлен под именем"
    )


def test_zip_success_replaces_existing_target(qtbot, window, tmp_path, monkeypatch):
    import labeling.labeling_window as lw

    target = tmp_path / "out.zip"
    target.write_bytes(b"old-content")
    monkeypatch.setattr(
        lw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "ZIP архивы (*.zip)")),
    )
    window._on_export_zip()
    qtbot.waitUntil(
        lambda: (
            window._export_operation is not None
            and not window._export_operation.is_running()
        ),
        timeout=10000,
    )
    assert zipfile.is_zipfile(target), "успешный экспорт не заменил старый файл"
