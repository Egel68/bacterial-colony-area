"""Задача 6.4: пакетное копирование исходных файлов вне GUI-потока.

Покрывает: успех с прогрессом, семантику перезаписи `shutil.copy2` при
совпадении имён, ошибку копирования с сохранением уже скопированного и
отмену между файлами.
"""

import threading
import time

import numpy as np
import cv2
import pytest

from labeling.labeling_window import LabelingWindow

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


@pytest.fixture
def empty_window(qtbot, tmp_path, monkeypatch):
    window = LabelingWindow(tmp_path / "copy-session")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    import labeling.labeling_window as lw

    monkeypatch.setattr(
        lw.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(lw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    window._load_file_list()
    qtbot.waitUntil(lambda: not window._list_operation.is_running(), timeout=5000)
    return window


def _make_source_files(tmp_path, count: int) -> list[str]:
    files = []
    for i in range(count):
        path = tmp_path / f"src_{i}.png"
        cv2.imwrite(str(path), np.full((10, 12, 3), i + 1, dtype=np.uint8))
        files.append(str(path))
    return files


def _run_add(window, monkeypatch, files):
    import labeling.labeling_window as lw

    monkeypatch.setattr(
        lw.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: (files, "")),
    )
    window._on_add_images()


def test_copy_success_with_progress(qtbot, empty_window, tmp_path, monkeypatch):
    window = empty_window
    files = _make_source_files(tmp_path, 3)

    progress = []
    original = window._on_copy_progress
    monkeypatch.setattr(
        window,
        "_on_copy_progress",
        lambda done, total: (progress.append((done, total)), original(done, total))[1],
    )

    _run_add(window, monkeypatch, files)
    qtbot.waitUntil(
        lambda: (
            window._copy_operation is not None
            and not window._copy_operation.is_running()
        ),
        timeout=5000,
    )
    assert "Скопировано изображений: 3" in window.status_label.text()
    assert progress == [(1, 3), (2, 3), (3, 3)]
    copied = sorted(f.name for f in window.source_dir.glob("*.png"))
    assert copied == ["src_0.png", "src_1.png", "src_2.png"]
    qtbot.waitUntil(lambda: window.file_list.count() == 3, timeout=5000)


def test_copy_runs_off_gui_thread(qtbot, empty_window, tmp_path, monkeypatch):
    window = empty_window
    files = _make_source_files(tmp_path, 2)
    thread_ids = []
    gui_thread = threading.get_ident()
    import labeling.labeling_window as lw

    original_copy2 = lw.shutil.copy2

    def tracked_copy2(src, dst):
        thread_ids.append(threading.get_ident())
        return original_copy2(src, dst)

    monkeypatch.setattr(lw.shutil, "copy2", tracked_copy2)
    _run_add(window, monkeypatch, files)
    qtbot.waitUntil(
        lambda: (
            window._copy_operation is not None
            and not window._copy_operation.is_running()
        ),
        timeout=5000,
    )
    assert thread_ids and thread_ids[0] != gui_thread


def test_copy_overwrites_matching_names(qtbot, empty_window, tmp_path, monkeypatch):
    """Семантика перезаписи shutil.copy2 при совпадении имён сохранена."""
    window = empty_window
    files = _make_source_files(tmp_path, 1)
    # Существующий файл с тем же именем и другим содержимым.
    existing = window.source_dir / "src_0.png"
    cv2.imwrite(str(existing), np.full((10, 12, 3), 250, dtype=np.uint8))

    _run_add(window, monkeypatch, files)
    qtbot.waitUntil(
        lambda: (
            window._copy_operation is not None
            and not window._copy_operation.is_running()
        ),
        timeout=5000,
    )
    overwritten = cv2.imread(str(existing))
    expected = cv2.imread(files[0])
    assert np.array_equal(overwritten, expected), (
        "файл с совпадающим именем не перезаписан, как делал shutil.copy2"
    )


def test_copy_failure_keeps_already_copied_files(
    qtbot, empty_window, tmp_path, monkeypatch
):
    window = empty_window
    files = _make_source_files(tmp_path, 3)
    import labeling.labeling_window as lw

    original_copy2 = lw.shutil.copy2
    calls = {"n": 0}

    def failing_copy2(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("диск недоступен")
        return original_copy2(src, dst)

    monkeypatch.setattr(lw.shutil, "copy2", failing_copy2)
    _run_add(window, monkeypatch, files)
    qtbot.waitUntil(
        lambda: (
            window._copy_operation is not None
            and not window._copy_operation.is_running()
        ),
        timeout=5000,
    )
    assert "Ошибка" in window.status_label.text()
    assert (window.source_dir / "src_0.png").exists(), (
        "успешно скопированный файл не сохранён"
    )
    assert not (window.source_dir / "src_1.png").exists()
    # Частично скопированные файлы видны в списке.
    qtbot.waitUntil(lambda: window.file_list.count() >= 1, timeout=5000)


def test_copy_cancellation_stops_between_files(
    qtbot, empty_window, tmp_path, monkeypatch
):
    window = empty_window
    files = _make_source_files(tmp_path, 10)
    import labeling.labeling_window as lw

    original_copy2 = lw.shutil.copy2

    def slow_copy2(src, dst):
        time.sleep(0.05)
        return original_copy2(src, dst)

    monkeypatch.setattr(lw.shutil, "copy2", slow_copy2)
    _run_add(window, monkeypatch, files)
    qtbot.waitUntil(
        lambda: (
            window._copy_operation is not None and window._copy_operation.is_running()
        ),
        timeout=5000,
    )
    qtbot.wait(120)
    window._copy_operation.cancel()
    qtbot.waitUntil(lambda: not window._copy_operation.is_running(), timeout=5000)
    copied = list(window.source_dir.glob("*.png"))
    assert 0 < len(copied) < 10, (
        f"отмена не остановила копирование между файлами: {len(copied)}"
    )
    qtbot.waitUntil(lambda: "отменено" in window.status_label.text(), timeout=5000)
