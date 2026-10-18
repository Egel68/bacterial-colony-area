"""Задача 6.1: асинхронная загрузка изображения/маски в окне разметки.

Покрывает: загрузку вне GUI-потока, generation checks (устаревший результат
не перезаписывает более новый выбор файлов или режим) и переиспользование
уже декодированного оригинала для авто-поиска и обрезки.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.geometry import PetriInfo
from labeling.labeling_window import LabelingWindow
from ui.background import BackgroundOperation

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


@pytest.fixture
def session_window(qtbot, tmp_path):
    window = LabelingWindow(tmp_path / "label-session")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    # Два файла с различимым содержимым.
    cv2.imwrite(
        str(window.source_dir / "a.png"), np.full((60, 80, 3), 10, dtype=np.uint8)
    )
    cv2.imwrite(
        str(window.source_dir / "b.png"), np.full((60, 80, 3), 200, dtype=np.uint8)
    )
    window.controller.detect_petri = lambda image_bgr: PetriInfo(40, 30, 25, (60, 80))
    window._load_file_list()
    qtbot.waitUntil(lambda: window.file_list.count() > 0, timeout=5000)
    return window


def _select(window, index: int):
    item = window.file_list.item(index)
    assert item is not None
    window.file_list.setCurrentItem(item)
    window._on_file_selected(item)
    return item


def test_selection_loads_asynchronously_off_gui_thread(
    qtbot, session_window, monkeypatch
):
    window = session_window
    import threading

    thread_ids = []
    real_loader = None

    def tracked_loader(path):
        thread_ids.append(threading.get_ident())
        time.sleep(0.05)
        return real_loader(path)

    import labeling.labeling_window as lw

    real_loader = lw.load_image_worker_safe
    monkeypatch.setattr(lw, "load_image_worker_safe", tracked_loader)

    gui_thread = threading.get_ident()
    _select(window, 0)
    # До завершения загрузки канвас пуст, статус сообщает о загрузке.
    assert window.paint_label._image is None
    assert "Загрузка" in window.status_label.text()

    qtbot.waitUntil(lambda: window.paint_label._image is not None, timeout=5000)
    assert window.current_path == window.source_dir / "a.png"
    assert thread_ids and thread_ids[0] != gui_thread


def test_mask_is_loaded_together_with_image(qtbot, session_window):
    window = session_window
    mask = np.zeros((60, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 30), 10, 255, -1)
    assert cv2.imwrite(str(window.masks_dir / "a_mask.png"), mask)

    _select(window, 0)
    qtbot.waitUntil(lambda: window.paint_label._image is not None, timeout=5000)
    assert np.array_equal(window.paint_label.mask, mask)


def test_stale_result_cannot_overwrite_newer_selection(
    qtbot, session_window, monkeypatch
):
    """Поздняя загрузка файла A не перезаписывает уже выбранный файл B."""
    window = session_window
    import labeling.labeling_window as lw

    monkeypatch.setattr(
        BackgroundOperation, "cancel", lambda self: None
    )  # обе загрузки завершатся

    real_loader = lw.load_image_worker_safe

    def delayed_loader(path):
        if Path(path).name == "a.png":
            time.sleep(0.25)  # A завершится последним
        return real_loader(path)

    monkeypatch.setattr(lw, "load_image_worker_safe", delayed_loader)

    _select(window, 0)  # A (медленно)
    _select(window, 1)  # B (быстро)
    qtbot.waitUntil(
        lambda: window.current_path == window.source_dir / "b.png", timeout=5000
    )
    # Ждём, когда завершится и устаревшая загрузка A.
    qtbot.wait(400)

    assert window.current_path == window.source_dir / "b.png"
    assert window.paint_label._image is not None
    assert int(window.paint_label._image[0, 0, 0]) == 200, (
        "устаревший результат перезаписал более новый выбор"
    )


def test_stale_result_cannot_overwrite_mode_change(qtbot, session_window, monkeypatch):
    window = session_window
    import labeling.labeling_window as lw

    monkeypatch.setattr(BackgroundOperation, "cancel", lambda self: None)
    real_loader = lw.load_image_worker_safe

    def delayed_loader(path):
        time.sleep(0.2)
        return real_loader(path)

    monkeypatch.setattr(lw, "load_image_worker_safe", delayed_loader)

    _select(window, 0)
    generation_at_selection = window._selection_generation
    window._on_mode_changed(1)  # смена режима очищает канвас и поколение
    assert window.paint_label._image is None

    qtbot.wait(400)  # загрузка завершится устаревшей
    assert window.paint_label._image is None, (
        "устаревший результат перезаписал состояние после смены режима"
    )
    assert window._selection_generation > generation_at_selection


def test_late_result_with_old_generation_is_dropped(qtbot, session_window):
    window = session_window
    _select(window, 1)
    qtbot.waitUntil(lambda: window.paint_label._image is not None, timeout=5000)
    old_generation = window._selection_generation - 1

    stale = {
        "image": np.full((60, 80, 3), 7, dtype=np.uint8),
        "mask": None,
        "path": str(window.source_dir / "a.png"),
        "stem": "a",
    }
    window._on_image_loaded(stale, old_generation)
    assert int(window.paint_label._image[0, 0, 0]) == 200, (
        "результат со старым поколением был применён"
    )
    assert window.current_path == window.source_dir / "b.png"


def test_auto_detect_and_crop_reuse_decoded_original(
    qtbot, session_window, monkeypatch
):
    """Авто-поиск и обрезка используют декодированный оригинал, не читая файл."""
    window = session_window
    import labeling.labeling_window as lw

    def forbidden_loader(path):
        raise AssertionError(f"повторное чтение файла: {path}")

    monkeypatch.setattr(lw, "load_image", forbidden_loader)
    # Модальные диалоги в тестах не показываются.
    monkeypatch.setattr(
        lw.QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: None),
    )

    detected_inputs = []
    window.controller.detect_petri = lambda image_bgr: (
        detected_inputs.append(image_bgr.shape),
        PetriInfo(40, 30, 25, (60, 80)),
    )[1]

    _select(window, 0)
    qtbot.waitUntil(lambda: window.paint_label._image is not None, timeout=5000)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)
    assert detected_inputs == [(60, 80, 3)], (
        "авто-поиск не получил декодированный оригинал"
    )

    window._on_crop()
    expected_crop = window.cropped_dir / "a_cropped.png"
    qtbot.waitUntil(lambda: expected_crop.exists(), timeout=5000)
    qtbot.waitUntil(lambda: window.mode == "cropped", timeout=5000)
    assert list(window.cropped_dir.glob("*_cropped.png")), "обрезка не создала файл"
