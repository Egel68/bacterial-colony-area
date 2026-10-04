"""Задача 6.3: авто-поиск чашки и обрезка в фоновых worker-потоках.

Покрывает: геометрию чашки, пиксели обрезка, контракт имён файлов,
восстановление после ошибок и отбрасывание устаревших результатов.
"""

import threading

import cv2
import numpy as np
import pytest

from analysis.geometry import PetriInfo
from labeling.labeling_window import LabelingWindow

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


@pytest.fixture
def labeled_window(qtbot, tmp_path, monkeypatch):
    window = LabelingWindow(tmp_path / "crop-session")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    image = np.full((60, 80, 3), 40, dtype=np.uint8)
    cv2.circle(image, (40, 30), 20, (180, 180, 180), -1)
    assert cv2.imwrite(str(window.source_dir / "sample.png"), image)
    import labeling.labeling_window as lw

    monkeypatch.setattr(
        lw.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        lw.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    window._load_file_list()
    qtbot.waitUntil(lambda: window.file_list.count() > 0, timeout=5000)
    return window


def _select(window, index: int = 0):
    item = window.file_list.item(index)
    window.file_list.setCurrentItem(item)
    window._on_file_selected(item)


def test_auto_detect_runs_in_worker_and_preserves_geometry(
    qtbot, labeled_window, monkeypatch
):
    window = labeled_window
    thread_ids = []
    expected = PetriInfo(cx=41, cy=29, radius=22, image_shape=(60, 80))

    def stub_detect(image_bgr):
        thread_ids.append(threading.get_ident())
        assert image_bgr.shape == (60, 80, 3)
        return expected

    window.controller.detect_petri = stub_detect
    gui_thread = threading.get_ident()

    _select(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)
    qtbot.waitUntil(lambda: not window._detect_operation.is_running(), timeout=5000)

    assert thread_ids and thread_ids[0] != gui_thread, (
        "поиск чашки выполнялся не в worker-потоке"
    )
    assert window.petri_info.cx == 41 and window.petri_info.cy == 29
    assert window.petri_info.radius == 22
    assert window.spin_cx.value() == 41
    assert window.spin_cy.value() == 29
    assert window.spin_radius.value() == 22
    assert window.paint_label.show_petri_circle


def test_crop_preserves_pixels_and_naming_contract(qtbot, labeled_window):
    window = labeled_window
    window.controller.detect_petri = lambda image_bgr: PetriInfo(
        40, 30, 25, (60, 80)
    )
    _select(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)

    window._on_crop()
    expected_path = window.cropped_dir / "sample_cropped.png"
    qtbot.waitUntil(lambda: expected_path.exists(), timeout=5000)
    qtbot.waitUntil(
        lambda: not window._crop_operation.is_running(), timeout=5000
    )

    # Контракт имён и пиксели обрезка совпадают с контроллером.
    reference = window.controller.crop_by_petri(
        cv2.imread(str(window.source_dir / "sample.png")),
        PetriInfo(40, 30, 25, (60, 80)),
    )
    saved = cv2.imread(str(expected_path))
    assert saved is not None
    assert np.array_equal(saved, reference), "пиксели обрезка искажены"
    assert window.mode == "cropped"


def test_auto_detect_failure_recovers(qtbot, labeled_window, monkeypatch):
    window = labeled_window

    def broken_detect(image_bgr):
        raise RuntimeError("детектор сломан")

    window.controller.detect_petri = broken_detect
    _select(window)
    qtbot.waitUntil(
        lambda: window._detect_operation is not None
        and not window._detect_operation.is_running(),
        timeout=5000,
    )
    assert "Ошибка" in window.status_label.text() or "не найдена" in (
        window.status_label.text()
    )

    # Окно восстановилось: можно выбрать файл повторно.
    window.controller.detect_petri = lambda image_bgr: PetriInfo(
        40, 30, 25, (60, 80)
    )
    _select(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)


def test_crop_failure_recovers(qtbot, labeled_window, monkeypatch):
    window = labeled_window
    window.controller.detect_petri = lambda image_bgr: PetriInfo(
        40, 30, 25, (60, 80)
    )
    _select(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)

    def broken_save(mask, path):
        raise OSError("диск недоступен")

    window.controller.save_mask = broken_save
    window._on_crop()
    qtbot.waitUntil(
        lambda: not window._crop_operation.is_running(), timeout=5000
    )
    assert "Ошибка" in window.status_label.text()
    assert not (window.cropped_dir / "sample_cropped.png").exists()


def test_stale_detect_result_is_dropped(qtbot, labeled_window):
    window = labeled_window
    window.controller.detect_petri = lambda image_bgr: None  # чашка не найдена
    _select(window)
    qtbot.waitUntil(lambda: window.paint_label._image is not None, timeout=5000)
    qtbot.waitUntil(
        lambda: window._detect_operation is not None
        and not window._detect_operation.is_running(),
        timeout=5000,
    )
    assert window.petri_info is None
    old_generation = window._selection_generation - 1

    stale_info = PetriInfo(cx=1, cy=1, radius=1, image_shape=(60, 80))
    window._on_detect_finished(stale_info, old_generation)
    assert window.petri_info is None, "устаревший результат поиска применён"
    assert window.spin_cx.value() != 1
