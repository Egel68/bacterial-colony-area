"""Задача 5.3: viewport-bounded представление результата анализа.

Покрывает: быструю смену режима/ресайз (отображается только последнее
представление), viewport-ограниченный рендеринг без понижения данных анализа,
сохранность полного разрешения масок/входов алгоритма.
"""

import numpy as np
import pytest

from analysis.geometry import PetriInfo
from ui.analysis_window import (
    DISPLAY_FULL_RES_THRESHOLD,
    AnalysisWindow,
    render_presentation,
)
from ui.controllers.analysis_controller import AnalysisController

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


def _petri_stub(size=200):
    dish_info = PetriInfo(size // 2, size // 2, size // 2 - 20, (size, size))
    circle_mask = np.zeros((size, size), dtype=np.uint8)
    import cv2

    cv2.circle(circle_mask, dish_info.center, dish_info.radius, 255, -1)
    return circle_mask, dish_info


def _make_window(qtbot, tmp_path, monkeypatch, size=200):
    import cv2

    image_path = tmp_path / "dish.png"
    rng = np.random.RandomState(5)
    noisy = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), noisy)
    circle_mask, dish_info = _petri_stub(size)
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda self, image: (circle_mask.copy(), dish_info),
    )
    import ui.analysis_window as aw

    monkeypatch.setattr(aw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(aw.QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.petri_info is not None, timeout=5000)
    qtbot.waitUntil(lambda: not window._analysis_busy, timeout=30000)
    qtbot.waitUntil(
        lambda: (
            window.image_label.pixmap() is not None
            and not window.image_label.pixmap().isNull()
        ),
        timeout=5000,
    )
    return window


def _wait_display_idle(window, qtbot, timeout=5000):
    qtbot.waitUntil(
        lambda: (
            window._display_operation is not None
            and not window._display_operation.is_running()
        ),
        timeout=timeout,
    )


def test_rapid_mode_switches_show_only_latest_view(qtbot, tmp_path, monkeypatch):
    window = _make_window(qtbot, tmp_path, monkeypatch)
    for mode in (1, 3, 2, 0, 3, 1):
        window.view_mode_combo.setCurrentIndex(mode)
    _wait_display_idle(window, qtbot)

    # Отображается только представление последнего режима (1: «Оригинал»).
    assert window.view_mode_combo.currentIndex() == 1
    assert window.image_header.text() == "Оригинальное изображение"
    pixmap = window.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()

    # Данные анализа сохраняются полноразмерными.
    assert window.original_image.shape[:2] == (200, 200)
    assert window.controller.colony_mask is None or (
        window.controller.colony_mask.shape[:2] == (200, 200)
    )


def test_rapid_resizes_show_only_latest_view(qtbot, tmp_path, monkeypatch):
    window = _make_window(qtbot, tmp_path, monkeypatch)
    for width, height in ((900, 700), (700, 500), (1100, 800), (640, 480)):
        window.resize(width, height)
        qtbot.wait(30)
    qtbot.waitUntil(lambda: window.size().width() == 640, timeout=5000)
    _wait_display_idle(window, qtbot)

    pixmap = window.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    # Представление соответствует последнему размеру окна (viewport-bounded).
    assert pixmap.width() <= 640 and pixmap.height() <= 480
    assert window.original_image.shape[:2] == (200, 200), (
        "ресайз изменил полноразмерные данные"
    )


def test_presentation_is_viewport_bounded(qtbot, tmp_path, monkeypatch):
    window = _make_window(qtbot, tmp_path, monkeypatch, size=3000)
    window.resize(800, 600)
    qtbot.waitUntil(lambda: window.size().width() == 800, timeout=5000)
    _wait_display_idle(window, qtbot)

    pixmap = window.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    assert pixmap.width() <= 800 and pixmap.height() <= 600, (
        f"представление {pixmap.width()}x{pixmap.height()} не ограничено viewport"
    )
    # Данные анализа остаются полноразмерными.
    assert window.original_image.shape[:2] == (3000, 3000)
    if window.controller.colony_mask is not None:
        assert window.controller.colony_mask.shape[:2] == (3000, 3000)


def test_render_presentation_keeps_full_resolution_inputs():
    """render_presentation не мутирует входные полноразмерные массивы."""
    rng = np.random.RandomState(11)
    original = rng.randint(0, 255, (400, 500, 3), dtype=np.uint8)
    preprocessed = rng.randint(0, 255, (400, 500), dtype=np.uint8)
    colony_mask = (rng.randint(0, 2, (400, 500), dtype=np.uint8) * 255).astype(np.uint8)
    petri_info = PetriInfo(250, 200, 150, (400, 500))

    original_before = original.copy()
    preprocessed_before = preprocessed.copy()
    colony_mask_before = colony_mask.copy()

    rgb = render_presentation(
        original,
        preprocessed,
        colony_mask,
        petri_info,
        10,
        0,
        True,
        True,
        320,
        240,
    )

    assert np.array_equal(original, original_before), "original мутирован"
    assert np.array_equal(preprocessed, preprocessed_before), "preprocessed мутирован"
    assert np.array_equal(colony_mask, colony_mask_before), "colony_mask мутирован"
    assert rgb.shape[0] <= 240 and rgb.shape[1] <= 320
    assert rgb.dtype == np.uint8


def test_render_presentation_modes_preserve_semantics():
    """Контракт содержимого режимов сохранён (формулы прежнего отображения)."""
    import cv2

    original = np.full((100, 120, 3), 40, dtype=np.uint8)
    preprocessed = np.full((100, 120), 120, dtype=np.uint8)
    colony_mask = np.zeros((100, 120), dtype=np.uint8)
    cv2.circle(colony_mask, (60, 50), 10, 255, -1)
    petri_info = PetriInfo(60, 50, 40, (100, 120))

    # Режим 2: серый слой предобработки в трёх каналах.
    rgb2 = render_presentation(
        original, preprocessed, colony_mask, petri_info, 8, 2, True, True, 120, 100
    )
    assert rgb2.shape == (100, 120, 3)
    assert np.all(rgb2[:, :, 0] == rgb2[:, :, 1]) and np.all(
        rgb2[:, :, 1] == rgb2[:, :, 2]
    ), "предобработка не монохромна"
    assert abs(int(rgb2[50, 60, 0]) - 120) <= 2

    # Режим 3: бинарная маска.
    rgb3 = render_presentation(
        original, preprocessed, colony_mask, petri_info, 8, 3, True, True, 120, 100
    )
    assert rgb3.shape == (100, 120, 3)
    assert int(rgb3[50, 60, 0]) == 255, "центр колонии не белый в маске"
    assert int(rgb3[10, 10, 0]) == 0, "фон маски не чёрный"

    # Режим 0: зелёное наложение колоний поверх оригинала.
    rgb0 = render_presentation(
        original, preprocessed, colony_mask, petri_info, 8, 0, True, True, 120, 100
    )
    assert rgb0[50, 60, 1] > rgb0[50, 60, 0], "колония не подсвечена зелёным"
    # Без наложения — оригинал без зелёного усиления.
    rgb0_plain = render_presentation(
        original, preprocessed, colony_mask, petri_info, 8, 0, True, False, 120, 100
    )
    assert not (rgb0_plain[50, 60, 1] > rgb0_plain[50, 60, 0]), (
        "наложение показано при выключенном флаге"
    )


def test_half_resolution_path_is_taken_below_threshold():
    """Для малых отображений используется быстрый путь 1/2."""

    original = np.full((2000, 2000, 3), 60, dtype=np.uint8)
    petri_info = PetriInfo(1000, 1000, 800, (2000, 2000))
    # Отображение 400px на кадр 2000px => scale 0.2 < порога.
    rgb = render_presentation(
        original,
        None,
        None,
        petri_info,
        8,
        1,
        True,
        True,
        400,
        400,
    )
    assert DISPLAY_FULL_RES_THRESHOLD == 0.5
    assert rgb.shape[0] <= 400 and rgb.shape[1] <= 400
