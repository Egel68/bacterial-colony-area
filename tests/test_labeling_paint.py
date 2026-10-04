"""Задача 6.5: canvas разметки — локальный рендеринг кисти и отзывчивость.

Проверяет:
- побитовую точность полноразмерной маски при рисовании/стирании на разных
  уровнях зума (координаты кисти привязаны к исходным пикселям);
- локальность обновлений отображения: события кисти не пересобирают полный
  overlay кадра (dirty rect + патч кеша уровней);
- сохранность zoom semantics (отображение widget<->image);
- heartbeat GUI в пределах reference-лимитов (p95 ≤ 100 мс, max ≤ 250 мс
  при периоде 50 мс) при рисовании на кадре ~20 MP.
"""

import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtTest import QTest

from labeling.labeling_window import PaintLabel

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

# Reference-лимиты heartbeat (см. specs/responsive-user-interface).
HEARTBEAT_PERIOD_MS = 50
HEARTBEAT_P95_LIMIT_MS = 100
HEARTBEAT_MAX_LIMIT_MS = 250


def _make_label(qtbot, width=320, height=240, image_size=(240, 180)):
    label = PaintLabel()
    qtbot.addWidget(label)
    label.resize(width, height)
    label.show()
    qtbot.waitExposed(label)
    h, w = image_size
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    cv2.circle(image, (w // 2, h // 2), min(h, w) // 3, (140, 140, 140), -1)
    label.set_image(image)
    return label


def _set_zoom(label, zoom_factor):
    label._zoom_factor = zoom_factor
    label._update_scaled()


def _image_to_widget_point(label, x: int, y: int) -> QPoint:
    """Точка изображения -> точка widget (для синтеза событий мыши)."""
    return QPoint(
        int(x / label._scale + label._offset_x),
        int(y / label._scale + label._offset_y),
    )


@pytest.mark.parametrize("zoom_factor", [0.3, 1.0, 2.5, 7.0])
def test_brush_paints_exact_mask_pixels_at_zoom_levels(qtbot, zoom_factor):
    label = _make_label(qtbot)
    _set_zoom(label, zoom_factor)
    label.brush_size = 18
    r = max(1, label.brush_size // 2)

    h, w = label.mask.shape
    for image_point in ((w // 3, h // 3), (2 * w // 3, 2 * h // 3)):
        widget_pos = _image_to_widget_point(label, *image_point)
        x, y = label._widget_to_image(widget_pos)
        assert 0 <= x < w and 0 <= y < h, (
            f"zoom={zoom_factor}: точка ({x}, {y}) вне кадра"
        )
        reference = np.zeros_like(label.mask)
        cv2.circle(reference, (x, y), r, 255, -1)
        reference[label.mask > 0] = 255

        label.is_drawing = True
        QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=widget_pos)
        assert np.array_equal(label.mask, reference), (
            f"zoom={zoom_factor}: маска после клика не бит-в-бит совпала "
            "с эталонной окружностью"
        )

    # Ластик убивает ровно те же пиксели.
    widget_pos = _image_to_widget_point(label, w // 3, h // 3)
    x, y = label._widget_to_image(widget_pos)
    reference = label.mask.copy()
    cv2.circle(reference, (x, y), r, 0, -1)
    label.is_drawing = False
    QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=widget_pos)
    assert np.array_equal(label.mask, reference), "ластик изменил лишние пиксели"


def test_brush_trail_via_mouse_move_is_exact(qtbot):
    label = _make_label(qtbot)
    _set_zoom(label, 1.7)
    label.brush_size = 12
    r = max(1, label.brush_size // 2)

    points = [QPoint(60 + i * 12, 50 + i * 7) for i in range(6)]
    reference = label.mask.copy()
    for pos in points:
        x, y = label._widget_to_image(pos)
        cv2.circle(reference, (x, y), r, 255, -1)

    QTest.mousePress(label, Qt.MouseButton.LeftButton, pos=points[0])
    for pos in points[1:]:
        QTest.mouseMove(label, pos)
    QTest.mouseRelease(label, Qt.MouseButton.LeftButton, pos=points[-1])

    assert np.array_equal(label.mask, reference), (
        "маска после протяжки кисти не бит-в-бит совпала с эталоном"
    )


def test_mask_stays_full_resolution(qtbot):
    label = _make_label(qtbot, image_size=(240, 180))
    assert label.mask.shape == (240, 180)
    _set_zoom(label, 0.2)
    pos = _image_to_widget_point(label, 120, 90)
    QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=pos)
    assert label.mask.shape == (240, 180)
    assert label.mask.dtype == np.uint8


def test_brush_events_do_not_rebuild_full_frame_overlay(qtbot, monkeypatch):
    """События кисти ограничены bbox: без пересборки полного кадра."""
    label = _make_label(qtbot, width=300, height=220, image_size=(400, 300))
    _set_zoom(label, 1.0)
    label.brush_size = 20

    build_calls = []
    original_build = PaintLabel._build_level
    monkeypatch.setattr(
        PaintLabel,
        "_build_level",
        lambda self, ps: (build_calls.append(ps), original_build(self, ps))[1],
    )
    composite_calls = []
    original_composite = PaintLabel._composite_region
    monkeypatch.setattr(
        PaintLabel,
        "_composite_region",
        lambda self, x0, y0, x1, y1: (
            composite_calls.append((x0, y0, x1, y1)),
            original_composite(self, x0, y0, x1, y1),
        )[1],
    )

    # Прогрев: один полный показ (построение кеша уровней допустимо).
    label.grab()
    qtbot.wait(20)
    build_calls.clear()
    composite_calls.clear()

    # Один мазок: обновление ограничено bbox кисти (с запасом на интерполяцию).
    label._paint_at(label._widget_to_image(QPoint(130, 110)))
    qtbot.wait(20)
    assert build_calls == [], "событие кисти пересобрало кеш отображения целиком"
    assert composite_calls, "repaint не обратился к композиту"
    r = max(1, label.brush_size // 2)
    # bbox кисти (2r+3) + запас на интерполяцию и коалесценцию update rect.
    local_limit = 2 * r + 16
    for x0, y0, x1, y1 in composite_calls:
        assert (x1 - x0) <= local_limit and (y1 - y0) <= local_limit, (
            f"один мазок обновил область {x1 - x0}x{y1 - y0} пикселей "
            f"(bbox кисти ~{2 * r + 3}px, лимит {local_limit}px)"
        )

    # Серия мазков: Qt коалесцирует dirty rects, но полный кадр не пересобирается.
    build_calls.clear()
    composite_calls.clear()
    frame_area = label.mask.shape[0] * label.mask.shape[1]
    for i in range(30):
        pos = QPoint(120 + (i % 5) * 8, 100 + (i % 4) * 8)
        label._paint_at(label._widget_to_image(pos))
    qtbot.wait(30)

    assert build_calls == [], "события кисти пересобрали кеш отображения целиком"
    assert composite_calls, "repaint не обратился к композиту"
    for x0, y0, x1, y1 in composite_calls:
        area = (x1 - x0) * (y1 - y0)
        assert area <= frame_area * 0.25, (
            f"обновление кисти затронуло область {x1 - x0}x{y1 - y0} "
            "пикселей — близко к полному кадру"
        )


def test_zoom_semantics_preserved(qtbot):
    label = _make_label(qtbot, width=300, height=220, image_size=(240, 180))
    for zoom_factor in (0.3, 1.0, 2.5, 7.0):
        _set_zoom(label, zoom_factor)
        # Обратное отображение точно до целого пикселя изображения.
        tolerance = 1.0 / label._scale + 1.5
        for pos in (QPoint(12, 34), QPoint(150, 110), QPoint(288, 208)):
            x, y = label._widget_to_image(pos)
            back = _image_to_widget_point(label, x, y)
            assert abs(back.x() - pos.x()) <= tolerance and (
                abs(back.y() - pos.y()) <= tolerance
            ), (
                f"zoom={zoom_factor}: отображение widget->image->widget "
                f"дрейфует: {pos} -> {(x, y)} -> {back}"
            )
        assert label.zoom_percent == int(zoom_factor * 100)
        # sizeHint соответствует масштабированному кадру (скролл при зуме).
        assert label.sizeHint().width() == label._scaled_size.width()


def test_level_cache_patch_stays_close_to_full_rebuild(qtbot):
    """Патч кеша уровней после кисти визуально совпадает с полной сборкой."""
    label = _make_label(qtbot, width=300, height=220, image_size=(400, 300))
    _set_zoom(label, 0.2)  # уменьшенное представление -> кеш уровней
    label.brush_size = 24
    ps = label._select_level(1.0 / label._scale)
    assert ps < 1.0, "тест ожидает работу через кеш уровней"

    label.grab()
    qtbot.wait(10)

    h, w = label.mask.shape
    for i in range(5):
        label._paint_at((w // 2 + i * 6, h // 2 + i * 4))
    qtbot.wait(10)

    patched = label._get_level(ps).copy()
    fresh = label._build_level(ps)
    diff = np.abs(patched.astype(np.int16) - fresh.astype(np.int16))
    assert float(diff.mean()) < 4.0, (
        f"патч кеша заметно разошёлся с полной сборкой: mean={diff.mean():.2f}"
    )
    assert float((diff > 16).mean()) < 0.01, (
        "более 1% пикселей кеша отображения разошлись с полной сборкой"
    )


def test_large_image_brush_heartbeat_within_reference_limits(qtbot):
    """Heartbeat GUI при рисовании на кадре ~20 MP остаётся в лимитах."""
    size = 4500  # 20.25 MP — нижняя граница «крупных 20–35 MP кадров»
    image = np.full((size, size, 3), 70, dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 3, (150, 150, 150), -1)

    label = PaintLabel()
    qtbot.addWidget(label)
    label.resize(800, 600)
    label.show()
    qtbot.waitExposed(label)
    label.set_image(image)
    label.brush_size = 20

    # Прогревочный полный показ вне измерения.
    label.grab()
    qtbot.wait(20)

    intervals = []
    ticks = []
    heartbeat = QTimer(label)
    heartbeat.setInterval(HEARTBEAT_PERIOD_MS)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    actions_done = []
    state = {"i": 0}
    total_strokes = 1500
    rng = np.random.RandomState(42)

    def stroke():
        if state["i"] >= total_strokes:
            return
        x = int(rng.randint(50, size - 50))
        y = int(rng.randint(50, size - 50))
        label._paint_at((x, y))
        state["i"] += 1
        actions_done.append(state["i"])

    driver = QTimer(label)
    driver.setInterval(0)
    driver.timeout.connect(stroke)
    driver.start()

    qtbot.waitUntil(lambda: state["i"] >= total_strokes, timeout=30000)
    driver.stop()
    heartbeat.stop()

    assert len(actions_done) == total_strokes
    assert len(ticks) >= 3, "heartbeat timer почти не срабатывал"
    for prev, cur in zip(ticks, ticks[1:]):
        intervals.append((cur - prev) * 1000.0)
    intervals_sorted = sorted(intervals)
    p95 = intervals_sorted[int(0.95 * (len(intervals_sorted) - 1))]
    max_interval = intervals_sorted[-1]
    assert p95 <= HEARTBEAT_P95_LIMIT_MS, (
        f"p95 heartbeat {p95:.0f} мс > {HEARTBEAT_P95_LIMIT_MS} мс"
    )
    assert max_interval <= HEARTBEAT_MAX_LIMIT_MS, (
        f"максимальный интервал heartbeat {max_interval:.0f} мс > "
        f"{HEARTBEAT_MAX_LIMIT_MS} мс"
    )
