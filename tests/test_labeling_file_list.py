"""Задача 6.2: обновление списка файлов сессии вне GUI-потока.

Покрывает: перечисление/сортировку тысяч файлов без блокировки GUI
(heartbeat в пределах reference-лимитов), идентичность списка контроллеру
и отбрасывание устаревших обновлений (смена режима/новое обновление).
"""

import time

import pytest
from PyQt6.QtCore import QTimer

from labeling.labeling_window import LabelingWindow

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

HEARTBEAT_PERIOD_MS = 50
HEARTBEAT_P95_LIMIT_MS = 100
HEARTBEAT_MAX_LIMIT_MS = 250

FILE_COUNT = 2000


@pytest.fixture
def big_session(qtbot, tmp_path):
    window = LabelingWindow(tmp_path / "big-session")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    for i in range(FILE_COUNT):
        (window.source_dir / f"img_{i:05d}.png").touch()
    return window


def test_file_list_refresh_keeps_heartbeat_and_sorts(qtbot, big_session, monkeypatch):
    window = big_session

    # Папка «отвечает медленно» (сценарий спеки): перечисление с паузами.
    real_list = window.controller.list_image_files

    def slow_list(directory):
        import time as _time

        files = real_list(directory)
        for i in range(0, len(files), 500):
            _time.sleep(0.05)
        return files

    window.controller.list_image_files = slow_list

    ticks = []
    heartbeat = QTimer(window)
    heartbeat.setInterval(HEARTBEAT_PERIOD_MS)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    window._load_file_list()
    qtbot.waitUntil(lambda: window.file_list.count() > 0, timeout=15000)
    qtbot.waitUntil(lambda: window.file_list.count() == FILE_COUNT, timeout=15000)
    heartbeat.stop()

    # Список совпадает с результатом контроллера (перечисление + сортировка).
    expected = [f.name for f in window.controller.list_image_files(window.source_dir)]
    actual = [window.file_list.item(i).text() for i in range(window.file_list.count())]
    assert actual == expected
    assert actual == sorted(actual)
    assert len(actual) == FILE_COUNT

    # Heartbeat в reference-лимитах во время обновления.
    intervals = [(b - a) * 1000.0 for a, b in zip(ticks, ticks[1:])]
    assert len(intervals) >= 2, "heartbeat timer почти не срабатывал"
    intervals_sorted = sorted(intervals)
    p95 = intervals_sorted[int(0.95 * (len(intervals_sorted) - 1))]
    assert p95 <= HEARTBEAT_P95_LIMIT_MS, f"p95 heartbeat {p95:.0f} мс"
    assert intervals_sorted[-1] <= HEARTBEAT_MAX_LIMIT_MS, (
        f"максимальный интервал {intervals_sorted[-1]:.0f} мс"
    )


def test_file_list_refresh_runs_off_gui_thread(qtbot, big_session, monkeypatch):
    import threading

    window = big_session

    thread_ids = []
    real_list = window.controller.list_image_files

    def tracked_list(directory):
        thread_ids.append(threading.get_ident())
        return real_list(directory)

    window.controller.list_image_files = tracked_list
    gui_thread = threading.get_ident()

    window._load_file_list()
    qtbot.waitUntil(lambda: window.file_list.count() == FILE_COUNT, timeout=15000)
    assert thread_ids and thread_ids[0] != gui_thread


def test_stale_file_list_refresh_is_dropped(qtbot, big_session):
    """Устаревшее обновление списка не перезаписывает более новое."""
    window = big_session
    import time as _time
    from pathlib import Path

    real_list = window.controller.list_image_files
    calls = {"n": 0}

    def flaky_list(directory):
        calls["n"] += 1
        if calls["n"] == 1:
            _time.sleep(0.25)  # первое обновление завершится последним
            return [Path("stale.png")]
        return real_list(directory)

    window.controller.list_image_files = flaky_list

    window._load_file_list()  # устареет
    window._load_file_list()  # более новое обновление

    qtbot.waitUntil(lambda: window.file_list.count() == FILE_COUNT, timeout=15000)
    qtbot.wait(400)  # даём завершиться устаревшему обновлению
    texts = [window.file_list.item(i).text() for i in range(window.file_list.count())]
    assert "stale.png" not in texts, (
        "устаревшее обновление списка перезаписало более новое"
    )
    assert len(texts) == FILE_COUNT
