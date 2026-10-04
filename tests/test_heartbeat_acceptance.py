"""Задача 8.1: автоматизированное heartbeat-покрытие GUI-сценариев.

Проверяет требование спеки «responsive-user-interface»: timer-heartbeat
(период 50 мс) продолжает срабатывать во время длительной фоновой работы
главных окон. На reference-профиле: p95 интервала ≤ 100 мс, максимум
≤ 250 мс.

Покрываемые области:
- анализ (загрузка/поиск чашки/детекция — CPU-bound);
- разметка: выбор файла, кисть, копирование изображений, сохранение, ZIP;
- тестирование: прогон, показ результатов, экспорт отчёта;
- чисто CPU-bound фоновая работа (`BackgroundOperation`).
"""

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

HEARTBEAT_PERIOD_MS = 50
HEARTBEAT_P95_LIMIT_MS = 100
HEARTBEAT_MAX_LIMIT_MS = 250


def measure_heartbeat(
    owner, qtbot, work, *, wait_predicate=None, timeout=30000, minimum_ticks=3
):
    """Замеряет интервалы heartbeat (QTimer 50 мс) во время work().

    `work` — callable, запускающий операцию; `wait_predicate` — условие
    завершения (по умолчанию — возврат work). Возвращает (p95, max) в мс.
    """
    from PyQt6.QtCore import QTimer

    ticks = []
    heartbeat = QTimer(owner)
    heartbeat.setInterval(HEARTBEAT_PERIOD_MS)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()
    try:
        work()
        if wait_predicate is not None:
            qtbot.waitUntil(lambda: bool(wait_predicate()), timeout=timeout)
    finally:
        heartbeat.stop()

    assert len(ticks) >= 2, f"heartbeat timer не дал измеримого интервала: {len(ticks)}"
    intervals = sorted((b - a) * 1000.0 for a, b in zip(ticks, ticks[1:]))
    p95 = intervals[int(0.95 * (len(intervals) - 1))]
    return p95, intervals[-1]


def assert_within_reference_limits(p95, max_interval):
    assert p95 <= HEARTBEAT_P95_LIMIT_MS, f"p95 heartbeat {p95:.0f} мс > 100 мс"
    assert max_interval <= HEARTBEAT_MAX_LIMIT_MS, (
        f"максимальный интервал {max_interval:.0f} мс > 250 мс"
    )


# ---------------------------------------------------------------------------
# CPU-bound фоновая работа (базовый сценарий)
# ---------------------------------------------------------------------------


def test_cpu_bound_background_operation_keeps_heartbeat(qtbot):
    """Heartbeat живёт во время чисто CPU-bound работы BackgroundOperation."""
    from ui.background import BackgroundOperation

    def cpu_work(ctx, rounds):
        acc = 0
        for index in range(rounds):
            # Замеряемо CPU-bound: перемножение матриц без sleep.
            matrix = np.full((240, 240), index % 7 + 1, dtype=np.float64)
            acc += float((matrix @ matrix.T)[0, 0])
            if index % 25 == 0:
                ctx.checkpoint()
        return acc

    op = BackgroundOperation(cpu_work)
    finished = []
    op.result_ready.connect(lambda result, gen: finished.append(result))

    p95, max_interval = measure_heartbeat(
        op,
        qtbot,
        lambda: op.start(rounds=1200, copy_inputs=False),
        wait_predicate=lambda: finished,
        timeout=30000,
    )
    assert finished, "фоновая операция не завершилась"
    assert_within_reference_limits(p95, max_interval)


# ---------------------------------------------------------------------------
# Анализ: загрузка + детекция
# ---------------------------------------------------------------------------


def test_analysis_load_and_detection_keep_heartbeat(qtbot, tmp_path, monkeypatch):
    """Heartbeat живёт при загрузке кадра и поиске чашки Петри (анализ)."""
    import cv2

    import ui.analysis_window as aw
    from ui.analysis_window import AnalysisWindow

    image_path = tmp_path / "dish.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 80, dtype=np.uint8))

    def slow_find(self, image):
        time.sleep(0.2)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        from analysis.geometry import PetriInfo

        h, w = image.shape[:2]
        return mask, PetriInfo(w // 2, h // 2, min(w, h) // 3, (h, w))

    monkeypatch.setattr(aw.AnalysisController, "find_petri_dish", slow_find)
    monkeypatch.setattr(aw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(aw.QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    window = AnalysisWindow(str(image_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    p95, max_interval = measure_heartbeat(
        window,
        qtbot,
        lambda: None,
        wait_predicate=lambda: (
            window.original_image is not None
            and window._init_operation is not None
            and not window._init_operation.is_running()
            and window._analysis_operation is not None
            and not window._analysis_operation.is_running()
        ),
        timeout=15000,
    )
    assert_within_reference_limits(p95, max_interval)


# ---------------------------------------------------------------------------
# Разметка: выбор файла, кисть, копирование, сохранение, ZIP
# ---------------------------------------------------------------------------


@pytest.fixture
def labeling_window(qtbot, tmp_path, monkeypatch):
    import labeling.labeling_window as lw
    from labeling.labeling_window import LabelingWindow

    win = LabelingWindow(tmp_path / "hb-session")
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    if win._list_operation is not None and win._list_operation.is_running():
        win._list_operation.cancel()
        qtbot.waitUntil(lambda: not win._list_operation.is_running(), timeout=5000)
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


def test_labeling_file_selection_keeps_heartbeat(qtbot, labeling_window, monkeypatch):
    """Heartbeat живёт при выборе/загрузке файла в разметке."""
    import cv2

    window = labeling_window
    for index in range(12):
        path = window.source_dir / f"sel-{index}.png"
        cv2.imwrite(str(path), np.full((1200, 1200, 3), index * 5 + 1, dtype=np.uint8))
    window._load_file_list()

    def select_next():
        item = window.file_list.item(1) or window.file_list.item(0)
        assert item is not None
        window._on_file_selected(item)

    qtbot.waitUntil(
        lambda: (
            window._list_operation is not None
            and not window._list_operation.is_running()
            and window.file_list.count() > 0
        ),
        timeout=5000,
    )
    import labeling.labeling_window as lw

    real_loader = lw.load_image_worker_safe

    def delayed_loader(path):
        time.sleep(0.2)
        return real_loader(path)

    monkeypatch.setattr(lw, "load_image_worker_safe", delayed_loader)
    window._on_file_list_loaded(
        [str(path) for path in sorted(window.source_dir.glob("*.png"))],
        window._list_generation,
    )
    window._selection_generation += 1  # игнорировать автозагрузку первого файла
    p95, max_interval = measure_heartbeat(
        window,
        qtbot,
        select_next,
        wait_predicate=lambda: (
            window.paint_label._image is not None
            and window._detect_operation is not None
            and not window._detect_operation.is_running()
        ),
        timeout=15000,
    )
    assert_within_reference_limits(p95, max_interval)


def test_labeling_brush_strokes_keep_heartbeat(qtbot, labeling_window):
    """Heartbeat живёт при серии мазков кисти на крупном кадре."""
    window = labeling_window
    size = 3000  # крупный кадр (~9 MP)
    window.paint_label.set_image(
        np.full((size, size, 3), 70, dtype=np.uint8),
        np.zeros((size, size), dtype=np.uint8),
    )
    window.paint_label.brush_size = 20

    state = {"i": 0}
    total = 1200

    def stroke_tick():
        if state["i"] >= total:
            driver.stop()
            return
        window.paint_label._paint_at(
            (
                100 + (state["i"] * 7) % (size - 200),
                100 + (state["i"] * 11) % (size - 200),
            )
        )
        state["i"] += 1

    from PyQt6.QtCore import QTimer

    driver = QTimer(window)
    driver.setInterval(1)
    driver.timeout.connect(stroke_tick)

    def start_strokes():
        driver.start()

    p95, max_interval = measure_heartbeat(
        window,
        qtbot,
        start_strokes,
        wait_predicate=lambda: state["i"] >= total,
        timeout=30000,
    )
    assert state["i"] == total
    if window._detect_operation is not None and window._detect_operation.is_running():
        window._detect_operation.cancel()
        qtbot.waitUntil(lambda: not window._detect_operation.is_running(), timeout=5000)
    assert_within_reference_limits(p95, max_interval)


def test_labeling_copy_images_keeps_heartbeat(qtbot, labeling_window, monkeypatch):
    """Heartbeat живёт при копировании изображений в сессию (задача 6.4)."""
    import cv2

    window = labeling_window
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        source_dir = Path(tmp)
        files = []
        for index in range(6):
            path = str(source_dir / f"copy-{index}.png")
            cv2.imwrite(
                path,
                np.full((1500, 1500, 3), index * 9 + 2, dtype=np.uint8),
            )
            files.append(path)
        monkeypatch.setattr(
            "labeling.labeling_window.QFileDialog.getOpenFileNames",
            lambda *a, **k: (files, ""),
        )
        import labeling.labeling_window as lw

        real_copy = lw.shutil.copy2

        def delayed_copy(src, dst):
            time.sleep(0.12)
            return real_copy(src, dst)

        monkeypatch.setattr(lw.shutil, "copy2", delayed_copy, raising=False)
        p95, max_interval = measure_heartbeat(
            window,
            qtbot,
            window._on_add_images,
            wait_predicate=lambda: (
                getattr(window, "_copy_operation", None) is not None
                and not window._copy_operation.is_running()
            ),
            timeout=30000,
        )
    assert_within_reference_limits(p95, max_interval)


def test_labeling_save_and_zip_keep_heartbeat(qtbot, labeling_window, monkeypatch):
    """Heartbeat живёт при сохранении маски и ZIP-экспорте."""
    window = labeling_window
    rng = np.random.RandomState(3)
    size = 3000
    mask = (rng.randint(0, 2, (size, size), dtype=np.uint8) * 255).astype(np.uint8)
    window.paint_label.set_image(np.zeros((size, size, 3), dtype=np.uint8), mask)
    # Детерминированно удерживаем только файловую операцию дольше heartbeat
    # интервала; изображение и маска остаются компактными и реалистичными.

    real_save = window.controller.save_mask

    def delayed_save(mask_snapshot, path):
        time.sleep(0.35)
        return real_save(mask_snapshot, path)

    window.controller.save_mask = delayed_save
    window.current_stem = "hb"
    window.current_path = window.source_dir / "hb.png"

    def save_then_zip():
        window._on_save()

    p95_save, max_save = measure_heartbeat(
        window,
        qtbot,
        save_then_zip,
        wait_predicate=lambda: (
            window._save_operation is not None
            and not window._save_operation.is_running()
        ),
        minimum_ticks=1,
        timeout=30000,
    )
    assert_within_reference_limits(p95_save, max_save)

    real_zip = window.controller.export_session_to_zip

    def delayed_zip(*args, **kwargs):
        time.sleep(0.35)
        return real_zip(*args, **kwargs)

    window.controller.export_session_to_zip = delayed_zip
    zip_target = window.session_dir.parent / "hb-session.zip"
    monkeypatch.setattr(
        "labeling.labeling_window.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(zip_target), "ZIP архивы (*.zip)"),
    )
    p95_zip, max_zip = measure_heartbeat(
        window,
        qtbot,
        window._on_export_zip,
        wait_predicate=lambda: (
            getattr(window, "_export_operation", None) is not None
            and not window._export_operation.is_running()
        ),
        timeout=30000,
        minimum_ticks=1,
    )
    assert_within_reference_limits(p95_zip, max_zip)


# ---------------------------------------------------------------------------
# Тестирование: прогон, показ результатов, экспорт отчёта
# ---------------------------------------------------------------------------


def test_testing_run_and_presentation_keep_heartbeat(qtbot, tmp_path, monkeypatch):
    """Heartbeat живёт во время прогона и показа результатов (окно тестирования)."""
    import cv2

    from testing.registry import _INSTANCES
    from ui.testing_window import TestingWindow

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    for index in range(8):
        image = np.full((64, 64, 3), index * 7 + 1, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "source" / f"p{index}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"p{index}_mask.png"), mask)

    class _HbAlgo:
        name = "HbRunAlgo"
        description = "тест: heartbeat прогона"

        def detect(self, image, is_cropped=False):
            matrix = np.full((90, 90), 2.0, dtype=np.float64)
            float((matrix @ matrix.T)[0, 0])
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["HbRunAlgo"] = _HbAlgo()
    window = TestingWindow()
    qtbot.addWidget(window)
    window.dataset_input.setText(str(tmp_path))
    monkeypatch.setattr("ui.testing_window.QMessageBox.warning", lambda *a, **k: None)
    monkeypatch.setattr("ui.testing_window.QMessageBox.critical", lambda *a, **k: None)
    for name, checkbox in window._algo_checkboxes.items():
        checkbox.setChecked(name == "HbRunAlgo")
    try:
        p95, max_interval = measure_heartbeat(
            window,
            qtbot,
            window._start_run,
            wait_predicate=lambda: (
                window._results is not None
                and window._thread is not None
                and not window._thread.isRunning()
            ),
            minimum_ticks=10,
            timeout=30000,
        )
        assert window.summary_model.rowCount() >= 1, "результаты не показаны"
    finally:
        _INSTANCES.pop("HbRunAlgo", None)

    assert_within_reference_limits(p95, max_interval)


def test_testing_report_export_keeps_heartbeat(qtbot, tmp_path, monkeypatch):
    """Heartbeat живёт при экспорте HTML-отчёта (фоновая генерация, 7.3)."""
    from ui.testing_window import TestingWindow

    window = TestingWindow()
    qtbot.addWidget(window)
    window._results = {
        "ClassicDefault": {
            f"sample{index}": {
                "source": {
                    "iou": 0.9,
                    "dice": 0.95,
                    "f1": 0.92,
                    "precision": 0.93,
                    "recall": 0.91,
                    "accuracy": 0.99,
                    "tp": 90,
                    "fp": 0,
                    "fn": 10,
                    "tn": 900,
                }
            }
            for index in range(40)
        }
    }
    # HTML generation itself is CPU work; model a slow report builder to make
    # the worker phase observable independent of the host's storage speed.
    import ui.testing_window as tw

    real_generate = tw.generate_report_atomic

    def delayed_report(*args, **kwargs):
        time.sleep(0.35)
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(tw, "generate_report_atomic", delayed_report)
    window._results["ClassicDefault"] = {
        f"sample{index}": window._results["ClassicDefault"]["sample0"]
        for index in range(6000)
    }
    out_path = str(tmp_path / "hb_report.html")
    monkeypatch.setattr(
        "ui.testing_window.QFileDialog.getSaveFileName",
        lambda *a, **k: (out_path, "HTML reports (*.html)"),
    )

    p95, max_interval = measure_heartbeat(
        window,
        qtbot,
        window._export_report,
        wait_predicate=lambda: (
            window._export_op is not None
            and not window._export_op.is_running()
            and Path(out_path).exists()
        ),
        timeout=30000,
        minimum_ticks=1,
    )
    assert Path(out_path).exists(), "отчёт не опубликован"
    assert_within_reference_limits(p95, max_interval)
