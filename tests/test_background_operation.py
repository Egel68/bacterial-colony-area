"""Задача 4.2: жизненный цикл фоновых операций (BackgroundOperation).

Покрывает: доставку результатов только в GUI-поток, generation checks
(устаревшие результаты не применяются), кооперативную отмену, nonblocking
deferred close после непрерываемого native-вызова и snapshot-иммутабельность.
"""

import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QTimer

from ui.background import (
    BackgroundOperation,
    OperationCancelled,
    snapshot,
)

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui


def test_snapshot_copies_arrays_recursively():
    array = np.ones((4, 4), dtype=np.uint8)
    data = {"image": array, "nested": [array, (array,)]}
    copied = snapshot(data)
    array[:] = 0
    assert np.array_equal(copied["image"], np.ones((4, 4), dtype=np.uint8))
    assert np.array_equal(copied["nested"][0], np.ones((4, 4), dtype=np.uint8))
    assert np.array_equal(copied["nested"][1][0], np.ones((4, 4), dtype=np.uint8))


def test_result_delivered_on_gui_thread_and_worker_runs_off_thread(qtbot):
    gui_thread = threading.get_ident()
    seen = {"worker_thread": None, "result_thread": None, "result": None}

    def work(ctx):
        seen["worker_thread"] = threading.get_ident()
        ctx.status("работаем")
        ctx.progress(1, 2)
        return "done"

    op = BackgroundOperation(work, generation=7)

    def on_result(result, generation):
        seen["result_thread"] = threading.get_ident()
        seen["result"] = (result, generation)

    op.result_ready.connect(on_result)
    op.start()
    qtbot.waitUntil(lambda: seen["result"] is not None, timeout=5000)

    assert seen["worker_thread"] != gui_thread
    assert seen["result_thread"] == gui_thread
    assert seen["result"] == ("done", 7)
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)


def test_stale_result_is_not_applied_to_newer_generation(qtbot):
    """Generation check: поздний результат не перезаписывает новый выбор."""
    state = {"applied": None, "current_generation": 2}
    finished = []

    def on_result(result, generation):
        if generation != state["current_generation"]:
            return  # устаревший результат отбрасывается
        state["applied"] = result

    def slow_work(ctx):
        time.sleep(0.15)
        return "old"

    def fast_work(ctx):
        return "new"

    op_old = BackgroundOperation(slow_work, generation=1)
    op_old.result_ready.connect(on_result)
    op_old.finished.connect(lambda gen: finished.append(("old", gen)))
    op_old.start()

    op_new = BackgroundOperation(fast_work, generation=2)
    op_new.result_ready.connect(on_result)
    op_new.finished.connect(lambda gen: finished.append(("new", gen)))
    op_new.start()

    qtbot.waitUntil(lambda: len(finished) == 2, timeout=5000)
    qtbot.waitUntil(lambda: state["applied"] == "new", timeout=5000)
    assert ("old", 1) in finished and ("new", 2) in finished


def test_cooperative_cancellation_stops_worker_at_checkpoint(qtbot):
    progress = {"iterations": 0}
    outcome = {}

    def work(ctx):
        for _ in range(200):
            ctx.checkpoint()  # безопасная точка отмены
            progress["iterations"] += 1
            time.sleep(0.005)
        return "completed"

    op = BackgroundOperation(work)
    op.cancellation_confirmed.connect(lambda gen: outcome.setdefault("canceled", gen))
    op.finished.connect(lambda gen: outcome.setdefault("finished", gen))
    op.start()
    qtbot.waitUntil(lambda: progress["iterations"] > 2, timeout=5000)
    op.cancel()
    qtbot.waitUntil(lambda: "canceled" in outcome, timeout=5000)
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)

    assert outcome["canceled"] == op.generation
    assert progress["iterations"] < 200, "отмена не остановила worker"


def test_deferred_close_waits_for_uninterruptible_native_call(qtbot):
    """request_close не блокирует GUI: close_reached после возврата операции."""
    outcome = {"close_reached": False, "finished": False}
    ticks = []

    def native_work(ctx):
        # Непрерываемый native-вызов: checkpoint() внутри не вызывается.
        time.sleep(0.3)
        return "native-done"

    op = BackgroundOperation(native_work)
    op.close_reached.connect(lambda: outcome.update(close_reached=True))
    op.finished.connect(lambda gen: outcome.update(finished=True))

    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()

    op.start()
    qtbot.wait(20)
    assert op.is_running()
    op.request_close()
    qtbot.wait(80)
    assert not outcome["close_reached"], (
        "close_reached дошёл до завершения непрерываемой операции"
    )
    assert not outcome["close_reached"] and len(ticks) >= 2, (
        "GUI event loop заблокирован во время операции"
    )

    qtbot.waitUntil(lambda: outcome["close_reached"], timeout=5000)
    heartbeat.stop()
    assert outcome["finished"], "операция не сообщила о завершении"
    assert len(ticks) >= 4, "heartbeat не работал во время фоновой операции"


def test_deferred_close_reaches_immediately_when_idle(qtbot):
    outcome = {"close_reached": False}
    op = BackgroundOperation(lambda ctx: "unused")
    op.close_reached.connect(lambda: outcome.update(close_reached=True))
    op.request_close()
    qtbot.waitUntil(lambda: outcome["close_reached"], timeout=2000)


def test_error_is_delivered_with_generation(qtbot):
    outcome = {}

    def broken(ctx):
        raise ValueError("сломанная операция")

    op = BackgroundOperation(broken, generation=3)
    op.error_raised.connect(
        lambda message, generation: outcome.update(
            message=message, generation=generation
        )
    )
    op.start()
    qtbot.waitUntil(lambda: "message" in outcome, timeout=5000)
    assert "ValueError" in outcome["message"]
    assert "сломанная операция" in outcome["message"]
    assert outcome["generation"] == 3
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)


def test_worker_sees_snapshot_not_live_mutation(qtbot):
    outcome = {}

    def work(ctx, image):
        time.sleep(0.1)
        return int(image.sum())

    image = np.ones((10, 10), dtype=np.uint8)
    op = BackgroundOperation(work)
    op.result_ready.connect(lambda result, gen: outcome.update(result=result))
    op.start(image=image)
    image[:] = 0  # мутация после старта не должна влиять на worker
    qtbot.waitUntil(lambda: "result" in outcome, timeout=5000)
    assert outcome["result"] == 100
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)


def test_start_while_running_is_rejected(qtbot):
    release = threading.Event()

    def blocking(ctx):
        release.wait(timeout=5)
        return "ok"

    op = BackgroundOperation(blocking)
    op.start()
    assert op.is_running()
    with pytest.raises(RuntimeError):
        op.start()
    release.set()
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)


def test_cancelled_worker_reports_cancellation_not_error(qtbot):
    outcome = {}

    def work(ctx):
        for _ in range(100):
            ctx.checkpoint()
            time.sleep(0.005)
        return "done"

    op = BackgroundOperation(work)
    op.error_raised.connect(lambda *_: outcome.update(error=True))
    op.result_ready.connect(lambda *_: outcome.update(result=True))
    op.cancellation_confirmed.connect(lambda gen: outcome.update(canceled=True))
    op.start()
    qtbot.waitUntil(lambda: op.is_running(), timeout=2000)
    op.cancel()
    qtbot.waitUntil(lambda: outcome.get("canceled"), timeout=5000)
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)
    assert not outcome.get("error")
    assert not outcome.get("result")


def test_operation_cancel_exception_marks_cancellation(qtbot):
    outcome = {}

    def work(ctx):
        raise OperationCancelled()

    op = BackgroundOperation(work)
    op.cancellation_confirmed.connect(lambda gen: outcome.update(canceled=True))
    op.error_raised.connect(lambda *_: outcome.update(error=True))
    op.start()
    qtbot.waitUntil(lambda: outcome.get("canceled"), timeout=5000)
    qtbot.waitUntil(lambda: not op.is_running(), timeout=5000)
    assert not outcome.get("error")
