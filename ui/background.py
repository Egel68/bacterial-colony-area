"""Жизненный цикл длительных фоновых операций для GUI-окон.

Задача 4.2 change `optimize-image-processing-ui-responsiveness`: единый
контракт фоновой операции для analysis/labeling/testing окон.

Контракт:

- **immutable snapshot входов** — `snapshot()` копирует numpy-массивы при
  старте, worker получает неизменяемые данные и не трогает widgets;
- **доставка status/progress/result/error** — только queued-сигналами в
  GUI-поток (создание/изменение widgets в worker запрещено);
- **generation checks** — результаты помечены id поколения: применяйте их,
  только если поколение всё ещё актуально (устаревший результат не должен
  перезаписать более новый выбор пользователя);
- **кооперативная отмена** — `cancel()`; worker проверяет
  `ctx.checkpoint()` между безопасными единицами; непрерываемый native-вызов
  не блокирует GUI и завершается естественно;
- **nonblocking deferred close** — `request_close()`: окно показывает
  «Завершение…» и продолжает обрабатывать события; сигнал `close_reached`
  появляется после возврата текущей операции (или сразу, если операции нет).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

log = logging.getLogger(__name__)

# Реестр выполняющихся операций: защищает QThread от уничтожения GC,
# пока worker ещё работает (иначе Qt аварийно завершает процесс).
_ACTIVE: set["BackgroundOperation"] = set()


class OperationCancelled(Exception):
    """Поднимается worker-функцией (через ctx.checkpoint()) при отмене."""


def snapshot(value: Any) -> Any:
    """Создаёт неизменяемую копию входных данных для worker-функции.

    numpy-массивы копируются, словари/списки/кортежи обрабатываются
    рекурсивно; остальные объекты передаются как есть.
    """
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    return value


class OperationContext:
    """Контекст worker-функции: прогресс, статус, кооперативная отмена."""

    def __init__(
        self,
        cancel_event: threading.Event,
        status_cb: Callable[[str], None],
        progress_cb: Callable[[int, int], None],
    ):
        self._cancel_event = cancel_event
        self._status_cb = status_cb
        self._progress_cb = progress_cb

    def status(self, text: str) -> None:
        self._status_cb(text)

    def progress(self, completed: int, total: int | None = None) -> None:
        self._progress_cb(completed, -1 if total is None else total)

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def checkpoint(self) -> None:
        """Безопасная точка отмены: вызывать между единицами работы."""
        if self._cancel_event.is_set():
            raise OperationCancelled()


class _Worker(QObject):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    canceled = pyqtSignal()
    progressed = pyqtSignal(int, int)
    statused = pyqtSignal(str)
    done = pyqtSignal()

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        cancel_event: threading.Event,
    ):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._cancel_event = cancel_event

    def run(self) -> None:
        ctx = OperationContext(
            self._cancel_event,
            self.statused.emit,
            self.progressed.emit,
        )
        try:
            result = self._fn(ctx, *self._args, **self._kwargs)
        except OperationCancelled:
            self.canceled.emit()
        except Exception as exc:  # noqa: BLE001 — ошибка уходит в GUI как текст
            log.exception("Background operation failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit(result)
        finally:
            self.done.emit()


class BackgroundOperation(QObject):
    """Одна фоновая операция с полным жизненным циклом.

    Использование::

        op = BackgroundOperation(work_fn, generation=self._generation, parent=self)
        op.status_changed.connect(self._on_status)
        op.result_ready.connect(self._on_result)   # (result, generation)
        op.error_raised.connect(self._on_error)    # (message, generation)
        op.finished.connect(self._on_finished)     # (generation,)
        op.close_reached.connect(self._deferred_close)
        op.start(image=image, mask=mask)           # входы копируются
    """

    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(int, int)
    result_ready = pyqtSignal(object, int)
    error_raised = pyqtSignal(str, int)
    cancellation_confirmed = pyqtSignal(int)
    finished = pyqtSignal(int)
    close_reached = pyqtSignal()

    def __init__(
        self,
        fn: Callable[..., Any],
        generation: int = 0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._fn = fn
        self.generation = int(generation)
        self._cancel_event = threading.Event()
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._close_requested = False
        self._running = False

    # ------------------------------------------------------------------
    # Управление
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def start(self, *args, copy_inputs: bool = True, **kwargs) -> None:
        """Запускает worker с входными данными (snapshot по умолчанию).

        `copy_inputs=True` (по умолчанию) копирует numpy-массивы — worker
        гарантированно получает неизменяемые данные. `copy_inputs=False`
        передаёт массивы по ссылке: вызывающий ГАРАНТИРУЕТ, что они не
        изменяются до завершения операции (например, окно только заменяет
        массивы целиком, но не мутирует их in-place).
        """
        if self._running:
            raise RuntimeError("BackgroundOperation уже выполняется")
        self._cancel_event = threading.Event()
        self._close_requested = False
        self._running = True
        _ACTIVE.add(self)

        if copy_inputs:
            args = snapshot(args)
            kwargs = snapshot(kwargs)

        self._thread = QThread(self)
        self._worker = _Worker(self._fn, args, kwargs, self._cancel_event)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.statused.connect(self.status_changed)
        self._worker.progressed.connect(self.progress_changed)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._worker.canceled.connect(self._on_canceled)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def cancel(self) -> None:
        """Кооперативная отмена: останавливает worker на ближайшей safe-точке.

        Непрерываемый native-вызов продолжит работу и завершится сам —
        GUI-поток при этом не блокируется.
        """
        self._cancel_event.set()

    def request_close(self) -> None:
        """Nonblocking deferred close: `close_reached` после safe-точки."""
        self._close_requested = True
        if not self._running:
            self.close_reached.emit()

    # ------------------------------------------------------------------
    # Обработка исхода (GUI-поток, queued)
    # ------------------------------------------------------------------

    def _on_succeeded(self, result: object) -> None:
        self.result_ready.emit(result, self.generation)

    def _on_failed(self, message: str) -> None:
        self.error_raised.emit(message, self.generation)

    def _on_canceled(self) -> None:
        self.cancellation_confirmed.emit(self.generation)

    def _on_thread_finished(self) -> None:
        self._running = False
        worker = self._worker
        self._worker = None
        self._thread = None
        _ACTIVE.discard(self)
        if worker is not None:
            worker.deleteLater()
        self.finished.emit(self.generation)
        if self._close_requested:
            self._close_requested = False
            self.close_reached.emit()
