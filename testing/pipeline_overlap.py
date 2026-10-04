"""Перекрытие decode/compute с backpressure и детерминированной очисткой (задача 2.7).

Producer готовит следующие элементы (batch'и декодированных пар), пока
consumer обрабатывает текущий. Две формы backpressure:

- **по очереди** — очередь ограничена `maxsize`, producer блокируется до
  освобождения места;
- **по памяти** — суммарный объём декодированных входов (активные у
  consumer + заранее загруженные) не превышает `max_bytes`; producer
  блокируется, пока consumer не освободит элемент через `task_done()`.

Метрика **input-starvation** (доля worker-slot-seconds): время, когда
consumer готов принять следующую работу, а очередь входов пуста, делённое
на длительность активной фазы обработки. Период до первого элемента
(старт) в числитель не входит — по спеке намеренное ожидание стартовых
и финальных фаз не считается голоданием.

Детерминированная очистка при ошибке/отмене: `close()` останавливает
producer, освобождает очередь и дожидается потока (без утечек потоков и
освобождённых массивов). Класс работает как context manager.

Использование::

    source = iter_batches(...)
    with PrefetchLoader(source, maxsize=1, max_bytes=budget, size_fn=...) as loader:
        for batch in loader:
            process(batch)
            loader.task_done(batch)
    starvation = loader.input_starvation_ratio
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Optional

# Маркер конца потока (отличается от любых валидных данных).
_SENTINEL = object()


class _ErrorItem:
    """Обёртка исключения producer: доставляется consumer'у по порядку."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


# Таймаут ожидания в блокирующих циклах producer/consumer (сек).
_POLL_INTERVAL = 0.05


class PrefetchLoader(Iterator):
    """Ограниченный prefetch: producer заготовливает элементы заранее.

    Наследует `Iterator` (который уже является `Iterable`) — потребитель
    использует `for item in loader` и вызывает `task_done(item)` после
    обработки элемента.

    Параметры:
        `source` — исходная последовательность элементов (например, batch'и
            из `iter_batches`);
        `maxsize` — максимум заранее загруженных элементов в очереди;
        `max_bytes` — бюджет декодированных входов (активные + загруженные);
            None — без ограничения по памяти;
        `size_fn` — функция размера элемента в байтах (для бюджета памяти);
        `name` — имя потока producer (для отладки).
    """

    def __init__(
        self,
        source: Iterable[Any],
        maxsize: int = 1,
        max_bytes: Optional[int] = None,
        size_fn: Optional[Callable[[Any], int]] = None,
        name: str = "prefetch",
    ):
        self._source = source
        self._maxsize = max(1, int(maxsize))
        self._max_bytes = max_bytes
        self._size_fn = size_fn or (lambda _: 0)
        self._queue: queue.Queue = queue.Queue(maxsize=self._maxsize)
        self._mem_lock = threading.Condition()
        self._outstanding_bytes = 0
        self._name = name
        self._stop = threading.Event()
        self._producer_error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        # Метрики input-starvation.
        self._starvation_time = 0.0
        self._active_start: Optional[float] = None
        self._active_end: Optional[float] = None
        self._items_consumed = 0

    # ------------------------------------------------------------------
    # Управление producer-потоком
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Запускает producer (идемпотентно)."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._produce, name=f"prefetch-{self._name}", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Детерминированная очистка: останавливает producer и дожидается его."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Разбудить producer, ожидающий освобождения бюджета памяти.
        with self._mem_lock:
            self._mem_lock.notify_all()
        # Разблокировать producer, ожидающий места в очереди.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():  # pragma: no cover — не должно происходить
                thread.join(timeout=5.0)

    def __enter__(self) -> "PrefetchLoader":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self):  # pragma: no cover — страховка от утечки потока
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _produce(self) -> None:
        try:
            for item in self._source:
                if self._stop.is_set():
                    break
                item_size = int(self._size_fn(item))
                if not self._acquire_memory(item_size):
                    break  # отмена во время ожидания бюджета
                if not self._put(item):
                    self._release_memory(item_size)
                    break  # отмена во время ожидания очереди
            if not self._stop.is_set():
                self._put(_SENTINEL)
        except BaseException as exc:  # noqa: BLE001 — уведомить consumer
            self._producer_error = exc
            # Ошибка уходит в очередь как элемент: consumer получает её
            # ПОСЛЕ уже поставленных элементов (семантика потока).
            self._put(_ErrorItem(exc))

    def _acquire_memory(self, item_size: int) -> bool:
        """Backpressure по памяти: ждёт, пока бюджет не позволит загрузить.

        Элемент крупнее бюджета обрабатывается изолированно (задача 2.5):
        он проходит только при пустом бюджете, а следующие элементы ждут
        его освобождения через `task_done()`.
        """
        if self._max_bytes is None:
            return True
        with self._mem_lock:
            while (
                self._outstanding_bytes > 0
                and self._outstanding_bytes + item_size > self._max_bytes
                and not self._stop.is_set()
            ):
                self._mem_lock.wait(timeout=_POLL_INTERVAL)
            if self._stop.is_set():
                return False
            self._outstanding_bytes += item_size
            return True

    def _release_memory(self, item_size: int) -> None:
        with self._mem_lock:
            self._outstanding_bytes = max(0, self._outstanding_bytes - item_size)
            self._mem_lock.notify_all()

    def _put(self, item: Any) -> bool:
        """Backpressure по очереди: блокируется до освобождения места."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=_POLL_INTERVAL)
                return True
            except queue.Full:
                continue
        return False

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def __iter__(self) -> "PrefetchLoader":
        self.start()
        return self

    def __next__(self) -> Any:
        self.start()
        wait_started = time.perf_counter()
        while True:
            try:
                item = self._queue.get(timeout=_POLL_INTERVAL)
                break
            except queue.Empty:
                thread = self._thread
                if self._producer_error is not None and self._queue.empty():
                    raise self._producer_error
                if thread is not None and not thread.is_alive() and self._queue.empty():
                    raise StopIteration
        wait_time = time.perf_counter() - wait_started
        now = time.perf_counter()
        if self._active_start is None:
            self._active_start = wait_started
        self._active_end = now
        if isinstance(item, _ErrorItem):
            raise item.exc
        if item is _SENTINEL:
            raise StopIteration
        self._items_consumed += 1
        if self._items_consumed > 1:
            # Input-starvation: ожидание входа после первого элемента.
            # Первое ожидание — startup-фаза, исключается по спеке.
            self._starvation_time += wait_time
        return item

    def task_done(self, item: Any) -> None:
        """Освобождает элемент: уменьшает задолженность бюджета памяти.

        Вызывать после завершения обработки элемента (consumer).
        """
        if self._max_bytes is None:
            return
        self._release_memory(int(self._size_fn(item)))

    # ------------------------------------------------------------------
    # Метрики input-starvation
    # ------------------------------------------------------------------

    @property
    def starvation_time(self) -> float:
        """Суммарное ожидание входа consumer'ом (сек)."""
        return self._starvation_time

    @property
    def active_time(self) -> float:
        """Длительность активной фазы обработки (от первого до последнего элемента)."""
        if self._active_start is None or self._active_end is None:
            return 0.0
        return self._active_end - self._active_start

    @property
    def input_starvation_ratio(self) -> float:
        """Доля worker-slot-seconds с пустой очередью входов (0..1).

        Определение спеки: время, когда compute slot готов принять работу,
        но входы не подготовлены, делённое на доступные compute slot-seconds
        активной фазы. Стартовые/финальные ожидания исключены.
        """
        active = self.active_time
        if active <= 0:
            return 0.0
        return min(1.0, max(0.0, self._starvation_time / active))

    @property
    def items_consumed(self) -> int:
        return self._items_consumed

    @property
    def outstanding_bytes(self) -> int:
        with self._mem_lock:
            return self._outstanding_bytes
