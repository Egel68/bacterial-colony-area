"""Единая ресурсная политика для всех уровней параллелизма (задача 2.6).

Координирует три уровня конкуренции за одни и те же CPU-ядра:

1. **декодирование** (загрузка/распаковка входных файлов);
2. **внешние algorithm workers** (задачи ThreadPoolExecutor в
   `testing.scheduler`);
3. **внутренняя параллельность нативных библиотек** — пулы OpenCV
   (классические алгоритмы) и ONNX Runtime (нейросетевые модели).

Все три уровня считаются частями одного бюджета: произведение
«внешние × внутренние» не должно превышать ёмкость процесса (иначе
возникает oversubscription и растёт latency при неизменном throughput).

Принципы (см. design.md → Decisions 3):

- ёмкость берётся из process-affinity/cgroup-квоты (см.
  `testing.scheduler._visible_cpu_capacity`), а не из `os.cpu_count()`;
- при одном внешнем задании нативные пулы получают полную ёмкость —
  классическая обработка и tiled ONNX выигрывают от широкой внутренней
  параллельности;
- при нескольких внешних заданиях нативные пулы сужаются, чтобы суммарное
  число активных потоков осталось в бюджете;
- в интерактивном (GUI) режиме резервируется доля мощности под обработку
  событий интерфейса; политика при этом остаётся автоматической — ручных
  регуляторов нет;
- CLI-флаг `--workers` остаётся ВЕРХНЕЙ границей внешних workers, но не
  изменяет автоматически выбранные нативные лимиты сверх бюджета.

Использование::

    policy = resolve_resource_policy(workers=4, interactive=True)
    with apply_native_limits(policy):
        execute_pipeline(...)  # внешние и нативные лимиты согласованы

Быстрый бенчмарк сравнения конфигураций — см. `benchmark_workloads`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterator, List, Optional

# Резерв CPU-ёмкости под event loop интерактивного (GUI) приложения.
GUI_RESERVED_CORES = 1

# Максимальная доля бюджета, отдаваемая декодированию при перекрытии
# decode/compute. Остальное — вычислениям (см. задачу 2.7).
DECODE_BUDGET_RATIO = 0.25


@dataclass(frozen=True)
class ResourcePolicy:
    """Согласованный бюджет CPU для всех уровней параллелизма."""

    # Ёмкость процесса (ядра-эквиваленты), на которую опирается политика.
    total_capacity: int
    # Внешние algorithm workers (слоты ThreadPoolExecutor).
    algorithm_workers: int
    # Слоты декодирования/загрузки входных файлов.
    decode_workers: int
    # Потоки внутри одного алгоритма (OpenCV/ONNX Runtime).
    native_threads: int
    # "auto" — выбрано автоматически; "explicit" — задан пользователем (--workers).
    source: str
    # Что явно запросил пользователь (None — авто-режим).
    requested_workers: Optional[int] = None
    # Интерактивный (GUI) режим: зарезервирована доля мощности под UI.
    interactive: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def effective_concurrency(self) -> int:
        """Верхняя оценка одновременно активных потоков вычислений.

        Не заявляется как фактическое ускорение — только конфигурация
        параллелизма (ускорение подтверждается end-to-end замером, см.
        задачу 8.2).
        """
        return max(1, self.algorithm_workers) * max(1, self.native_threads)


def resolve_resource_policy(
    workers: Optional[int] = None,
    concurrent_decode: bool = False,
    interactive: bool = False,
    task_count: Optional[int] = None,
    total_capacity: Optional[int] = None,
) -> ResourcePolicy:
    """Считает согласованный бюджет CPU для заданной конфигурации.

    Параметры:
        `workers` — явный лимит внешних workers (CLI `--workers`): верхняя
            граница, а не требуемое число;
        `concurrent_decode` — перекрывать ли загрузку с вычислениями (задача 2.7);
        `interactive` — GUI-режим: резервируется доля мощности под UI;
        `task_count` — сколько задач готовится (ограничивает workers);
        `total_capacity` — ёмкость процесса (для тестов; по умолчанию
            определяется автоматически).
    """
    if workers is not None and workers <= 0:
        raise ValueError("workers must be positive")

    if total_capacity is None:
        from testing.scheduler import _visible_cpu_capacity

        total_capacity = _visible_cpu_capacity()
    capacity = max(1, int(total_capacity))

    # Интерактивный режим: оставляем ядро под обработку событий GUI.
    budget = capacity
    if interactive and capacity > 1:
        budget = capacity - GUI_RESERVED_CORES

    # Декодирование: либо перекрывающиеся загрузчики, либо последовательная
    # загрузка одним слотом (текущее поведение scheduler).
    if concurrent_decode and budget > 1:
        decode_workers = max(1, int(budget * DECODE_BUDGET_RATIO))
    else:
        decode_workers = 1
    compute_budget = max(1, budget - (decode_workers if concurrent_decode else 0))

    # Внешние workers: явный --workers — верхняя граница.
    if workers is not None:
        algorithm_workers = max(1, min(int(workers), compute_budget))
        source = "explicit"
    else:
        algorithm_workers = compute_budget
        source = "auto"
    if task_count is not None:
        algorithm_workers = max(1, min(algorithm_workers, max(1, int(task_count))))

    # Нативные пулы: делим остаток ёмкости между внешними задачами.
    # При одном внешнем задании нативный алгоритм (классика/tiled ONNX)
    # получает полный бюджет — иначе он ограничится одним потоком.
    if algorithm_workers >= capacity:
        native_threads = 1
    else:
        native_threads = max(1, capacity // algorithm_workers)

    return ResourcePolicy(
        total_capacity=capacity,
        algorithm_workers=algorithm_workers,
        decode_workers=decode_workers,
        native_threads=native_threads,
        source=source,
        requested_workers=workers,
        interactive=interactive,
    )


_current_policy: Optional[ResourcePolicy] = None


def current_policy() -> Optional[ResourcePolicy]:
    """Активная политика (для нативных адаптеров ONNX/OpenCV)."""
    return _current_policy


def onnx_session_options(policy: Optional[ResourcePolicy] = None):
    """SessionOptions для ONNX Runtime по активной политике.

    Лениво импортирует onnxruntime. Возвращает None, если onnxruntime
    недоступен или лимиты не определены.
    """
    policy = policy if policy is not None else _current_policy
    if policy is None:
        return None
    try:
        import onnxruntime
    except ImportError:
        return None
    options = onnxruntime.SessionOptions()
    threads = max(1, policy.native_threads)
    # Внутренние и межоператорные пулы держим в бюджете политики.
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return options


@contextmanager
def apply_native_limits(policy: ResourcePolicy) -> Iterator[ResourcePolicy]:
    """Применяет нативные лимиты OpenCV и делает политику текущей.

    Лимит OpenCV устанавливается вокруг вычислений и восстанавливается
    после — глобальное состояние не утекает в другие контексты приложения.
    ONNX-адаптеры получают лимит через `onnx_session_options()` при создании
    сессии (лимит применяется к новым сессиям).
    """
    global _current_policy

    import cv2

    previous_cv_threads: Optional[int]
    try:
        previous_cv_threads = cv2.getNumThreads()
    except AttributeError:
        previous_cv_threads = None
    previous_policy = _current_policy
    _current_policy = policy

    if previous_cv_threads is not None:
        try:
            cv2.setNumThreads(max(1, policy.native_threads))
        except cv2.error:  # pragma: no cover — зависит от сборки OpenCV
            pass
    try:
        yield policy
    finally:
        _current_policy = previous_policy
        if previous_cv_threads is not None:
            try:
                cv2.setNumThreads(previous_cv_threads)
            except cv2.error:  # pragma: no cover
                pass


def benchmark_workloads() -> List[str]:
    """Имена рабочих нагрузок для сравнения конфигураций (задача 2.6).

    Сравнение обязано покрывать оба класса вычислительных профилей:
    классический (OpenCV-native) и tiled ONNX (ONNX Runtime).
    """
    return ["ClassicDefault", "NN:colony_seg"]


def policy_matrix(
    total_capacity: Optional[int] = None,
    interactive: bool = False,
) -> List[ResourcePolicy]:
    """Конфигурации для сравнения производительности (matrix outer/native).

    Всегда содержит последовательную (workers=1) и консервативно
    параллельную (auto) конфигурации, а также явные ограничения — так
    сравнение покрывает обе крайности координации внешних и нативных
    потоков. Ускорение засчитывается только по end-to-end замеру
    этих конфигураций (задача 8.2).
    """
    return [
        resolve_resource_policy(
            workers=1, interactive=interactive, total_capacity=total_capacity
        ),
        resolve_resource_policy(
            workers=2, interactive=interactive, total_capacity=total_capacity
        ),
        resolve_resource_policy(
            workers=None, interactive=interactive, total_capacity=total_capacity
        ),
    ]


def describe(policy: ResourcePolicy) -> str:
    return (
        f"capacity={policy.total_capacity} workers={policy.algorithm_workers} "
        f"decode={policy.decode_workers} native={policy.native_threads} "
        f"source={policy.source}"
        f"{' interactive' if policy.interactive else ''}"
    )
