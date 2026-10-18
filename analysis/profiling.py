"""Профилирование стадий обработки изображений (задача 3.1).

Замеряет wall time отдельных стадий конвейера — поиск чашки, классическая
предобработка, фильтрация компонентов и т.д. — не изменяя результатов работы
алгоритмов. Используется для подтверждения «узких мест» перед оптимизациями
(задачи 3.2–3.3) и для отчёта о производительности.

Использование::

    from analysis.profiling import profile_algorithm_run

    stages = profile_algorithm_run(image, is_cropped=False)
    # {"dish_search": 3.41, "classic_preprocess": 0.52, ...}
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np

# Активный профилировщик потока. `None` — режим без замеров (быстрая ветка:
# одна проверка глобального состояния на стадию, результаты не меняются).
_active: Optional["StageProfiler"] = None


class StageProfiler:
    """Собирает длительности стадий; повторные стадии суммируются по списку."""

    def __init__(self) -> None:
        self.stages: Dict[str, List[float]] = {}

    def record(self, name: str, duration: float) -> None:
        self.stages.setdefault(name, []).append(max(0.0, duration))

    def totals(self) -> Dict[str, float]:
        return {name: sum(values) for name, values in self.stages.items()}

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """Сводка: {стадия: {total, calls, mean}} для отчётов и тестов."""
        out: Dict[str, Dict[str, float]] = {}
        for name, values in self.stages.items():
            out[name] = {
                "total": float(sum(values)),
                "calls": float(len(values)),
                "mean": float(sum(values) / len(values)),
            }
        return out


@contextmanager
def measure_stage(name: str):
    """Замеряет длительность стадии, если профилирование активно.

    Без активного профилировщика не выполняет никаких замеров и не влияет
    на результат вычислений.
    """
    if _active is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        _active.record(name, time.perf_counter() - started)


@contextmanager
def profile_stages():
    """Активирует сбор длительностей стадий в текущем потоке."""
    global _active
    profiler = StageProfiler()
    previous = _active
    _active = profiler
    try:
        yield profiler
    finally:
        _active = previous


def active_profiler() -> Optional[StageProfiler]:
    return _active


def profile_algorithm_run(
    image: np.ndarray,
    is_cropped: bool = False,
    algorithm_name: str = "ClassicDefault",
) -> Dict[str, Dict[str, float]]:
    """Профилирует один прогон алгоритма и возвращает сводку по стадиям.

    Результаты детекции не изменяются и не сохраняются — функция нужна
    только для замеров.
    """
    from testing.registry import get_algorithm

    algorithm = get_algorithm(algorithm_name)
    with profile_stages() as profiler:
        algorithm.detect(image, is_cropped=is_cropped)
    return profiler.as_dict()
