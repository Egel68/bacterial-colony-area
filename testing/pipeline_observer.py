"""Наблюдатель событий конвейера (задача 7.1).

Позволяет слоям выше (`ui.testing_window`) наблюдать фазы работы pipeline
и контекстные ошибки отдельных задач, не меняя семантику вычислений.

Фазы (в порядке прохождения):

- `dataset_scan` — перечисление снимков датасета;
- `read_decode` — чтение/декодирование пар «изображение + маска»;
- `algorithm` — вычисление детекций алгоритмами;
- `comparison` — парное сравнение алгоритмов;
- `result_preparation` — сборка сводки/отчёта по результатам.

Методы — no-op по умолчанию: наблюдатель наследуется частично, реализуются
только нужные события. `execute_pipeline`/`run_all_with_comparison` принимают
`observer` опционально и не требуют его.
"""

from __future__ import annotations

from typing import Any, Dict


class PipelineObserver:
    """Callback-интерфейс событий конвейера. Все методы — no-op по умолчанию."""

    def on_phase(self, phase: str, **context: Any) -> None:
        """Смена фазы конвейера (см. список фаз в докстринге модуля)."""

    def on_task_error(self, context: str, error: str) -> None:
        """Контекстная ошибка отдельной задачи.

        `context` — человекочитаемое описание (например,
        `"ClassicDefault/s1/source"`); `error` — текст ошибки.
        Конвейер продолжает обработку остальных задач.
        """

    def on_progress(self, completed: int, total: int) -> None:
        """Прогресс: `completed` из `total` задач завершено."""


# Экспорт имён фаз для переиспользования и тестов.
PHASE_DATASET_SCAN = "dataset_scan"
PHASE_READ_DECODE = "read_decode"
PHASE_ALGORITHM = "algorithm"
PHASE_COMPARISON = "comparison"
PHASE_RESULT_PREPARATION = "result_preparation"

PHASES = (
    PHASE_DATASET_SCAN,
    PHASE_READ_DECODE,
    PHASE_ALGORITHM,
    PHASE_COMPARISON,
    PHASE_RESULT_PREPARATION,
)
