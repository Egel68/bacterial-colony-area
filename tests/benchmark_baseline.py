"""Базовые замеры производительности на reference-профиле (задача 1.3).

Записывает и проверяет базовые характеристики детерминированного прогона
эталонных фикстур (`tests/reference_fixtures.py`) в двух конфигурациях:

- **sequential** — один worker (`workers=1`);
- **current_workers** — текущая auto-конфигурация (`workers=None`).

Для каждой конфигурации фиксируются end-to-end wall time, process CPU time,
peak RSS, throughput и число workers. Плюс измеряется GUI-heartbeat (QTimer,
период 50 мс) во время фонового прогона и условия I/O.

Условия I/O разделяются ЯВНО (инвариант задачи 1.3):

- **измерено**: warm-cache поведение — логическое чтение/декодирование
  повторяющихся входов (page cache прогрет);
- **НЕ измерено**: cold-storage I/O (задержки холодного диска) — такие
  значения НЕ подменяются нулями и НЕ интерпретируются как отсутствие I/O.

Использование::

    UV_PROJECT_ENVIRONMENT=.venv-dev uv run python tests/benchmark_baseline.py --write
    UV_PROJECT_ENVIRONMENT=.venv-dev uv run python tests/benchmark_baseline.py --check

`--write` сохраняет отчёт в `tests/baseline/benchmark_baseline.json`;
`--check` пересобирает отчёт и проверяет структуру/санити (не абсолютные
тайминги: они зависят от нагрузки ОС; ускорение засчитывается только по
end-to-end замеру, см. задачу 8.2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

BASELINE_PATH = Path(__file__).parent / "baseline" / "benchmark_baseline.json"
SCHEMA_VERSION = 1

# Запуск как скрипта: корень проекта и каталог tests/ в sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# GUI-heartbeat: период таймера и бюджеты спеки (reference-окружение).
HEARTBEAT_PERIOD_MS = 50
HEARTBEAT_P95_BUDGET_MS = 100
HEARTBEAT_MAX_BUDGET_MS = 250


def _run_detection(
    fixture_names: List[str],
    algorithms: List[str],
    workers: int,
    telemetry,
) -> Dict[str, Any]:
    """Прогоняет детекции фикстур в пуле workers и собирает тайминги."""
    from reference_fixtures import build_fixture
    from testing.registry import get_algorithm

    tasks = [
        (fixture_name, algo_name)
        for fixture_name in fixture_names
        for algo_name in algorithms
    ]
    telemetry.start()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for fixture_name, algo_name in tasks:
            fixture = build_fixture(fixture_name)

            def _work(fixture=fixture, algo_name=algo_name):
                task_started = time.perf_counter()
                algo = get_algorithm(algo_name)
                algo.detect(fixture.image, is_cropped=fixture.is_cropped)
                telemetry.record_stage("detect", time.perf_counter() - task_started)

            futures.append(executor.submit(_work))
        for future in futures:
            future.result()
    duration = time.perf_counter() - started
    summary = telemetry.finish(workers=workers, batch_size=len(tasks))
    if not summary:
        # Telemetry выключен: минимальная сводка без счётчиков.
        return {
            "workers": workers,
            "tasks": len(tasks),
            "duration_seconds": duration,
            "process_cpu_seconds": None,
            "cpu_core_equivalents": None,
            "peak_rss_bytes": None,
            "tasks_per_second": (len(tasks) / duration if duration > 0 else None),
        }
    return {
        "workers": workers,
        "tasks": len(tasks),
        "duration_seconds": duration,
        "process_cpu_seconds": summary["resources"]["process_cpu_time"],
        "cpu_core_equivalents": summary["resources"]["cpu_core_equivalents"],
        "peak_rss_bytes": summary["resources"]["memory"]["peak_rss"],
        "tasks_per_second": (len(tasks) / duration if duration > 0 else None),
    }


def _measure_heartbeat(work, period_ms: int = HEARTBEAT_PERIOD_MS) -> Dict[str, Any]:
    """Измеряет интервалы GUI-heartbeat (QTimer) во время фоновой работы.

    Возвращает статистику интервалов; если Qt недоступен — статус
    `unavailable` (НЕ нулевые значения: неизвестное не подменяется).
    """
    try:
        from PyQt6.QtCore import QCoreApplication, QTimer
    except ImportError:
        return {"status": "unavailable", "reason": "PyQt6 не установлен"}

    import os
    import threading

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])

    intervals: List[float] = []
    last_tick = time.perf_counter()

    def _tick():
        nonlocal last_tick
        now = time.perf_counter()
        intervals.append((now - last_tick) * 1000.0)
        last_tick = now

    timer = QTimer()
    timer.setInterval(period_ms)
    timer.timeout.connect(_tick)

    worker_thread = threading.Thread(target=work, daemon=True)
    last_tick = time.perf_counter()
    timer.start()
    worker_thread.start()
    while worker_thread.is_alive():
        app.processEvents()
        time.sleep(0.001)
    worker_thread.join()
    timer.stop()
    app.processEvents()

    if not intervals:
        return {"status": "unavailable", "reason": "нет отсчётов heartbeat"}
    ordered = sorted(intervals)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "status": "measured",
        "period_ms": period_ms,
        "samples": len(intervals),
        "p95_ms": p95,
        "max_ms": ordered[-1],
    }


def build_report(
    fixture_names: List[str] | None = None,
    algorithms: List[str] | None = None,
) -> Dict[str, Any]:
    """Собирает базовый отчёт: sequential + current-workers + heartbeat + I/O."""
    from reference_fixtures import ALGORITHMS, FIXTURE_NAMES
    from testing.resource_policy import resolve_resource_policy
    from testing.telemetry import TelemetryCollector

    if fixture_names is None:
        fixture_names = list(FIXTURE_NAMES)
    if algorithms is None:
        algorithms = list(ALGORITHMS)

    # Прогрев page cache: первый проход — warm-up, замеры отражают warm-cache.
    _run_detection(fixture_names, algorithms, 1, TelemetryCollector(enabled=True))

    auto_policy = resolve_resource_policy(workers=None)
    modes = {
        "sequential": _run_detection(
            fixture_names, algorithms, 1, TelemetryCollector(enabled=True)
        ),
        "current_workers": _run_detection(
            fixture_names,
            algorithms,
            auto_policy.algorithm_workers,
            TelemetryCollector(enabled=True),
        ),
    }

    def _heartbeat_work():
        _run_detection(fixture_names, algorithms, 1, TelemetryCollector(enabled=False))

    heartbeat = _measure_heartbeat(_heartbeat_work)

    return {
        "schema_version": SCHEMA_VERSION,
        "workload": {
            "fixtures": list(fixture_names),
            "algorithms": list(algorithms),
            "cache_state": "warm-cache (повторные проходы; page cache прогрет)",
        },
        "modes": modes,
        "heartbeat": heartbeat,
        # Инвариант 1.3: измеренное warm-cache поведение отделено от
        # неизмеренного cold-storage I/O; неизвестное не подменяется нулями.
        "io_conditions": {
            "measured": (
                "warm-cache: логическое чтение/декодирование повторных входов"
            ),
            "unmeasured": ("cold-storage I/O: задержки холодного диска НЕ измерялись"),
            "physical_io_counters": "см. telemetry.io_condition (могут быть 0 из-за page cache)",
            "note": (
                "Ноль физических read-bytes не означает отсутствие логических "
                "чтений и не доказывает отсутствие I/O-задержек."
            ),
        },
    }


def _check_report(report: Dict[str, Any]) -> List[str]:
    """Структурная проверка отчёта (не абсолютных таймингов)."""
    problems: List[str] = []
    for key in ("schema_version", "workload", "modes", "heartbeat", "io_conditions"):
        if key not in report:
            problems.append(f"нет поля {key!r}")
    modes = report.get("modes", {})
    for mode in ("sequential", "current_workers"):
        entry = modes.get(mode)
        if not entry:
            problems.append(f"нет режима {mode!r}")
            continue
        for field in (
            "duration_seconds",
            "process_cpu_seconds",
            "peak_rss_bytes",
            "workers",
        ):
            if entry.get(field) is None:
                problems.append(f"{mode}: нет поля {field!r}")
        if entry.get("duration_seconds") is not None and (
            entry["duration_seconds"] <= 0
        ):
            problems.append(f"{mode}: duration_seconds должен быть > 0")
    heartbeat = report.get("heartbeat", {})
    if heartbeat.get("status") == "measured":
        for field in ("p95_ms", "max_ms", "samples"):
            if heartbeat.get(field) is None:
                problems.append(f"heartbeat: нет поля {field!r}")
    io_conditions = report.get("io_conditions", {})
    if "warm-cache" not in str(io_conditions.get("measured", "")):
        problems.append("io_conditions.measured должно фиксировать warm-cache")
    if "cold" not in str(io_conditions.get("unmeasured", "")).lower():
        problems.append(
            "io_conditions.unmeasured должно фиксировать неизмеренный cold-storage I/O"
        )
    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="сохранить отчёт в tests/baseline/"
    )
    parser.add_argument(
        "--check", action="store_true", help="проверить пересобранный отчёт"
    )
    args = parser.parse_args(argv)

    report = build_report()

    if args.write:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Базовый отчёт сохранён: {BASELINE_PATH}")
        return 0

    if args.check:
        problems = _check_report(report)
        if problems:
            print("Проблемы в базовом отчёте:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        heartbeat = report["heartbeat"]
        if heartbeat.get("status") == "measured":
            print(
                "Heartbeat: p95={:.1f} ms (бюджет {} ms), max={:.1f} ms "
                "(бюджет {} ms)".format(
                    heartbeat["p95_ms"],
                    HEARTBEAT_P95_BUDGET_MS,
                    heartbeat["max_ms"],
                    HEARTBEAT_MAX_BUDGET_MS,
                )
            )
        print(
            "Базовый отчёт корректен: warm-cache измерен, cold-storage I/O отмечен как неизмеренный."
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
