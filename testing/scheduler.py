"""In-memory batched scheduler shared by report and baseline runners."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from analysis.sample_context import SampleContextCache
from .cache import PredictionCache
from .metrics import compute_segmentation_metrics
from .pipeline_observer import (
    PHASE_ALGORITHM,
    PHASE_DATASET_SCAN,
    PHASE_READ_DECODE,
    PipelineObserver,
)
from .registry import get_algorithm, list_algorithms
from .telemetry import TelemetryCollector

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleRef:
    name: str
    variant: str
    image_path: str
    mask_path: str
    is_cropped: bool
    image: np.ndarray | None = None
    mask: np.ndarray | None = None


@dataclass
class LoadedSample:
    ref: SampleRef
    image: np.ndarray
    mask: np.ndarray
    # Время завершения загрузки/декодирования (perf_counter). Нужно для
    # раздельного измерения input_wait (задача 2.2).
    load_complete_time: float | None = None


@dataclass
class PipelineDiagnostics:
    """Ошибки и счётчики прогона, доступные независимо от telemetry."""

    detail_limit: int = 3
    loaded_pairs: int = 0
    completed_tasks: int = 0
    load_failure_count: int = 0
    algorithm_failure_count: int = 0
    load_failure_details: list[str] = field(default_factory=list)
    algorithm_failure_details: list[str] = field(default_factory=list)

    def record_load_failure(self, error: Exception) -> None:
        self.load_failure_count += 1
        if len(self.load_failure_details) < self.detail_limit:
            self.load_failure_details.append(str(error))

    def record_algorithm_failure(
        self,
        name: str,
        sample_name: str,
        variant: str,
        image_path: str,
        error: Exception,
    ) -> None:
        self.algorithm_failure_count += 1
        if len(self.algorithm_failure_details) < self.detail_limit:
            self.algorithm_failure_details.append(
                f"{name}/{sample_name}/{variant} ({image_path}): {error}"
            )


def _sample_refs(dataset: Any, *, use_cropped: bool = True) -> list[SampleRef]:
    refs: list[SampleRef] = []
    if hasattr(dataset, "samples") and dataset.samples:
        samples = dataset.samples
    else:
        samples = list(dataset)
    for sample in samples:
        if hasattr(sample, "source_path"):
            refs.append(
                SampleRef(
                    sample.name,
                    "source",
                    sample.source_path,
                    sample.source_mask_path,
                    False,
                    getattr(sample, "source_image", None),
                    getattr(sample, "source_mask", None),
                )
            )
            if use_cropped and sample.cropped_path and sample.cropped_mask_path:
                refs.append(
                    SampleRef(
                        sample.name,
                        "cropped",
                        sample.cropped_path,
                        sample.cropped_mask_path,
                        True,
                        getattr(sample, "cropped_image", None),
                        getattr(sample, "cropped_mask", None),
                    )
                )
        else:
            if use_cropped or sample.variant != "cropped":
                refs.append(
                    SampleRef(
                        sample.name,
                        sample.variant,
                        str(sample.image_path),
                        str(sample.mask_path),
                        sample.variant == "cropped",
                    )
                )
    return refs


def _load_ref(
    ref: SampleRef,
    telemetry: TelemetryCollector | None = None,
) -> LoadedSample:
    image = ref.image
    mask = ref.mask
    if image is None:
        image = _decode_file(
            ref.image_path,
            cv2.IMREAD_COLOR,
            "изображение",
            telemetry=telemetry,
        )
    if mask is None:
        mask = _decode_file(
            ref.mask_path,
            cv2.IMREAD_GRAYSCALE,
            "эталонную маску",
            telemetry=telemetry,
        )
    return LoadedSample(ref, image, mask, load_complete_time=time.perf_counter())


def _decode_file(
    path: str,
    flags: int,
    description: str,
    telemetry: TelemetryCollector | None = None,
) -> np.ndarray:
    """Декодирует файл через байты, чтобы поддерживать Unicode-пути на Windows.

    Телеметрия (задача 2.1) раздельно учитывает логическое чтение файла
    (`file_read`) и декодирование (`image_decode`), включая число
    логически прочитанных байтов. Физические I/O-счётчики ОС — отдельная
    capability (ноль может означать page cache, а не отсутствие чтения).
    """
    file_path = Path(path)
    read_started = time.perf_counter()
    try:
        encoded = file_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"Не удалось прочитать {description} «{file_path}»: {exc}"
        ) from exc
    read_duration = time.perf_counter() - read_started
    if telemetry:
        telemetry.record_stage("file_read", read_duration)
        telemetry.record_logical_io("file_read", len(encoded))

    if not encoded:
        raise RuntimeError(f"Файл {description} пуст: «{file_path}»")

    decode_started = time.perf_counter()
    try:
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), flags)
    except cv2.error as exc:
        raise RuntimeError(
            f"Не удалось декодировать {description} «{file_path}»: {exc}"
        ) from exc
    if decoded is None:
        raise RuntimeError(
            f"Не удалось декодировать {description} «{file_path}»: "
            "формат не поддерживается или файл повреждён"
        )
    decode_duration = time.perf_counter() - decode_started
    if telemetry:
        telemetry.record_stage("image_decode", decode_duration)
        telemetry.record_logical_io(
            "image_decode", decoded.nbytes, decoded_bytes=decoded.nbytes
        )
    return decoded


# Доля бюджета памяти, резервируемая под временные буферы конкурентных
# алгоритмов (морфология, кеши OpenCV/ONNX Runtime). Декодированные входы
# занимают не более остатка — «консервативный headroom» (задача 2.5).
ALGORITHM_HEADROOM_RATIO = 0.25


def _decoded_input_budget(memory_budget: int | None) -> int | None:
    """Бюджет памяти под декодированные входы (image + mask) одного batch.

    Из `memory_budget` вычитается консервативный резерв под временные
    буферы конкурентных алгоритмов. None — без ограничения.
    """
    if memory_budget is None:
        return None
    if memory_budget <= 0:
        raise ValueError("memory_budget must be positive")
    reserve = int(memory_budget * ALGORITHM_HEADROOM_RATIO)
    return max(1, memory_budget - reserve)


def iter_batches(
    dataset: Any,
    *,
    batch_size: int = 8,
    memory_budget: int | None = None,
    telemetry: TelemetryCollector | None = None,
    use_cropped: bool = True,
    diagnostics: PipelineDiagnostics | None = None,
    observer: "PipelineObserver | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> Iterable[list[LoadedSample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Суммарный объём декодированных входов ограничен бюджетом с резервом
    # под конкурентные операции (задача 2.5).
    input_budget = _decoded_input_budget(memory_budget)
    batch: list[LoadedSample] = []
    batch_bytes = 0
    for ref in _sample_refs(dataset, use_cropped=use_cropped):
        # Кооперативная отмена (задача 7.4): новые пары не загружаются.
        if cancel_event is not None and cancel_event.is_set():
            break
        started = time.perf_counter()
        try:
            loaded = _load_ref(ref, telemetry=telemetry)
        except Exception as exc:
            if diagnostics:
                diagnostics.record_load_failure(exc)
            if telemetry:
                telemetry.record_error(str(exc))
                telemetry.record_task(failed=1)
            if observer is not None:
                # Контекстная ошибка отдельной пары (задача 7.1).
                observer.on_task_error(
                    f"{ref.name}/{ref.variant}", f"{type(exc).__name__}: {exc}"
                )
            log.error("Unable to load %s/%s: %s", ref.name, ref.variant, exc)
            continue
        load_time = time.perf_counter() - started
        if telemetry:
            telemetry.record_stage("load", load_time)
        size = loaded.image.nbytes + loaded.mask.nbytes
        if memory_budget is not None and size > memory_budget and telemetry:
            telemetry.event(
                "memory_budget_exceeded",
                count=1,
                bytes=size,
                memory_budget=memory_budget,
            )

        # Изолированная обработка одной oversized-пары (задача 2.5): никаких
        # других декодированных пар рядом — активных и заранее загруженных.
        if input_budget is not None and size > input_budget:
            if batch:
                if telemetry:
                    telemetry.event("batch_loaded", count=len(batch), bytes=batch_bytes)
                yield batch
                batch = []
                batch_bytes = 0
            if telemetry:
                telemetry.event(
                    "oversized_pair_isolated",
                    count=1,
                    bytes=size,
                    input_budget=input_budget,
                )
            yield [loaded]
            continue

        if batch and (
            len(batch) >= batch_size
            or (input_budget is not None and batch_bytes + size > input_budget)
        ):
            if telemetry:
                telemetry.event("batch_loaded", count=len(batch), bytes=batch_bytes)
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(loaded)
        batch_bytes += size
    if batch:
        if telemetry:
            telemetry.event("batch_loaded", count=len(batch), bytes=batch_bytes)
        yield batch


class _AlgorithmRuntime:
    def __init__(self, telemetry: TelemetryCollector | None = None) -> None:
        self._local = threading.local()
        self._locks: dict[str, threading.Lock] = {}
        self._lock_guard = threading.Lock()
        self.telemetry = telemetry
        # Инвариантный контекст на sample/variant (задача 3.3): переиспользует
        # геометрию чашки и предобработку между совместимыми запусками.
        self.sample_contexts = SampleContextCache(maxsize=32)

    @staticmethod
    def _is_instance(name: str) -> bool:
        from . import registry

        return name in registry._INSTANCES

    def _get_instance(self, name: str):
        if name not in self._locks:
            with self._lock_guard:
                self._locks.setdefault(name, threading.Lock())
        return get_algorithm(name)

    @staticmethod
    def _accepts_context(algorithm) -> bool:
        """True, если `detect()` алгоритма принимает необязательный `context`."""
        import inspect

        try:
            signature = inspect.signature(algorithm.detect)
        except (TypeError, ValueError):
            return False
        return "context" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def cache_params(name: str) -> dict[str, object]:
        algo = get_algorithm(name)
        params: dict[str, object] = {"type": type(algo).__qualname__}
        for key in ("model_path", "img_size", "threshold"):
            if hasattr(algo, key):
                params[key] = getattr(algo, key)
        return params

    def detect(
        self,
        name: str,
        image: np.ndarray,
        is_cropped: bool,
        sample_key: tuple | None = None,
    ) -> np.ndarray:
        if self._is_instance(name):
            instance = self._get_instance(name)
            lock = self._locks[name]
            self.telemetry.record_serial_call() if self.telemetry else None
            with lock:
                return self._detect_with_context(
                    instance, name, image, is_cropped, sample_key
                )
        instances = getattr(self._local, "instances", None)
        if instances is None:
            instances = self._local.instances = {}
        if name not in instances:
            instances[name] = get_algorithm(name)
        return self._detect_with_context(
            instances[name], name, image, is_cropped, sample_key
        )

    def _detect_with_context(
        self,
        algorithm,
        name: str,
        image: np.ndarray,
        is_cropped: bool,
        sample_key: tuple | None = None,
    ):
        if sample_key is None or not self._accepts_context(algorithm):
            return algorithm.detect(image, is_cropped=is_cropped)
        # Ключ контекста: sample + variant. Контекст общий для задач
        # разных алгоритмов одного снимка; SampleContext сериализует
        # обращения через собственный RLock (write-once при совпадении ключа).
        context = self.sample_contexts.get_or_create(sample_key, is_cropped)
        return algorithm.detect(image, is_cropped=is_cropped, context=context)


def _affinity_cpu_count() -> int | None:
    """Число ядер в process-affinity (None, если операция не поддерживается)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        import psutil

        affinity = psutil.Process().cpu_affinity()
        return len(affinity) if affinity else None
    except (ImportError, AttributeError, OSError):
        return None


def _quota_cpu_cores() -> float | None:
    """Эффективная ёмкость CPU из cgroup-квоты.

    Возвращает None, если квота не задана (unlimited) или недоступна —
    неизвестное не подменяется нулём (задачи 2.3/2.4).
    """
    for version, path in (
        (2, Path("/sys/fs/cgroup/cpu.max")),
        (1, Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")),
    ):
        try:
            if version == 2:
                raw = path.read_text(encoding="utf-8").strip()
                quota, period = raw.split()
                if quota == "max":
                    return None
                return int(quota) / max(1, int(period))
            quota_us = int(path.read_text(encoding="utf-8").strip())
            if quota_us <= 0:
                return None
            period_us = int(
                (path.parent / "cpu.cfs_period_us").read_text(encoding="utf-8").strip()
            )
            return quota_us / max(1, period_us)
        except (OSError, ValueError):
            continue
    return None


def _host_cpu_count() -> int | None:
    """Физические ядра хоста (fallback: логические ядра, затем None).

    Физические ядра консервативнее логических (hyperthreading не даёт
    линейного ускорения CPU-bound задач).
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return physical
    except (ImportError, AttributeError):
        pass
    return os.cpu_count()


def _visible_cpu_capacity() -> int:
    """Ёмкость CPU, видимая процессу (задача 2.4).

    Консервативно берёт МИНИМУМ из известных ограничений:
    process-affinity (реально доступные ядра), cgroup-квота (доля
    CPU-времени) и физические ядра хоста. Так ограничения контейнеров
    и CPU-affinity соблюдаются, а на обычных машинах поведение не хуже
    прежнего (физические ядра). Консервативный fallback: если неизвестно
    ничего — одна единица параллелизма.
    """
    signals = [
        value
        for value in (
            _affinity_cpu_count(),
            _quota_cpu_cores(),
            _host_cpu_count(),
        )
        if value is not None and value >= 1
    ]
    if not signals:
        # Консервативный fallback: одна единица параллелизма на неизвестной системе.
        return 1
    return max(1, int(min(signals)))


def _effective_workers(
    requested: int | None,
    task_count: int,
    memory_budget: int | None,
    batch_bytes: int,
) -> int:
    if task_count <= 0:
        return 1
    if requested is not None and requested <= 0:
        raise ValueError("workers must be positive")
    # Авто-ёмкость из affinity/quota процесса (задача 2.4).
    visible = _visible_cpu_capacity()
    try:
        import psutil

        available_memory = psutil.virtual_memory().available
        if memory_budget is not None:
            visible = min(visible, max(1, available_memory // max(memory_budget, 1)))
    except (ImportError, OSError, AttributeError):
        pass
    # `requested` (--workers) — верхняя граница, а не гарантия.
    limit = requested if requested is not None else visible
    return max(1, min(limit, visible, task_count))


def execute_pipeline(
    dataset: Any,
    algorithm_names: list[str] | None = None,
    *,
    workers: int | None = None,
    batch_size: int = 8,
    memory_budget: int | None = None,
    use_cache: bool = False,
    telemetry: TelemetryCollector | None = None,
    use_cropped: bool = True,
    diagnostics: PipelineDiagnostics | None = None,
    observer: "PipelineObserver | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    names = list_algorithms() if algorithm_names is None else list(algorithm_names)
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        name: {} for name in names
    }
    if not names:
        return {}
    telemetry = telemetry or TelemetryCollector()
    telemetry.start()
    # Фиксируем доступность CPU и эффективные лимиты параллелизма (задача 2.3).
    from analysis.host_profile import _cpu_affinity, _cpu_quota

    try:
        import psutil

        logical_count = os.cpu_count()
        physical_count = psutil.cpu_count(logical=False)
    except (ImportError, AttributeError):
        logical_count = os.cpu_count()
        physical_count = None
    telemetry.record_cpu_availability(
        logical_count=logical_count,
        physical_count=physical_count,
        affinity=_cpu_affinity(),
        quota=_cpu_quota(),
    )
    # Единая ресурсная политика (задача 2.6): внешние algorithm workers и
    # нативные пулы OpenCV/ONNX RT считаются одним бюджетом CPU.
    from .resource_policy import apply_native_limits, resolve_resource_policy

    policy = resolve_resource_policy(workers=workers)
    native_threads = policy.native_threads
    data_root = getattr(dataset, "root", None)
    prediction_cache = PredictionCache(data_root) if use_cache and data_root else None
    runtime = _AlgorithmRuntime(telemetry)
    cache_params = {name: runtime.cache_params(name) for name in names}
    effective_workers = policy.algorithm_workers
    counted_objects: set[str] = set()
    with apply_native_limits(policy):
        results = _execute_batches(
            dataset,
            names,
            results,
            runtime,
            cache_params,
            prediction_cache,
            telemetry,
            diagnostics,
            effective_workers,
            workers,
            batch_size,
            memory_budget,
            use_cropped,
            counted_objects,
            observer,
            cancel_event,
        )
    telemetry.record_worker_limits(
        outer_workers=effective_workers,
        native_threads=native_threads,
        source=policy.source,
    )
    telemetry.finish(workers=effective_workers or 1, batch_size=batch_size)
    return {
        name: {
            sample_name: {
                variant: sample_results[variant] for variant in sorted(sample_results)
            }
            for sample_name, sample_results in sorted(results[name].items())
        }
        for name in names
    }


def _execute_batches(
    dataset: Any,
    names: list[str],
    results: dict,
    runtime: "_AlgorithmRuntime",
    cache_params: dict,
    prediction_cache,
    telemetry: TelemetryCollector,
    diagnostics: PipelineDiagnostics | None,
    effective_workers: int,
    workers: int | None,
    batch_size: int,
    memory_budget: int | None,
    use_cropped: bool,
    counted_objects: set[str],
    observer: "PipelineObserver | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> dict:
    total_tasks: int | None = None
    completed_tasks = 0
    # Перекрытие decode/compute (задача 2.7): следующий batch декодируется
    # в producer-потоке, пока текущий обрабатывается. Backpressure по очереди
    # (maxsize=1) и по памяти (input-budget с резервом, задача 2.5).
    from .pipeline_overlap import PrefetchLoader

    def _batch_bytes(batch: list[LoadedSample]) -> int:
        return sum(item.image.nbytes + item.mask.nbytes for item in batch)

    batches = iter_batches(
        dataset,
        batch_size=batch_size,
        memory_budget=memory_budget,
        telemetry=telemetry,
        use_cropped=use_cropped,
        diagnostics=diagnostics,
        observer=observer,
        cancel_event=cancel_event,
    )
    loader = PrefetchLoader(
        batches,
        maxsize=1,
        max_bytes=_decoded_input_budget(memory_budget),
        size_fn=_batch_bytes,
        name="decode",
    )
    if observer is not None:
        # Фазы dataset_scan/read_decode эмитируются из consumer-потока ДО
        # старта producer — гарантированный порядок сигналов (задача 7.1).
        observer.on_phase(PHASE_DATASET_SCAN)
        observer.on_phase(PHASE_READ_DECODE)
    with loader:
        results = _run_batch_loop(
            loader,
            dataset,
            names,
            results,
            runtime,
            cache_params,
            prediction_cache,
            telemetry,
            diagnostics,
            effective_workers,
            workers,
            batch_size,
            memory_budget,
            use_cropped,
            counted_objects,
            observer,
            total_tasks,
            completed_tasks,
            cancel_event,
        )
    # Input-starvation (задача 2.7): доля worker-slot-seconds с пустой
    # очередью входов. Startup/финишные ожидания исключены в PrefetchLoader.
    telemetry.record_input_starvation(
        starvation_seconds=loader.starvation_time,
        active_seconds=loader.active_time,
        ratio=loader.input_starvation_ratio,
    )
    return results


def _run_batch_loop(
    loader,
    dataset: Any,
    names: list[str],
    results: dict,
    runtime: "_AlgorithmRuntime",
    cache_params: dict,
    prediction_cache,
    telemetry: TelemetryCollector,
    diagnostics: PipelineDiagnostics | None,
    effective_workers: int,
    workers: int | None,
    batch_size: int,
    memory_budget: int | None,
    use_cropped: bool,
    counted_objects: set[str],
    observer: "PipelineObserver | None",
    total_tasks: int | None,
    completed_tasks: int,
    cancel_event: "threading.Event | None" = None,
) -> dict:
    for batch_number, batch in enumerate(loader, start=1):
        if diagnostics:
            diagnostics.loaded_pairs += len(batch)
        task_count = len(batch) * len(names)
        batch_object_names = {item.ref.name for item in batch}
        telemetry.record_work(
            objects=len(batch_object_names - counted_objects),
            variants=len(batch),
            algorithm_calls=task_count,
        )
        counted_objects.update(batch_object_names)
        if observer is not None and total_tasks is None:
            # Общее число задач: снимки × варианты × алгоритмы (первый batch).
            total_samples = len(_sample_refs(dataset, use_cropped=use_cropped))
            total_tasks = total_samples * len(names)
            observer.on_phase(PHASE_ALGORITHM, total=total_tasks)
        batch_bytes = sum(item.image.nbytes + item.mask.nbytes for item in batch)
        effective_workers = _effective_workers(
            effective_workers, task_count, memory_budget, batch_bytes
        )
        telemetry.event(
            "batch_started",
            batch=batch_number,
            count=len(batch),
            bytes=batch_bytes,
            workers=effective_workers,
        )
        future_map = {}
        batch_completed = 0
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            for item in batch:
                if cancel_event is not None and cancel_event.is_set():
                    # Кооперативная отмена (7.4): новые задачи не ставятся,
                    # уже запущенные native-вызовы дорабатывают сами.
                    break
                for name in names:
                    # Собственная submit-время каждой задачи (задача 2.2):
                    # queue_wait измеряется от неё, а не от старта batch.
                    task_submitted_at = time.perf_counter()
                    future = executor.submit(
                        _run_task,
                        runtime,
                        name,
                        item,
                        prediction_cache,
                        telemetry,
                        task_submitted_at,
                        cache_params[name],
                    )
                    future_map[future] = (
                        name,
                        item.ref.name,
                        item.ref.variant,
                        item.ref.image_path,
                    )
                    telemetry.record_task(submitted=1, active=len(future_map))
            for future in as_completed(future_map):
                name, sample_name, variant, image_path = future_map[future]
                try:
                    metrics = future.result()
                except Exception as exc:
                    log.error(
                        "Task failed: %s/%s/%s: %s", name, sample_name, variant, exc
                    )
                    telemetry.record_task(failed=1)
                    telemetry.record_error(f"{name}/{sample_name}/{variant}: {exc}")
                    if diagnostics:
                        diagnostics.record_algorithm_failure(
                            name, sample_name, variant, image_path, exc
                        )
                    if observer is not None:
                        # Контекстная ошибка отдельной задачи (задача 7.1):
                        # успешные задачи остаются доступны.
                        observer.on_task_error(
                            f"{name}/{sample_name}/{variant}",
                            f"{type(exc).__name__}: {exc}",
                        )
                    continue
                results[name].setdefault(sample_name, {})[variant] = metrics
                telemetry.record_task(completed=1)
                if diagnostics:
                    diagnostics.completed_tasks += 1
                batch_completed += 1
                completed_tasks += 1
                if observer is not None:
                    observer.on_progress(completed_tasks, total_tasks or 0)
                telemetry.record_task(
                    active=max(0, len(future_map) - batch_completed),
                    queue_depth=max(0, len(future_map) - batch_completed),
                )
                telemetry.maybe_snapshot(
                    batch=batch_number,
                    queue_depth=max(0, len(future_map) - batch_completed),
                    workers=effective_workers,
                )
        telemetry.event(
            "batch_completed",
            batch=batch_number,
            count=len(batch),
            bytes=batch_bytes,
            workers=effective_workers,
        )
        # Освобождение бюджета памяти: следующий batch может декодироваться.
        loader.task_done(batch)
        if effective_workers > 1 and _should_reduce_workers(
            telemetry, memory_budget, batch_bytes
        ):
            effective_workers = max(1, effective_workers // 2)
            telemetry.event(
                "workers_reduced", workers=effective_workers, reason="resource_pressure"
            )
    return results


def _should_reduce_workers(
    telemetry: TelemetryCollector,
    memory_budget: int | None,
    batch_bytes: int,
) -> bool:
    if memory_budget is not None and batch_bytes > memory_budget:
        return True
    queue = telemetry._latencies.get("queue_wait", [])
    detect = telemetry._latencies.get("detect", [])
    return (
        len(queue) >= 2 and len(detect) >= 2 and sum(queue[-2:]) > sum(detect[-2:]) * 2
    )


def _run_task(
    runtime: _AlgorithmRuntime,
    name: str,
    item: LoadedSample,
    prediction_cache: PredictionCache | None,
    telemetry: TelemetryCollector,
    submitted_at: float,
    cache_params: dict[str, object],
) -> dict[str, float]:
    task_started = time.perf_counter()
    # Очередь executor: от собственной submit-времени задачи (задача 2.2).
    telemetry.record_stage("queue_wait", task_started - submitted_at)
    # Ожидание подготовленных входных данных: от завершения загрузки/декодирования
    # до старта вычислений. Отдельно от executor queue_wait — чтобы различать
    # input starvation (данные ещё не готовы) и перегрузку очереди (задачи
    # ждут executor-слот).
    if item.load_complete_time is not None:
        telemetry.record_stage("input_wait", task_started - item.load_complete_time)
    cache_started = time.perf_counter()
    task_cache_params = {**cache_params, "is_cropped": item.ref.is_cropped}
    prediction = (
        prediction_cache.get_array(item.image, name, task_cache_params)
        if prediction_cache
        else None
    )
    telemetry.record_stage("cache", time.perf_counter() - cache_started)
    telemetry.record_cache(prediction is not None)
    if prediction is None:
        detect_started = time.perf_counter()
        prediction = runtime.detect(
            name,
            item.image,
            item.ref.is_cropped,
            sample_key=(item.ref.name, item.ref.variant),
        )
        telemetry.record_stage("detect", time.perf_counter() - detect_started)
        if prediction_cache:
            prediction_cache.put_array(item.image, name, prediction, task_cache_params)
    metrics_started = time.perf_counter()
    metrics = compute_segmentation_metrics(prediction, item.mask)
    telemetry.record_stage("metrics", time.perf_counter() - metrics_started)
    telemetry.record_task_event(
        algorithm=name,
        sample=item.ref.name,
        variant=item.ref.variant,
        duration=time.perf_counter() - task_started,
    )
    return metrics
