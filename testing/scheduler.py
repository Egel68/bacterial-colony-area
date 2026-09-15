"""In-memory batched scheduler shared by report and baseline runners."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from .cache import PredictionCache
from .metrics import compute_segmentation_metrics
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


def _load_ref(ref: SampleRef) -> LoadedSample:
    image = ref.image
    mask = ref.mask
    if image is None:
        image = cv2.imread(ref.image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {ref.image_path}")
    if mask is None:
        mask = cv2.imread(ref.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {ref.mask_path}")
    return LoadedSample(ref, image, mask)


def iter_batches(
    dataset: Any,
    *,
    batch_size: int = 8,
    memory_budget: int | None = None,
    telemetry: TelemetryCollector | None = None,
    use_cropped: bool = True,
) -> Iterable[list[LoadedSample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[LoadedSample] = []
    batch_bytes = 0
    for ref in _sample_refs(dataset, use_cropped=use_cropped):
        started = time.perf_counter()
        try:
            loaded = _load_ref(ref)
        except Exception as exc:
            if telemetry:
                telemetry.record_error(str(exc))
                telemetry.record_task(load_failed=1)
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
        if batch and (
            len(batch) >= batch_size
            or (memory_budget is not None and batch_bytes + size > memory_budget)
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
    def cache_params(name: str) -> dict[str, object]:
        algo = get_algorithm(name)
        params: dict[str, object] = {"type": type(algo).__qualname__}
        for key in ("model_path", "img_size", "threshold"):
            if hasattr(algo, key):
                params[key] = getattr(algo, key)
        return params

    def detect(self, name: str, image: np.ndarray, is_cropped: bool) -> np.ndarray:
        if self._is_instance(name):
            instance = self._get_instance(name)
            lock = self._locks[name]
            self.telemetry.record_serial_call() if self.telemetry else None
            with lock:
                return instance.detect(image, is_cropped=is_cropped)
        instances = getattr(self._local, "instances", None)
        if instances is None:
            instances = self._local.instances = {}
        if name not in instances:
            instances[name] = get_algorithm(name)
        return instances[name].detect(image, is_cropped=is_cropped)


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
    physical = os.cpu_count() or 1
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or physical
        available = psutil.virtual_memory().available
        if memory_budget is not None:
            physical = min(physical, max(1, available // max(memory_budget, 1)))
    except (ImportError, OSError, AttributeError):
        pass
    limit = requested if requested is not None else physical
    return max(1, min(limit, physical, task_count))


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
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    names = list_algorithms() if algorithm_names is None else list(algorithm_names)
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        name: {} for name in names
    }
    if not names:
        return {}
    telemetry = telemetry or TelemetryCollector()
    telemetry.start()
    data_root = getattr(dataset, "root", None)
    prediction_cache = PredictionCache(data_root) if use_cache and data_root else None
    runtime = _AlgorithmRuntime(telemetry)
    cache_params = {name: runtime.cache_params(name) for name in names}
    effective_workers = workers
    counted_objects: set[str] = set()
    for batch_number, batch in enumerate(
        iter_batches(
            dataset,
            batch_size=batch_size,
            memory_budget=memory_budget,
            telemetry=telemetry,
            use_cropped=use_cropped,
        ),
        start=1,
    ):
        task_count = len(batch) * len(names)
        batch_object_names = {item.ref.name for item in batch}
        telemetry.record_work(
            objects=len(batch_object_names - counted_objects),
            variants=len(batch),
            algorithm_calls=task_count,
        )
        counted_objects.update(batch_object_names)
        batch_bytes = sum(item.image.nbytes + item.mask.nbytes for item in batch)
        if effective_workers is None:
            effective_workers = _effective_workers(
                None, task_count, memory_budget, batch_bytes
            )
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
        submitted_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            for item in batch:
                for name in names:
                    future = executor.submit(
                        _run_task,
                        runtime,
                        name,
                        item,
                        prediction_cache,
                        telemetry,
                        submitted_at,
                        cache_params[name],
                    )
                    future_map[future] = (name, item.ref.name, item.ref.variant)
                    telemetry.record_task(submitted=1, active=len(future_map))
            for future in as_completed(future_map):
                name, sample_name, variant = future_map[future]
                try:
                    metrics = future.result()
                except Exception as exc:
                    log.error(
                        "Task failed: %s/%s/%s: %s", name, sample_name, variant, exc
                    )
                    telemetry.record_task(failed=1)
                    telemetry.record_error(f"{name}/{sample_name}/{variant}: {exc}")
                    continue
                results[name].setdefault(sample_name, {})[variant] = metrics
                telemetry.record_task(completed=1)
                batch_completed += 1
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
        if effective_workers > 1 and _should_reduce_workers(
            telemetry, memory_budget, batch_bytes
        ):
            effective_workers = max(1, effective_workers // 2)
            telemetry.event(
                "workers_reduced", workers=effective_workers, reason="resource_pressure"
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
    telemetry.record_stage("queue_wait", task_started - submitted_at)
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
        prediction = runtime.detect(name, item.image, item.ref.is_cropped)
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
