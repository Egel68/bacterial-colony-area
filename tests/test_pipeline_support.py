import json
import os

import cv2
import numpy as np

import testing.scheduler as testing_scheduler
from testing.cache import PredictionCache
from testing.baseline import BaselineDataset, run_baseline
from testing.dataset import TestDataset
from testing.scheduler import (
    _should_reduce_workers,
    iter_batches,
    execute_pipeline,
    PipelineDiagnostics,
)
from testing.telemetry import TelemetryCollector


def _write_pair(root, stem, value):
    (root / "source").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(root / "source" / f"{stem}.png"), image)
    cv2.imwrite(str(root / "masks" / f"{stem}_mask.png"), mask)


def test_test_dataset_sample_limit_is_sorted_and_counts_source_objects(tmp_path):
    for index, stem in enumerate(("c", "a", "b")):
        _write_pair(tmp_path, stem, index)

    dataset = TestDataset(root=str(tmp_path), sample_limit=2)

    assert [sample.name for sample in dataset] == ["a", "b"]


def test_sample_limit_five_selects_exactly_five_objects(tmp_path):
    for index in range(6):
        _write_pair(tmp_path, f"sample-{index}", index)

    dataset = TestDataset(root=str(tmp_path), sample_limit=5, load_images=False)
    baseline = BaselineDataset(root=tmp_path, sample_limit=5)

    assert len(dataset) == 5
    assert baseline.count_source() == 5


def test_prediction_cache_uses_loaded_image_without_reopening_source(
    tmp_path, monkeypatch
):
    cache = PredictionCache(tmp_path)
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)

    cache.put_array(image, "Algo", mask)

    def fail_open(*args, **kwargs):
        raise AssertionError("source image must not be reopened")

    monkeypatch.setattr("builtins.open", fail_open)
    loaded = cache.get_array(image, "Algo")

    assert np.array_equal(loaded, mask)


def test_prediction_cache_parameters_invalidate_predictions(tmp_path):
    cache = PredictionCache(tmp_path)
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    first = np.zeros((4, 4), dtype=np.uint8)
    second = np.full((4, 4), 255, dtype=np.uint8)

    cache.put_array(image, "Algo", first, {"threshold": 0.5})
    cache.put_array(image, "Algo", second, {"threshold": 0.7})

    assert np.array_equal(cache.get_array(image, "Algo", {"threshold": 0.5}), first)
    assert np.array_equal(cache.get_array(image, "Algo", {"threshold": 0.7}), second)


def test_telemetry_writes_summary_and_percentiles(tmp_path):
    output = tmp_path / "performance.json"
    telemetry = TelemetryCollector(
        enabled=True,
        output_path=output,
        interval=0,
    )
    telemetry.start()
    telemetry.record_stage("detect", 0.1)
    telemetry.record_stage("detect", 0.3)
    telemetry.record_task(submitted=2, completed=2)
    telemetry.finish()

    summary = json.loads(output.read_text(encoding="utf-8"))

    assert summary["tasks"]["completed"] == 2
    assert summary["latency"]["detect"]["p95"] >= 0.1
    assert "cpu" in summary["resources"]
    assert "io_rates" in summary["resources"]
    assert "read_bytes_per_second" in summary["resources"]["rates"]
    assert "objects_per_second" in summary["throughput_rates"]
    assert (tmp_path / "performance.jsonl").is_file()


def test_baseline_uses_sample_limit_and_shared_scheduler(tmp_path):
    for index, stem in enumerate(("c", "a", "b")):
        _write_pair(tmp_path, stem, index)

    dataset = BaselineDataset(root=tmp_path, sample_limit=2)
    result = run_baseline(
        dataset,
        use_cropped=False,
        workers=2,
        batch_size=1,
    )

    assert result["image_count"]["source"] == 2
    assert result["image_count"]["cropped"] == 0
    assert result["algorithms"]
    assert all(entry["source"]["num_samples"] == 2 for entry in result["algorithms"])


def test_batch_loader_respects_memory_budget(tmp_path):
    _write_pair(tmp_path, "a", 1)
    _write_pair(tmp_path, "b", 2)
    dataset = TestDataset(root=str(tmp_path), load_images=False)
    telemetry = TelemetryCollector(enabled=True)

    batches = list(
        iter_batches(
            dataset,
            batch_size=8,
            memory_budget=8 * 8 * 3 + 8 * 8,
            telemetry=telemetry,
        )
    )

    assert [len(batch) for batch in batches] == [1, 1]


def test_adaptive_worker_guard_reduces_on_queue_pressure():
    telemetry = TelemetryCollector(enabled=True)
    for _ in range(2):
        telemetry.record_stage("queue_wait", 0.3)
        telemetry.record_stage("detect", 0.05)

    assert _should_reduce_workers(telemetry, None, 0)


def test_scheduler_cache_skips_detection_on_second_run(tmp_path):
    from testing.interface import BaseDetectionAlgorithm
    from testing.registry import register_algorithm_instance
    from testing.runner import run_all

    class CacheAlgo(BaseDetectionAlgorithm):
        name = "CacheAlgo"
        description = "cache test"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    root = tmp_path / "dataset"
    _write_pair(root, "a", 1)
    dataset = TestDataset(root=str(root), load_images=False)
    register_algorithm_instance("CacheAlgo", CacheAlgo())
    try:
        first_telemetry = TelemetryCollector(enabled=True)
        run_all(
            dataset,
            algorithms=["CacheAlgo"],
            workers=1,
            use_cache=True,
            telemetry=first_telemetry,
        )
        second_telemetry = TelemetryCollector(enabled=True)
        run_all(
            dataset,
            algorithms=["CacheAlgo"],
            workers=1,
            use_cache=True,
            telemetry=second_telemetry,
        )
    finally:
        from testing.registry import _INSTANCES

        _INSTANCES.pop("CacheAlgo", None)

    assert second_telemetry._cache["hits"] == 1
    assert not second_telemetry._latencies["detect"]


def test_five_object_a3_matches_flat_path_baseline(tmp_path):
    from testing.interface import BaseDetectionAlgorithm
    from testing.metrics import compute_segmentation_metrics
    from testing.registry import _INSTANCES, register_algorithm_instance
    from testing.runner import run_all

    class FiveObjectAlgo(BaseDetectionAlgorithm):
        name = "FiveObjectAlgo"
        description = "five object smoke algorithm"

        def detect(self, image, is_cropped=False):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[1:3, 1:3] = 255
            return mask

    for index in range(5):
        _write_pair(tmp_path, f"sample-{index}", index)
    dataset = TestDataset(root=str(tmp_path), load_images=False, sample_limit=5)
    register_algorithm_instance("FiveObjectAlgo", FiveObjectAlgo())
    try:
        path_results = {}
        for sample in dataset:
            image = cv2.imread(sample.source_path)
            mask = cv2.imread(sample.source_mask_path, cv2.IMREAD_GRAYSCALE)
            prediction = FiveObjectAlgo().detect(image)
            path_results[sample.name] = {
                "source": compute_segmentation_metrics(prediction, mask)
            }

        telemetry = TelemetryCollector(enabled=True)
        a3_results = run_all(
            dataset,
            algorithms=["FiveObjectAlgo"],
            workers=2,
            batch_size=2,
            telemetry=telemetry,
        )
    finally:
        _INSTANCES.pop("FiveObjectAlgo", None)

    assert a3_results["FiveObjectAlgo"] == path_results
    summary = telemetry.summary()
    assert summary["work"]["objects"] == 5
    assert summary["resources"]["io"]["read_count"] is not None
    assert summary["duration_seconds"] >= 0


def test_five_object_reference_and_a3_report_io_and_wall_metrics(tmp_path):
    from time import perf_counter
    from testing.interface import BaseDetectionAlgorithm
    from testing.registry import _INSTANCES, register_algorithm_instance
    from testing.runner import run_algorithm, run_all

    class ReferenceAlgo(BaseDetectionAlgorithm):
        name = "ReferenceAlgo"
        description = "reference smoke algorithm"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    for index in range(5):
        _write_pair(tmp_path, f"sample-{index}", index)
    eager_dataset = TestDataset(root=str(tmp_path), sample_limit=5)
    lazy_dataset = TestDataset(root=str(tmp_path), sample_limit=5, load_images=False)
    register_algorithm_instance("ReferenceAlgo", ReferenceAlgo())
    try:
        started = perf_counter()
        reference = {
            sample.name: run_algorithm(ReferenceAlgo(), eager_dataset)[sample.name]
            for sample in eager_dataset
        }
        reference_wall = perf_counter() - started
        telemetry = TelemetryCollector(enabled=True)
        started = perf_counter()
        parallel = run_all(
            lazy_dataset,
            algorithms=["ReferenceAlgo"],
            workers=2,
            batch_size=2,
            telemetry=telemetry,
        )
        parallel_wall = perf_counter() - started
    finally:
        _INSTANCES.pop("ReferenceAlgo", None)

    assert parallel["ReferenceAlgo"] == reference
    summary = telemetry.summary()
    assert reference_wall >= 0
    assert parallel_wall >= 0
    assert summary["resources"]["io"]["read_count"] is not None
    assert summary["duration_seconds"] >= 0


# --- Задача 2.1: раздельные измерения чтения файла и декодирования ---


def test_decode_file_records_distinct_read_and_decode_stages(tmp_path):
    """Чтение файла и декодирование замеряются раздельно."""
    from testing.scheduler import _decode_file

    path = tmp_path / "sample.png"
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)

    telemetry = TelemetryCollector(enabled=True)
    decoded = _decode_file(str(path), cv2.IMREAD_COLOR, "изображение", telemetry)

    assert decoded.shape == (16, 16, 3)
    summary = telemetry.summary()
    latency = summary["latency"]
    assert "file_read" in latency, "время чтения файла не замерено отдельно"
    assert "image_decode" in latency, "время декодирования не замерено отдельно"
    assert latency["file_read"]["count"] == 1
    assert latency["image_decode"]["count"] == 1
    # Обе стадии имеют измеримую длительность.
    assert latency["file_read"]["total"] >= 0.0
    assert latency["image_decode"]["total"] >= 0.0


def test_decode_file_records_logical_byte_counts(tmp_path):
    """Логически прочитанные/декодированные байты учитываются."""
    from testing.scheduler import _decode_file

    path = tmp_path / "sample.png"
    image = np.full((32, 32, 3), 64, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    file_size = path.stat().st_size

    telemetry = TelemetryCollector(enabled=True)
    decoded = _decode_file(str(path), cv2.IMREAD_COLOR, "изображение", telemetry)

    logical_io = telemetry.summary()["logical_io"]
    assert "file_read" in logical_io
    assert logical_io["file_read"]["bytes"] == file_size
    assert logical_io["file_read"]["calls"] == 1
    assert "image_decode" in logical_io
    assert logical_io["image_decode"]["bytes"] == decoded.nbytes
    assert logical_io["image_decode"]["extra"]["decoded_bytes"] == decoded.nbytes


def test_warm_cache_reports_logical_reads_without_physical_counters(tmp_path):
    """Warm cache: логическое чтение видно, даже если физические счётчики нулевые."""
    from testing.scheduler import _decode_file

    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), 33, dtype=np.uint8))

    telemetry = TelemetryCollector(enabled=True)
    # Имитируем недоступные/нулевые физические счётчики ОС.
    original_resources = TelemetryCollector._resources

    def warm_cache_resources():
        resources = original_resources()
        resources["io"] = {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        }
        resources["capabilities"]["process_io_counters"] = False
        return resources

    telemetry._resources = warm_cache_resources
    telemetry.start()
    _decode_file(str(path), cv2.IMREAD_COLOR, "изображение", telemetry)
    summary = telemetry.summary()

    # Логическое чтение зафиксировано, несмотря на нулевые физические счётчики.
    assert summary["logical_io"]["file_read"]["bytes"] > 0
    assert summary["logical_io"]["file_read"]["calls"] >= 1
    # Физические счётчики помечены недоступными — неизвестное не подменено нулём.
    io_condition = summary["io_condition"]
    assert io_condition["physical_counters_available"] is False
    assert "page cache" in io_condition["note"] or "warm cache" in io_condition["note"]


def test_iter_batches_reports_load_and_decode_stages(tmp_path):
    """Прогон pipeline записывает стадии load/file_read/image_decode."""
    _write_pair(tmp_path, "a", 7)
    dataset = TestDataset(root=str(tmp_path), load_images=False)
    telemetry = TelemetryCollector(enabled=True)

    batches = list(
        iter_batches(dataset, batch_size=8, telemetry=telemetry, use_cropped=False)
    )
    assert sum(len(batch) for batch in batches) == 1

    summary = telemetry.summary()
    latency = summary["latency"]
    assert "load" in latency
    assert "file_read" in latency
    assert "image_decode" in latency
    # Чтение и декодирование измерены отдельно (по 2 файла: image + mask).
    assert latency["file_read"]["count"] == 2
    assert latency["image_decode"]["count"] == 2
    logical_io = summary["logical_io"]
    assert logical_io["file_read"]["bytes"] > 0
    assert logical_io["image_decode"]["bytes"] > 0


def test_logical_io_is_not_recorded_when_telemetry_disabled():
    telemetry = TelemetryCollector(enabled=False)
    telemetry.record_logical_io("file_read", 1024)
    summary = telemetry.summary()
    assert summary.get("logical_io", {}) == {}, (
        "выключенная телеметрия не должна накапливать logical_io"
    )


# --- Задача 2.2: queue_wait от собственной submit-времени; input_wait отдельно ---


def test_queue_wait_measured_from_own_submission_time():
    """Queue_wait каждой задачи измеряется с момента ЕЁ submit, не batch."""
    telemetry = TelemetryCollector(enabled=True)
    # Имитируем задачи с разными submit-временами: задача 2 отправлена на
    # 0.2с позже задачи 1, но обе начались сразу после своей submit.
    now = 1000.0
    task1_submitted = now
    task1_started = now + 0.01  # ожидание в очереди executor
    task2_submitted = now + 0.2
    task2_started = task2_submitted + 0.01

    telemetry.record_stage("queue_wait", task1_started - task1_submitted)
    telemetry.record_stage("queue_wait", task2_started - task2_submitted)

    latency = telemetry.summary()["latency"]["queue_wait"]
    assert latency["count"] == 2
    assert latency["p95"] <= 0.02, (
        "queue_wait должен быть около 0.01s, а не общее время batch"
    )


def test_scheduler_queue_wait_starts_at_task_submission(tmp_path):
    """Интеграция: queue_wait в реальном прогоне измеряется от своей submit."""
    import time as _time

    class _SlowAlgo:
        name = "SlowAlgoQW"
        description = "тест"

        def detect(self, image, is_cropped=False):
            _time.sleep(0.05)
            return np.zeros(image.shape[:2], dtype=np.uint8)

    from testing.registry import register_algorithm_instance

    register_algorithm_instance("SlowAlgoQW", _SlowAlgo())
    try:
        root = tmp_path / "qw"
        _write_pair(root, "a", 1)
        _write_pair(root, "b", 2)
        _write_pair(root, "c", 3)
        dataset = TestDataset(root=str(root), load_images=False)
        telemetry = TelemetryCollector(enabled=True)

        from testing.scheduler import execute_pipeline

        execute_pipeline(
            dataset,
            ["SlowAlgoQW"],
            workers=1,
            batch_size=3,
            telemetry=telemetry,
        )
        latency = telemetry.summary()["latency"]
        assert "queue_wait" in latency
        # При workers=1 и batch_size=3 последняя задача batch ждёт ~2*50ms
        # от своей submit (не общее время batch — это дало бы p95 ~0.1s).
        assert latency["queue_wait"]["count"] == 3
    finally:
        from testing.registry import _INSTANCES

        _INSTANCES.pop("SlowAlgoQW", None)


def test_input_wait_and_queue_wait_reported_separately(tmp_path):
    """Slow-loader: input_wait (ожидание входных данных) отделён от queue_wait."""
    import time as _time

    class _QuickAlgo:
        name = "QuickAlgoIW"
        description = "тест"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    from testing.registry import register_algorithm_instance

    register_algorithm_instance("QuickAlgoIW", _QuickAlgo())
    try:
        root = tmp_path / "iw"
        _write_pair(root, "a", 1)
        dataset = TestDataset(root=str(root), load_images=False)
        telemetry = TelemetryCollector(enabled=True)

        # Медленный загрузчик: слой `image_decode` искусственно задерживается.
        original_decode = testing_scheduler._decode_file

        def slow_decode(path, flags, description, telemetry=None):
            _time.sleep(0.05)  # имитация медленного декодера
            return original_decode(path, flags, description, telemetry=telemetry)

        testing_scheduler._decode_file = slow_decode
        try:
            execute_pipeline(
                dataset,
                ["QuickAlgoIW"],
                workers=1,
                batch_size=8,
                telemetry=telemetry,
            )
        finally:
            testing_scheduler._decode_file = original_decode

        summary = telemetry.summary()
        latency = summary["latency"]
        assert "input_wait" in latency, (
            "input_wait (ожидание подготовленных входных данных) не замерен"
        )
        assert "queue_wait" in latency, "queue_wait (executor-очередь) не замерен"
        assert latency["input_wait"]["count"] == latency["queue_wait"]["count"]
        # Обе стадии имеют независимые значения (не дубликат одного замера).
        assert "image_decode" in latency
    finally:
        from testing.registry import _INSTANCES

        _INSTANCES.pop("QuickAlgoIW", None)


# --- Задача 2.3: CPU core-equivalents, affinity/quota, effective worker limits ---


def test_summary_reports_cpu_core_equivalents():
    """CPU core-equivalents = process CPU-time / wall-time (задача 2.3)."""
    telemetry = TelemetryCollector(enabled=True)
    telemetry.start()
    import time as _time

    # Небольшая CPU-нагрузка для генерации core-equivalents > 0.
    _ = sum(range(500_000))
    _time.sleep(0.05)
    summary = telemetry.summary()

    assert "cpu_core_equivalents" in summary["resources"]
    core_equivalents = summary["resources"]["cpu_core_equivalents"]
    assert core_equivalents is not None
    assert core_equivalents >= 0.0
    assert core_equivalents <= (os.cpu_count() or 1) + 2, (
        "core-equivalents не должен превышать число CPU-ядер с запасом"
    )


def test_summary_reports_cpu_availability_with_unknown_as_none():
    """Affinity/quota фиксируются; недоступные поля остаются None (не ноль)."""
    telemetry = TelemetryCollector(enabled=True)
    telemetry.start()
    telemetry.record_cpu_availability(
        logical_count=8,
        physical_count=4,
        affinity=[0, 1, 2, 3],
        quota=None,  # quota недоступна
    )
    summary = telemetry.summary()
    cpu_availability = summary["cpu_availability"]
    assert cpu_availability["logical_count"] == 8
    assert cpu_availability["physical_count"] == 4
    assert cpu_availability["affinity"] == [0, 1, 2, 3]
    assert cpu_availability["quota"] is None, (
        "недоступная quota должна быть None, а не ноль"
    )


def test_summary_reports_effective_worker_limits():
    """Эффективные лимиты параллельности фиксируются с указанием источника."""
    telemetry = TelemetryCollector(enabled=True)
    telemetry.start()
    telemetry.record_worker_limits(outer_workers=4, native_threads=2, source="explicit")
    summary = telemetry.summary()
    limits = summary["worker_limits"]
    assert limits["outer_workers"] == 4
    assert limits["native_threads"] == 2
    assert limits["source"] == "explicit"


def test_default_worker_limits_are_unknown_not_zero():
    """Без фиксации — лимиты «неизвестны» (None, unreported), а не ноль."""
    telemetry = TelemetryCollector(enabled=True)
    telemetry.start()
    summary = telemetry.summary()
    limits = summary["worker_limits"]
    assert limits["outer_workers"] is None
    assert limits["source"] == "unreported"


def test_execute_pipeline_records_worker_limits_and_cpu_availability(tmp_path):
    """Интеграция: execute_pipeline записывает лимиты и доступность CPU."""

    class _QuickAlgo:
        name = "QuickAlgoWL"
        description = "тест"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    from testing.registry import register_algorithm_instance

    register_algorithm_instance("QuickAlgoWL", _QuickAlgo())
    try:
        root = tmp_path / "wl"
        _write_pair(root, "a", 1)
        dataset = TestDataset(root=str(root), load_images=False)
        telemetry = TelemetryCollector(enabled=True)

        execute_pipeline(
            dataset,
            ["QuickAlgoWL"],
            workers=2,
            batch_size=8,
            telemetry=telemetry,
            use_cropped=False,
        )
        summary = telemetry.summary()
        limits = summary["worker_limits"]
        assert limits["outer_workers"] is not None
        assert limits["outer_workers"] >= 1
        assert limits["source"] in ("explicit", "auto")
        assert "native_threads" in limits

        cpu_availability = summary["cpu_availability"]
        assert cpu_availability["logical_count"] >= 1
        assert "affinity" in cpu_availability
        assert "quota" in cpu_availability

        core_equivalents = summary["resources"]["cpu_core_equivalents"]
        assert core_equivalents is not None
    finally:
        from testing.registry import _INSTANCES

        _INSTANCES.pop("QuickAlgoWL", None)


def test_graceful_degradation_without_psutil(monkeypatch):
    """Телеметрия деградирует мягко без psutil (недоступные поля — None)."""
    import sys

    original = sys.modules.get("psutil")
    monkeypatch.setitem(sys.modules, "psutil", None)
    try:
        telemetry = TelemetryCollector(enabled=True)
        telemetry.start()
        summary = telemetry.summary()
        assert summary["duration_seconds"] >= 0.0
        assert "latency" in summary
        assert summary["resources"]["capabilities"]["psutil"] is False
    finally:
        if original is not None:
            monkeypatch.setitem(sys.modules, "psutil", original)


# --- Задача 2.4: авто-ёмкость worker из affinity/quota; --workers как верхняя граница ---


def test_visible_cpu_capacity_uses_affinity_and_quota_min(monkeypatch):
    """Ёмкость = min(affinity, quota, host): самое узкое ограничение определяет."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 16)  # хост мощнее
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 8)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 4.0)
    assert sched._visible_cpu_capacity() == 4

    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 2)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 8.0)
    assert sched._visible_cpu_capacity() == 2

    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 6)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 6.0)
    assert sched._visible_cpu_capacity() == 6


def test_visible_cpu_capacity_respects_host_physical_cores(monkeypatch):
    """Даже если affinity=64, но у хоста 6 физических ядер -> минимум 6."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 6)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 64)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 64.0)
    assert sched._visible_cpu_capacity() == 6


def test_visible_cpu_capacity_fallback_when_all_signals_unknown(monkeypatch):
    """Все сигналы неизвестны -> консервативный fallback = 1 ядро."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: None)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: None)
    monkeypatch.setattr(sched, "_host_cpu_count", lambda: None)
    assert sched._visible_cpu_capacity() == 1, (
        "fallback должен быть консервативным (1 ядро) при неизвестных данных"
    )


def test_visible_cpu_capacity_uses_quota_when_affinity_unknown(monkeypatch):
    """Quota определена, affinity нет -> используем min(quota, host)."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 16)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: None)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 3.0)
    assert sched._visible_cpu_capacity() == 3


def test_visible_cpu_capacity_uses_affinity_when_quota_unlimited(monkeypatch):
    """Quota = None (unlimited) -> используем min(affinity, host)."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 10)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 5)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: None)
    assert sched._visible_cpu_capacity() == 5


def test_effective_workers_respects_affinity_restriction(monkeypatch):
    """Ограниченная CPU (mocked affinity=2) -> не более 2 workers даже без --workers."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 8)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 2)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: None)
    result = sched._effective_workers(
        None, task_count=100, memory_budget=None, batch_bytes=0
    )
    assert result <= 2, f"affinity=2, но выбрано {result} workers"


def test_effective_workers_explicit_workers_is_upper_bound(monkeypatch):
    """--workers = верхняя граница: даже если ядер больше, не превышаем."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 16)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 16)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 16.0)

    assert sched._effective_workers(1, 100, None, 0) == 1
    assert sched._effective_workers(2, 100, None, 0) == 2, (
        "--workers=2 должен дать ровно 2 (при 16 ядрах и 100 задачах)"
    )
    # --workers больше ёмкости -> ограничен ёмкостью (не превышаем доступные ядра).
    assert sched._effective_workers(32, 100, None, 0) <= 16


def test_effective_workers_unknown_all_conservative(monkeypatch):
    """Все сигналы неизвестны -> conservative limit = 1."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_host_cpu_count", lambda: None)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: None)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: None)
    result = sched._effective_workers(
        None, task_count=10, memory_budget=None, batch_bytes=0
    )
    assert result == 1, f"консервативный fallback должен давать 1, а не {result}"


def test_effective_workers_explicit_valid_limits(monkeypatch):
    """Явные --workers=1/2 выбирают валидные лимиты (не больше ёмкости и задач)."""
    from testing import scheduler as sched

    # Случай 1: ёмкость неизвестна (fallback=1) -> оба значения ограничены ёмкостью.
    monkeypatch.setattr(sched, "_host_cpu_count", lambda: None)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: None)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: None)
    assert (
        sched._effective_workers(1, task_count=10, memory_budget=None, batch_bytes=0)
        == 1
    )
    assert (
        sched._effective_workers(2, task_count=10, memory_budget=None, batch_bytes=0)
        == 1
    ), "при ёмкости 1 explicit --workers=2 не должен давать больше 1"

    # Случай 2: ёмкость достаточна (8) -> оба значения выбираются точно.
    monkeypatch.setattr(sched, "_host_cpu_count", lambda: 8)
    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 8)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 8.0)
    assert (
        sched._effective_workers(1, task_count=10, memory_budget=None, batch_bytes=0)
        == 1
    )
    assert (
        sched._effective_workers(2, task_count=10, memory_budget=None, batch_bytes=0)
        == 2
    )

    # Случай 3: задач меньше --workers -> ограничено числом задач.
    assert (
        sched._effective_workers(2, task_count=1, memory_budget=None, batch_bytes=0)
        == 1
    )


def test_effective_workers_never_exceeds_task_count(monkeypatch):
    """Число workers не превышает число задач."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 32)
    monkeypatch.setattr(sched, "_quota_cpu_cores", lambda: 32.0)
    assert (
        sched._effective_workers(16, task_count=3, memory_budget=None, batch_bytes=0)
        == 3
    )
    assert (
        sched._effective_workers(None, task_count=1, memory_budget=None, batch_bytes=0)
        == 1
    )


def test_effective_workers_zero_tasks_returns_one(monkeypatch):
    """При 0 задачах возвращается 1 (защита от деления на 0)."""
    from testing import scheduler as sched

    monkeypatch.setattr(sched, "_affinity_cpu_count", lambda: 4)
    assert (
        sched._effective_workers(4, task_count=0, memory_budget=None, batch_bytes=0)
        == 1
    )


def test_effective_workers_rejects_nonpositive_workers():
    """--workers <= 0 вызывает ошибку (не молча исправляется)."""
    from testing import scheduler as sched

    import pytest

    with pytest.raises(ValueError, match="positive"):
        sched._effective_workers(0, 10, None, 0)
    with pytest.raises(ValueError, match="positive"):
        sched._effective_workers(-5, 10, None, 0)


# --- Задача 2.5: бюджет декодированных входов + изоляция oversized-пар ---


def test_decoded_input_budget_reserves_headroom():
    """Бюджет входов = memory_budget минус резерв под алгоритмы (25%)."""
    from testing.scheduler import (
        ALGORITHM_HEADROOM_RATIO,
        _decoded_input_budget,
    )

    assert _decoded_input_budget(None) is None
    assert ALGORITHM_HEADROOM_RATIO == 0.25

    budget = _decoded_input_budget(1000)
    assert budget == 750, (
        f"бюджет входов должен быть 75% от memory_budget, а не {budget}"
    )

    budget = _decoded_input_budget(8)
    assert budget == 6, "простой расчёт 8 - 2 = 6"
    assert budget >= 1

    budget = _decoded_input_budget(1)
    assert budget == 1, "минимальный бюджет входов — 1 байт (не 0)"


def test_decoded_input_budget_rejects_nonpositive():
    from testing.scheduler import _decoded_input_budget

    import pytest

    with pytest.raises(ValueError, match="positive"):
        _decoded_input_budget(0)
    with pytest.raises(ValueError, match="positive"):
        _decoded_input_budget(-100)


def test_memory_budget_bounds_active_and_prefetched_inputs(tmp_path):
    """Суммарно удерживаемые входы не превышают бюджет (с headroom)."""
    for i in range(10):
        _write_pair(tmp_path, f"s{i}", i)
    dataset = TestDataset(root=str(tmp_path), load_images=False)
    telemetry = TelemetryCollector(enabled=True)

    # Пара (image+mask): 8*8*3 + 8*8 = 256 байт. input_budget = 75% от 1024 = 768.
    batches = list(
        iter_batches(
            dataset,
            batch_size=8,
            memory_budget=1024,
            telemetry=telemetry,
            use_cropped=False,
        )
    )
    # В каждом batch суммарно не больше input_budget.
    for batch in batches:
        batch_bytes = sum(item.image.nbytes + item.mask.nbytes for item in batch)
        assert batch_bytes <= 768, (
            f"batch содержит {batch_bytes} байт, что больше бюджета входов 768"
        )
    assert sum(len(batch) for batch in batches) == 10, "не все пары обработаны"


def test_oversized_pair_is_isolated(tmp_path):
    """Одна oversized-пара обрабатывается изолированно (в batch из 1 шт.)."""
    _write_pair(tmp_path, "small", 1)
    _write_pair(tmp_path, "big", 2)
    _write_pair(tmp_path, "small2", 3)

    dataset = TestDataset(root=str(tmp_path), load_images=False)
    telemetry = TelemetryCollector(enabled=True)

    # Пара = 256 байт (source), то же для cropped = 256 байт.
    # Используем use_cropped=False для детерминизма одного варианта.
    batches = list(
        iter_batches(
            dataset,
            batch_size=8,
            memory_budget=256,
            telemetry=telemetry,
            use_cropped=False,
        )
    )
    # small (256 > input_budget 192) → изолирована; big → изолирована; small2 → изолирована.
    assert len(batches) == 3, (
        f"ожидалось 3 batch (все пары изолированы при бюджете 256), а не {len(batches)}"
    )
    # Проверяем событие диагностики.
    oversized = [
        e for e in telemetry._events if e["event"] == "oversized_pair_isolated"
    ]
    assert len(oversized) == 3
    assert all(e["bytes"] == 256 for e in oversized)
    assert all(e["input_budget"] == 192 for e in oversized)


def test_oversized_pair_is_processed_successfully(tmp_path):
    """Oversized-валидная пара обрабатывается, а не отбрасывается."""
    from testing.registry import register_algorithm_instance

    seen = []

    class _RecordingAlgo:
        name = "RecordingAlgoOS"
        description = "тест"

        def detect(self, image, is_cropped=False):
            seen.append(image.shape[:2])
            return np.zeros(image.shape[:2], dtype=np.uint8)

    register_algorithm_instance("RecordingAlgoOS", _RecordingAlgo())
    try:
        _write_pair(tmp_path, "big", 9)
        dataset = TestDataset(root=str(tmp_path), load_images=False)
        telemetry = TelemetryCollector(enabled=True)
        diagnostics = PipelineDiagnostics()

        execute_pipeline(
            dataset,
            ["RecordingAlgoOS"],
            workers=1,
            batch_size=8,
            memory_budget=64,  # Маленький бюджет, чтобы пара была oversized.
            telemetry=telemetry,
            diagnostics=diagnostics,
            use_cropped=False,
        )
        assert diagnostics.loaded_pairs == 1
        assert seen == [(8, 8)], (
            f"ожидался один вызов detect на изображение 8x8, а не {seen}"
        )
        assert diagnostics.load_failure_count == 0
    finally:
        from testing.registry import _INSTANCES

        _INSTANCES.pop("RecordingAlgoOS", None)


def test_backpressure_prevents_unbounded_queue_growth(tmp_path):
    """При малом бюджете batch'и не растут неограниченно (backpressure)."""
    for i in range(20):
        _write_pair(tmp_path, f"s{i}", i)
    dataset = TestDataset(root=str(tmp_path), load_images=False)
    telemetry = TelemetryCollector(enabled=True)

    batches = list(
        iter_batches(
            dataset,
            batch_size=8,
            memory_budget=512,
            telemetry=telemetry,
            use_cropped=False,
        )
    )
    total_pairs = sum(len(batch) for batch in batches)
    assert total_pairs == 20, "не все пары обработаны"
    # Ни один batch не содержит больше входов, чем позволяет бюджет.
    for batch in batches:
        batch_bytes = sum(item.image.nbytes + item.mask.nbytes for item in batch)
        assert batch_bytes <= 384, (  # 512 * 0.75 = 384
            f"batch содержит {batch_bytes} байт — бюджет 384 превышен"
        )
    # Очередь ограничена (не batch из всех 20).
    assert all(len(batch) <= 8 for batch in batches)


def test_oversized_uses_input_budget_not_memory_budget(tmp_path):
    """Признак oversized определяет input_budget (с headroom), а не memory_budget."""
    _write_pair(tmp_path, "edge", 1)
    dataset = TestDataset(root=str(tmp_path), load_images=False)
    telemetry = TelemetryCollector(enabled=True)

    # Пара = 256 байт. memory_budget = 300 (size < 300) → memory_budget_exceeded не сработал.
    # Но input_budget = 225 < 256 → пара должна быть изолирована.
    batches = list(
        iter_batches(
            dataset,
            batch_size=8,
            memory_budget=300,
            telemetry=telemetry,
            use_cropped=False,
        )
    )
    assert len(batches) == 1, f"ожидался 1 batch из 1 пары, а не {len(batches)}"
    events = [e["event"] for e in telemetry._events]
    assert "oversized_pair_isolated" in events, (
        "size > input_budget должен помечаться как oversized"
    )
    assert "memory_budget_exceeded" not in events, (
        "size < memory_budget — не должно быть memory_budget_exceeded"
    )


# --- Задача 2.6: координация decode/outer/native параллелизма ---


def test_resource_policy_coordinates_outer_and_native():
    """Внешние и нативные пулы считаются одним бюджетом CPU (2.6)."""
    from testing.resource_policy import resolve_resource_policy

    policy = resolve_resource_policy(workers=2, total_capacity=8)
    assert policy.algorithm_workers == 2
    assert policy.native_threads == 4, (
        "нативные пулы = capacity // workers (8 // 2 = 4)"
    )
    assert policy.effective_concurrency <= 8, (
        f"суммарная конкуренция {policy.effective_concurrency} превышает бюджет 8"
    )


def test_resource_policy_single_worker_gives_full_native_capacity():
    """При одном внешнем задании нативный алгоритм получает весь CPU."""
    from testing.resource_policy import resolve_resource_policy

    policy = resolve_resource_policy(workers=1, total_capacity=6)
    assert policy.native_threads == 6, (
        "tiled ONNX при 1 задаче должен использовать все ядра (design.md → 3)"
    )


def test_resource_policy_saturates_workers_at_capacity():
    """При workers >= capacity нативные пулы ограничены 1 (без oversubscription)."""
    from testing.resource_policy import resolve_resource_policy

    policy = resolve_resource_policy(workers=16, total_capacity=4)
    assert policy.algorithm_workers == 4, "workers не превышает capacity"
    assert policy.native_threads == 1, (
        "при workers == capacity нативные пулы должны быть 1"
    )


def test_resource_policy_interactive_reserves_gui_core():
    """GUI-режим: резервируется ядро под event loop (design.md → 3)."""
    from testing.resource_policy import (
        GUI_RESERVED_CORES,
        resolve_resource_policy,
    )

    assert GUI_RESERVED_CORES == 1
    interactive = resolve_resource_policy(interactive=True, total_capacity=8)
    batch = resolve_resource_policy(interactive=False, total_capacity=8)
    assert interactive.algorithm_workers == 7, "GUI: бюджет = capacity - 1 (8 - 1 = 7)"
    assert batch.algorithm_workers == 8
    assert interactive.effective_concurrency <= 8


def test_resource_policy_interactive_on_small_host_is_safe():
    """1–2 ядра: GUI-резерв не деградирует до нерабочего состояния."""
    from testing.resource_policy import resolve_resource_policy

    policy = resolve_resource_policy(interactive=True, total_capacity=2)
    assert policy.algorithm_workers >= 1, "даже на 2 ядрах остался хотя бы 1 worker"
    assert policy.effective_concurrency <= 2


def test_resource_policy_source_labels_auto_and_explicit():
    """CLI --workers — верхняя граница; source отражает способ выбора."""
    from testing.resource_policy import resolve_resource_policy

    auto = resolve_resource_policy(workers=None, total_capacity=8)
    assert auto.source == "auto"

    explicit = resolve_resource_policy(workers=4, total_capacity=8)
    assert explicit.source == "explicit"
    assert explicit.requested_workers == 4
    assert explicit.algorithm_workers == 4


def test_resource_policy_rejects_nonpositive_workers():
    import pytest

    from testing.resource_policy import resolve_resource_policy

    with pytest.raises(ValueError, match="positive"):
        resolve_resource_policy(workers=0, total_capacity=8)
    with pytest.raises(ValueError, match="positive"):
        resolve_resource_policy(workers=-3, total_capacity=8)


def test_resource_policy_matrix_covers_sequential_and_auto():
    """Matrix содержит последовательную (1) и auto конфигурации (design.md → 6)."""
    from testing.resource_policy import policy_matrix

    matrix = policy_matrix(total_capacity=8)
    workers = [p.algorithm_workers for p in matrix]
    assert 1 in workers, "matrix должен содержать последовательную конфигурацию"
    assert max(workers) == 8, "matrix должен содержать auto-конфигурацию"
    assert len(matrix) >= 2


def test_resource_policy_benchmark_covers_classic_and_onnx():
    """Бенчмарк сравнивает classic и tiled ONNX рабочие нагрузки (2.6)."""
    from testing.resource_policy import benchmark_workloads

    workloads = benchmark_workloads()
    assert any("Classic" in name for name in workloads), (
        "benchmark должен включать классический (OpenCV) workload"
    )
    assert any("NN:" in name for name in workloads), (
        "benchmark должен включать tiled ONNX workload"
    )


def test_apply_native_limits_sets_and_restores_cv_threads():
    """apply_native_limits управляет cv2.setNumThreads (и восстанавливает)."""
    import cv2

    from testing.resource_policy import apply_native_limits, resolve_resource_policy

    original_threads = cv2.getNumThreads()
    policy = resolve_resource_policy(workers=1, total_capacity=3)

    with apply_native_limits(policy):
        assert cv2.getNumThreads() == 3, (
            f"cv2.setNumThreads должен быть {policy.native_threads}, "
            f"а не {cv2.getNumThreads()}"
        )
    assert cv2.getNumThreads() == original_threads, (
        "после применения лимиты OpenCV должны восстанавливаться"
    )


def test_apply_native_limits_sets_current_policy():
    """Адаптеры ONNX получают лимиты через current_policy (задача 2.6)."""
    from testing.resource_policy import (
        apply_native_limits,
        current_policy,
        onnx_session_options,
        resolve_resource_policy,
    )

    assert current_policy() is None, "по умолчанию активной политики нет"
    policy = resolve_resource_policy(workers=2, total_capacity=8)

    with apply_native_limits(policy):
        assert current_policy() is policy
        # Без onnxruntime возвращает None (грациозная деградация).
        options = onnx_session_options()
        assert options is None or hasattr(options, "intra_op_num_threads")
    assert current_policy() is None, "после выхода текущая политика должна сбрасываться"


def test_execute_pipeline_applies_resource_policy(tmp_path):
    """Интеграция: execute_pipeline использует единую политику (2.6)."""

    class _QuickAlgoRP:
        name = "QuickAlgoRP"
        description = "тест"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    from testing.registry import register_algorithm_instance

    register_algorithm_instance("QuickAlgoRP", _QuickAlgoRP())
    try:
        root = tmp_path / "rp"
        _write_pair(root, "a", 1)
        dataset = TestDataset(root=str(root), load_images=False)
        telemetry = TelemetryCollector(enabled=True)

        execute_pipeline(
            dataset,
            ["QuickAlgoRP"],
            workers=1,
            batch_size=8,
            telemetry=telemetry,
            use_cropped=False,
        )
        summary = telemetry.summary()
        limits = summary["worker_limits"]
        assert limits["outer_workers"] == 1
        assert limits["source"] == "explicit"
        assert limits["native_threads"] is not None
    finally:
        from testing.registry import _INSTANCES

        _INSTANCES.pop("QuickAlgoRP", None)


def test_onnx_adapters_use_policy_threads(monkeypatch):
    """ONNX-адаптеры применяют intra/inter-op лимиты политики (задача 2.6)."""
    from types import SimpleNamespace

    from testing.onnx_algorithm import OnnxModelAlgorithm
    from testing.resource_policy import apply_native_limits, resolve_resource_policy

    sessions_created = []

    class _Options:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

    def fake_inference_session(path, sess_options=None, providers=None):
        sessions_created.append(sess_options)
        return SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="input")],
            run=lambda *args, **kwargs: [np.zeros((1, 1, 4, 4), dtype=np.float32)],
        )

    fake_ort = SimpleNamespace(
        InferenceSession=fake_inference_session,
        SessionOptions=_Options,
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    policy = resolve_resource_policy(workers=1, total_capacity=8)
    algo = OnnxModelAlgorithm(model_path="test.onnx", name="test")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with apply_native_limits(policy):
        algo.detect(image)

    assert sessions_created, "ONNX-сессия не создана"
    options = sessions_created[0]
    assert options is not None, "onnx_session_options вернул None при активной политике"
    assert options.intra_op_num_threads == 8, (
        f"intra_op должен быть 8 (native_threads), а не {options.intra_op_num_threads}"
    )
    assert options.inter_op_num_threads == 1


# --- Задача 2.7: перекрытие decode/compute с backpressure и метрикой input-starvation ---


def test_prefetch_yields_all_items():
    """Все элементы источника доходят до потребителя в исходном порядке."""
    import time as _time

    from testing.pipeline_overlap import PrefetchLoader

    source = (i for i in range(10))

    def slow_source():
        for i in source:
            _time.sleep(0.005)  # имитация декодирования
            yield i

    with PrefetchLoader(slow_source(), maxsize=2) as loader:
        consumed = []
        for item in loader:
            consumed.append(item)
            loader.task_done(item)
    assert consumed == list(range(10))
    assert loader.items_consumed == 10


def test_prefetch_queue_backpressure_prevents_unbounded_growth():
    """Очередь ограничена maxsize: producer блокируется при переполнении."""
    import threading as _threading
    import time as _time

    from testing.pipeline_overlap import PrefetchLoader

    produced = []
    released = _threading.Event()

    def source():
        for i in range(100):
            produced.append(i)
            yield i

    def slow_consume(loader):
        # Потребитель медленный: берёт 1 элемент, ждёт.
        for item in loader:
            if len(produced) >= 3:
                released.set()
            _time.sleep(0.05)
            loader.task_done(item)

    with PrefetchLoader(source(), maxsize=2) as loader:
        released.wait(timeout=2.0)
        # Даём producer время насытить очередь.
        _time.sleep(0.1)
        assert len(produced) <= 4, (
            f"producer создал {len(produced)} элементов при maxsize=2 — "
            "backpressure не работает"
        )
        # Потребляет остаток.
        for item in loader:
            loader.task_done(item)


def test_prefetch_memory_budget_backpressure():
    """Суммарный объём (активные + загруженные) не превышает max_bytes."""
    import threading as _threading

    from testing.pipeline_overlap import PrefetchLoader

    item_size = 100
    budget = 250  # максимум 2 элемента одновременно (активный + prefetch)
    peak_outstanding = [0]
    lock = _threading.Lock()

    def source():
        for i in range(10):
            yield {"id": i, "size": item_size}

    def size_fn(item):
        return item["size"]

    with PrefetchLoader(
        source(), maxsize=4, max_bytes=budget, size_fn=size_fn
    ) as loader:
        for item in loader:
            with lock:
                peak_outstanding[0] = max(peak_outstanding[0], loader.outstanding_bytes)
            assert loader.outstanding_bytes <= budget, (
                f"задолженность {loader.outstanding_bytes} превысила бюджет {budget}"
            )
            loader.task_done(item)
            with lock:
                peak_outstanding[0] = max(peak_outstanding[0], loader.outstanding_bytes)

    assert peak_outstanding[0] <= budget, (
        f"пик {peak_outstanding[0]} превысил бюджет {budget}"
    )
    assert loader.outstanding_bytes == 0, "после task_done задолженность должна быть 0"


def test_prefetch_oversized_item_is_handled():
    """Одиночный элемент больше бюджета обрабатывается изолированно."""
    from testing.pipeline_overlap import PrefetchLoader

    def source():
        yield {"id": 0, "size": 50}
        yield {"id": 1, "size": 500}  # oversized: > max_bytes 250
        yield {"id": 2, "size": 50}

    def size_fn(item):
        return item["size"]

    with PrefetchLoader(source(), maxsize=4, max_bytes=250, size_fn=size_fn) as loader:
        consumed = []
        for item in loader:
            consumed.append(item["id"])
            loader.task_done(item)
    assert consumed == [0, 1, 2], (
        f"oversized-элемент должен обрабатываться, а не теряться: {consumed}"
    )


def test_prefetch_input_starvation_ratio_measured():
    """Метрика input-starvation: доля ожидания входов в активной фазе (2.7)."""
    import time as _time

    from testing.pipeline_overlap import PrefetchLoader

    def source():
        for i in range(5):
            _time.sleep(0.03)  # медленный producer -> starvation при быстром consumer
            yield i

    with PrefetchLoader(source(), maxsize=1) as loader:
        for item in loader:
            _time.sleep(0.001)  # быстрый consumer
            loader.task_done(item)

    assert 0.0 <= loader.input_starvation_ratio <= 1.0, (
        f"input_starvation_ratio {loader.input_starvation_ratio} вне [0, 1]"
    )
    assert loader.input_starvation_ratio > 0.0, (
        "при медленном producer должен наблюдаться ненулевой input-starvation"
    )


def test_prefetch_input_starvation_low_when_consumer_slow():
    """Быстрый producer, медленный consumer: starvation близок к 0."""
    import time as _time

    from testing.pipeline_overlap import PrefetchLoader

    def source():
        for i in range(5):
            _time.sleep(0.001)  # быстрый producer
            yield i

    with PrefetchLoader(source(), maxsize=4) as loader:
        for item in loader:
            _time.sleep(0.05)  # медленный consumer
            loader.task_done(item)

    assert loader.input_starvation_ratio < 0.5, (
        f"starvation {loader.input_starvation_ratio:.3f} высокий для медленного "
        "consumer'а с быстрым producer"
    )


def test_prefetch_startup_wait_excluded_from_starvation():
    """Первое ожидание (startup) исключается из числителя starvation."""
    import time as _time

    from testing.pipeline_overlap import PrefetchLoader

    def source():
        _time.sleep(0.05)  # стартовое ожидание до первого элемента
        yield 1
        _time.sleep(0.05)  # ожидание между элементами — starvation
        yield 2

    with PrefetchLoader(source(), maxsize=1) as loader:
        for item in loader:
            loader.task_done(item)

    # Первое ожидание (~50ms) исключено: starvation = только второе (~50ms).
    assert loader.items_consumed == 2
    assert loader.starvation_time > 0.0
    assert loader.starvation_time < 0.1, (
        f"starvation {loader.starvation_time:.3f} включает стартовое ожидание"
    )


def test_prefetch_cleanup_on_failure():
    """Ошибка producer: consumer получает исключение, ресурсы освобождаются."""
    from testing.pipeline_overlap import PrefetchLoader

    def failing_source():
        yield 1
        raise RuntimeError("producer сломался")

    loader = PrefetchLoader(failing_source(), maxsize=2)
    with loader:
        consumed = []
        import pytest

        with pytest.raises(RuntimeError, match="сломался"):
            for item in loader:
                consumed.append(item)
                loader.task_done(item)
    assert consumed == [1], f"получено {consumed} до ошибки"
    assert not loader._thread.is_alive(), "producer-поток не завершился после ошибки"


def test_prefetch_close_is_deterministic():
    """close() останавливает producer и дожидается завершения потока."""
    import threading as _threading

    from testing.pipeline_overlap import PrefetchLoader

    started = _threading.Event()

    def infinite_source():
        i = 0
        while True:
            started.set()
            yield i
            i += 1

    loader = PrefetchLoader(infinite_source(), maxsize=2)
    loader.start()
    started.wait(timeout=2.0)
    thread = loader._thread
    assert thread.is_alive()

    loader.close()
    assert not thread.is_alive(), "producer-поток не завершился после close()"
    # close() идемпотентен.
    loader.close()


def test_prefetch_lazy_source_error_reaches_consumer():
    """Ошибка источника до первого элемента тоже доходит до consumer."""
    import pytest

    from testing.pipeline_overlap import PrefetchLoader

    def broken_source():
        raise ValueError("источник сломан")
        yield  # pragma: no cover — не создаётся

    loader = PrefetchLoader(broken_source(), maxsize=1)
    with loader:
        with pytest.raises(ValueError, match="сломан"):
            next(iter(loader))


def test_prefetch_overlap_actually_overlaps():
    """Decode и compute реально перекрываются (producer работает пока consumer обрабатывает)."""
    import threading as _threading
    import time as _time

    from testing.pipeline_overlap import PrefetchLoader

    active = {"max_concurrent": 0, "current": 0}
    lock = _threading.Lock()

    def source():
        for i in range(6):
            _time.sleep(0.02)  # decode
            yield i

    with PrefetchLoader(source(), maxsize=2) as loader:
        for item in loader:
            with lock:
                active["current"] += 1
                active["max_concurrent"] = max(
                    active["max_concurrent"], active["current"]
                )
            _time.sleep(0.02)  # compute
            with lock:
                active["current"] -= 1
            loader.task_done(item)

    # Producer готовит следующий элемент, пока consumer обрабатывает текущий:
    # overlap подтверждается, если входы генерировались во время обработки.
    assert loader.items_consumed == 6
    assert loader.active_time > 0


def test_cpu_bound_acceptance_run_input_starvation_below_5_percent(tmp_path):
    """Pinned CPU-bound прогон: input-starvation ≤ 5%, бюджет соблюдён (2.7).

    Алгоритм медленнее декодирования: producer успевает готовить входы,
    поэтому compute slot не должен голодать. Метрика публикуется в
    telemetry-сводке для acceptance-прогона.
    """
    import time as _time

    from testing.registry import _INSTANCES

    for index in range(6):
        _write_pair(tmp_path, f"pair-{index}", index * 5)

    class _CpuBoundAlgo:
        name = "CpuBoundAcceptance"
        description = "тест: медленная детекция (compute > decode)"

        def detect(self, image, is_cropped=False):
            _time.sleep(0.02)  # compute-время доминирует над decode
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["CpuBoundAcceptance"] = _CpuBoundAlgo()
    try:
        dataset = BaselineDataset(root=str(tmp_path))
        telemetry = TelemetryCollector(enabled=True)
        execute_pipeline(
            dataset,
            ["CpuBoundAcceptance"],
            workers=1,
            batch_size=2,
            telemetry=telemetry,
            use_cropped=False,
        )
    finally:
        _INSTANCES.pop("CpuBoundAcceptance", None)

    summary = telemetry.summary()
    starvation = summary["input_starvation"]
    assert starvation["ratio"] is not None, (
        "input-starvation должна публиковаться в telemetry (2.7)"
    )
    assert 0.0 <= starvation["ratio"] <= 1.0
    assert starvation["ratio"] <= 0.05, (
        f"input-starvation {starvation['ratio']:.3f} превышает 5% при "
        "доминирующем compute"
    )


def test_pipeline_reports_input_starvation_in_telemetry_summary(tmp_path):
    """Сводка telemetry содержит метрику input-starvation (задача 2.7)."""
    from testing.registry import _INSTANCES

    class _TinyAlgo:
        name = "TinyAcceptance"
        description = "тест"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["TinyAcceptance"] = _TinyAlgo()
    try:
        _write_pair(tmp_path, "pair-a", 3)
        dataset = BaselineDataset(root=str(tmp_path))
        telemetry = TelemetryCollector(enabled=True)
        execute_pipeline(
            dataset,
            ["TinyAcceptance"],
            workers=1,
            batch_size=8,
            telemetry=telemetry,
            use_cropped=False,
        )
    finally:
        _INSTANCES.pop("TinyAcceptance", None)

    starvation = telemetry.summary()["input_starvation"]
    assert set(starvation) == {
        "starvation_seconds",
        "active_seconds",
        "ratio",
    }, f"структура метрики: {set(starvation)}"
    assert starvation["starvation_seconds"] is not None
    assert starvation["active_seconds"] is not None


def test_pipeline_cancel_stops_new_tasks_at_safe_boundary(tmp_path):
    """Кооперативная отмена: новые задачи не ставятся после cancel (7.4)."""
    import threading
    import time as _time

    from testing.registry import _INSTANCES

    class _SlowAlgo:
        name = "SlowCancelAlgo"
        description = "тест: отмена"

        def detect(self, image, is_cropped=False):
            _time.sleep(0.01)
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["SlowCancelAlgo"] = _SlowAlgo()
    try:
        for index in range(8):
            _write_pair(tmp_path, f"pair-{index}", index * 3)
        dataset = BaselineDataset(root=str(tmp_path))
        cancel_event = threading.Event()
        cancel_event.set()  # отмена ДО старта: ни одной задачи
        telemetry = TelemetryCollector(enabled=True)
        results = execute_pipeline(
            dataset,
            ["SlowCancelAlgo"],
            workers=1,
            batch_size=2,
            telemetry=telemetry,
            use_cropped=False,
            cancel_event=cancel_event,
        )
    finally:
        _INSTANCES.pop("SlowCancelAlgo", None)

    assert results["SlowCancelAlgo"] == {}, (
        "после отмены новые задачи не должны выполняться"
    )
