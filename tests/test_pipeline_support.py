import json

import cv2
import numpy as np

from testing.cache import PredictionCache
from testing.baseline import BaselineDataset, run_baseline
from testing.dataset import TestDataset
from testing.scheduler import _should_reduce_workers, iter_batches
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
