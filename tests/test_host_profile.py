"""Задача 1.1: воспроизводимый профиль хоста для бенчмарков.

Покрывает: захват ОС, CPU (модель, affinity, квота), RAM, версий OpenCV/
ONNX Runtime, идентичности снимков датасета и их размеров, состояния
тёплого кеша; проверку сохранённого профиля хоста.
"""

import json
from pathlib import Path

import pytest

from analysis.host_profile import (
    SCHEMA_VERSION,
    capture_host_profile,
    detect_io_counters,
    load_host_profile,
    write_host_profile,
)

pytest.importorskip("cv2")

PROFILE_PATH = Path(__file__).parent / "baseline" / "host_profile.json"


class TestCaptureHostProfile:
    def test_captures_os(self):
        profile = capture_host_profile()
        assert profile["os"]["system"]
        assert profile["os"]["machine"]
        assert "release" in profile["os"]

    def test_captures_cpu_model_and_counts(self):
        profile = capture_host_profile()
        cpu = profile["cpu"]
        # Модель CPU: может быть None вне Linux, но поле присутствует.
        assert "model" in cpu
        assert cpu["logical_count"] >= 1
        assert cpu["physical_count"] is None or cpu["physical_count"] >= 1

    def test_captures_cpu_affinity_or_none(self):
        profile = capture_host_profile()
        affinity = profile["cpu"]["affinity"]
        if affinity is not None:
            assert isinstance(affinity, list)
            assert all(isinstance(cpu, int) for cpu in affinity)
            assert len(affinity) >= 1

    def test_captures_cpu_quota_or_none(self):
        profile = capture_host_profile()
        quota = profile["cpu"]["quota"]
        if quota is not None:
            assert quota["version"] in (1, 2)
            assert "quota_us" in quota and "period_us" in quota
            assert quota["period_us"] > 0

    def test_captures_ram(self):
        profile = capture_host_profile()
        memory = profile["memory"]
        assert "total_bytes" in memory
        assert "available_bytes" in memory
        if memory["total_bytes"] is not None:
            assert memory["total_bytes"] > 0

    def test_captures_opencv_version(self):
        profile = capture_host_profile()
        assert profile["runtimes"]["opencv"], "версия OpenCV не зафиксирована"

    def test_captures_onnxruntime_version_or_none(self):
        profile = capture_host_profile()
        assert "onnxruntime" in profile["runtimes"]
        # None допустим, если onnxruntime не установлен.

    def test_captures_numpy_version(self):
        profile = capture_host_profile()
        assert profile["runtimes"]["numpy"], "версия NumPy не зафиксирована"

    def test_captures_dataset_sample_identities_and_dimensions(self):
        samples = [
            ("sp21_img24.jpg", 5927, 5968),
            ("sp21_img41.jpg", 5942, 5920),
        ]
        profile = capture_host_profile(sample_ids=samples)
        dataset = profile["dataset"]
        assert len(dataset["samples"]) == 2
        for entry, (name, w, h) in zip(dataset["samples"], samples):
            assert entry["id"] == name
            assert entry["width"] == w and entry["height"] == h

    def test_captures_warm_cache_condition(self):
        warm = capture_host_profile(warm_cache=True)
        assert warm["warm_cache"] is True
        cold = capture_host_profile(warm_cache=False)
        assert cold["warm_cache"] is False
        unknown = capture_host_profile(warm_cache=None)
        assert unknown["warm_cache"] is None, (
            "неизмерённое состояние тёплого кеша должно быть None"
        )

    def test_captures_dataset_root(self):
        profile = capture_host_profile(dataset_root="datasets/22022540")
        assert profile["dataset"]["root"] == "datasets/22022540"

    def test_schema_version(self):
        profile = capture_host_profile()
        assert profile["schema_version"] == SCHEMA_VERSION
        assert profile["captured_at"]

    def test_io_counters_are_detected_with_available_flag(self):
        counters = detect_io_counters()
        assert "available" in counters
        if counters["available"]:
            assert counters["read_bytes"] is not None


class TestProfilePersistence:
    def test_write_and_load_roundtrip(self, tmp_path):
        profile = capture_host_profile(sample_ids=[("a", 100, 200)], warm_cache=True)
        out = tmp_path / "profile.json"
        write_host_profile(profile, out)
        loaded = load_host_profile(out)
        assert loaded["captured_at"] == profile["captured_at"]
        assert loaded["dataset"]["samples"] == profile["dataset"]["samples"]
        assert loaded["warm_cache"] is True

    @pytest.mark.skipif(not PROFILE_PATH.exists(), reason="профиль хоста не сохранён")
    def test_committed_host_profile_is_well_formed(self):
        """Сохранённый профиль хоста содержит все обязательные поля."""
        profile = load_host_profile(PROFILE_PATH)
        assert profile["schema_version"] == SCHEMA_VERSION
        assert profile["os"]["system"]
        assert profile["cpu"]["model"] or profile["cpu"]["logical_count"] >= 1
        assert "affinity" in profile["cpu"]
        assert "quota" in profile["cpu"]
        assert "total_bytes" in profile["memory"]
        assert profile["runtimes"]["opencv"]
        assert "onnxruntime" in profile["runtimes"]
        assert profile["runtimes"]["numpy"]
        assert profile["dataset"]["samples"], "снимки датасета не зафиксированы"
        for entry in profile["dataset"]["samples"]:
            assert entry["id"]
            assert entry["width"] > 0 and entry["height"] > 0
        assert "warm_cache" in profile
        assert "io_counters" in profile
        assert "git_revision" in profile

    @pytest.mark.skipif(not PROFILE_PATH.exists(), reason="профиль хоста не сохранён")
    def test_committed_profile_records_reference_samples(self):
        """В профиле зафиксированы эталонные снимки с реальными размерами."""
        profile = load_host_profile(PROFILE_PATH)
        samples = profile["dataset"]["samples"]
        assert any(s["width"] * s["height"] >= 20_000_000 for s in samples), (
            "крупные кадры (20+ MP) не зафиксированы в профиле"
        )
        assert profile["dataset"].get("image_count", 0) > 0


# --- Задача 1.3: базовые замеры на reference-профиле ---


def test_benchmark_baseline_report_structure():
    """Отчёт различает warm-cache измеренное и неизмеренный cold-storage I/O (1.3)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_baseline import _check_report, build_report

    report = build_report(
        fixture_names=["ordinary_source", "ordinary_cropped"],
        algorithms=["ClassicDefault"],
    )

    # Sequential и current-worker режимы записаны.
    assert "sequential" in report["modes"]
    assert "current_workers" in report["modes"]
    for mode, entry in report["modes"].items():
        assert entry["duration_seconds"] > 0, f"{mode}: нет wall time"
        assert entry["process_cpu_seconds"] is not None, f"{mode}: нет CPU-time"
        assert entry["workers"] >= 1

    # Warm-cache измерен; cold-storage I/O явно НЕ измерен (не подменён нулём).
    io_conditions = report["io_conditions"]
    assert "warm-cache" in io_conditions["measured"]
    assert "cold" in io_conditions["unmeasured"].lower()
    assert "не" in io_conditions["unmeasured"].lower() or "NOT" in io_conditions["unmeasured"]

    # Heartbeat записан (или явно unavailable — не нулевые значения).
    heartbeat = report["heartbeat"]
    assert heartbeat["status"] in ("measured", "unavailable")
    if heartbeat["status"] == "measured":
        assert heartbeat["samples"] > 0
        assert heartbeat["p95_ms"] > 0

    assert _check_report(report) == [], "структура отчёта должна быть полной"


def test_benchmark_baseline_check_accepts_written_report():
    """--check принимает записанный базовый отчёт (1.3)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_baseline import BASELINE_PATH, _check_report

    if not BASELINE_PATH.exists():
        import pytest

        pytest.skip("базовый отчёт ещё не записан (--write)")
    report = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert _check_report(report) == [], "записанный отчёт должен проходить --check"
