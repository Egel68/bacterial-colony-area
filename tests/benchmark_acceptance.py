"""Acceptance benchmark for deterministic sequential and auto pipeline runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_PATH = Path(__file__).parent / "baseline" / "benchmark_acceptance.json"
PROFILE_SAMPLE_IDS = (
    "sp21_img24",
    "sp21_img41",
    "sp21_img38",
    "sp21_img19",
    "sp21_img37",
)
ALGORITHMS = (
    "ClassicDefault",
    "ClassicHighSensitivity",
    "ClassicSolidFill",
    "ClassicLowSensitivity",
)


def _hash(results: Any) -> str:
    value = json.dumps(results, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def _dataset():
    from testing.baseline import BaselineDataset

    dataset = BaselineDataset(ROOT / "datasets" / "22022540_imported")
    wanted = set(PROFILE_SAMPLE_IDS)
    samples = [s for s in dataset.samples if s.name.removesuffix("_cropped") in wanted]
    samples.sort(key=lambda s: (s.name.removesuffix("_cropped"), s.variant))
    missing = wanted - {s.name.removesuffix("_cropped") for s in samples}
    if missing:
        raise RuntimeError(f"Reference samples missing: {sorted(missing)}")
    dataset._samples = samples
    return dataset


def _run(dataset, workers):
    from testing.resource_policy import apply_native_limits, resolve_resource_policy
    from testing.runner import run_all
    from testing.scheduler import PipelineDiagnostics
    from testing.telemetry import TelemetryCollector

    count = len(dataset.samples) * len(ALGORITHMS)
    policy = resolve_resource_policy(workers=workers, task_count=count)
    telemetry = TelemetryCollector(enabled=True)
    diagnostics = PipelineDiagnostics()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    with apply_native_limits(policy):
        results = run_all(
            dataset,
            algorithms=list(ALGORITHMS),
            workers=workers,
            batch_size=2,
            memory_budget=768 * 1024 * 1024,
            telemetry=telemetry,
            diagnostics=diagnostics,
        )
    wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    summary = telemetry.summary()
    if (
        diagnostics.completed_tasks != count
        or diagnostics.load_failure_count
        or diagnostics.algorithm_failure_count
    ):
        raise RuntimeError(f"Incomplete benchmark: {diagnostics}")
    return {
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "cpu_core_equivalents": cpu / wall if wall else None,
        "tasks_per_second": count / wall if wall else None,
        "input_starvation": summary["input_starvation"],
        "effective_policy": policy.as_dict(),
        "resources": summary["resources"],
        "latency": summary["latency"],
        "logical_io": summary["logical_io"],
        "physical_io": summary["io_condition"],
        "results_sha256": _hash(results),
        "tasks": diagnostics.completed_tasks,
    }


def _mode(records, requested):
    first = records[0]
    result_hashes = {r["results_sha256"] for r in records}
    if len(result_hashes) != 1:
        raise RuntimeError("Non-deterministic results across benchmark repetitions")
    return {
        "workers_requested": requested,
        "effective_policy": first["effective_policy"],
        "repetitions": len(records),
        "tasks": first["tasks"],
        "results_sha256": first["results_sha256"],
        "wall_seconds_median": statistics.median(r["wall_seconds"] for r in records),
        "wall_seconds_repetitions": [r["wall_seconds"] for r in records],
        "process_cpu_seconds_median": statistics.median(
            r["process_cpu_seconds"] for r in records
        ),
        "cpu_core_equivalents_median": statistics.median(
            r["cpu_core_equivalents"] for r in records
        ),
        "tasks_per_second_median": statistics.median(
            r["tasks_per_second"] for r in records
        ),
        "effective_concurrency": first["effective_policy"]["algorithm_workers"]
        * first["effective_policy"]["native_threads"],
        "input_starvation_ratio_median": statistics.median(
            (r["input_starvation"] or {}).get("ratio") or 0 for r in records
        ),
        "input_starvation_seconds_median": statistics.median(
            (r["input_starvation"] or {}).get("starvation_seconds") or 0
            for r in records
        ),
        "peak_rss_bytes_max": max(
            (r["resources"]["memory"].get("peak_rss") or 0 for r in records), default=0
        )
        or None,
        "latency_stages": first["latency"],
        "logical_io": first["logical_io"],
        "physical_io": first["physical_io"],
    }


def build_report():
    from testing.resource_policy import resolve_resource_policy

    dataset = _dataset()
    warmup = _run(dataset, 1)
    configs = (("sequential", 1), ("conservative_auto", None))
    records = {name: [] for name, _ in configs}
    for repeat in range(3):
        order = configs if repeat % 2 == 0 else tuple(reversed(configs))
        for name, workers in order:
            records[name].append(_run(dataset, workers))
    modes = {name: _mode(records[name], workers) for name, workers in configs}
    hashes = {mode["results_sha256"] for mode in modes.values()}
    if len(hashes) != 1:
        raise RuntimeError("Sequential and conservative-auto results differ")

    samples = []
    for sample in dataset.samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        samples.append(
            {
                "id": sample.name,
                "variant": sample.variant,
                "image_path": str(sample.image_path.relative_to(ROOT)),
                "mask_path": str(sample.mask_path.relative_to(ROOT)),
                "image_bytes": sample.image_path.stat().st_size,
                "mask_bytes": sample.mask_path.stat().st_size,
                "image_shape": list(image.shape) if image is not None else None,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_profile": "tests/baseline/host_profile.json",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "visible_cpu_capacity": resolve_resource_policy().total_capacity,
            "warmup_wall_seconds": warmup["wall_seconds"],
            "warm_cache": "one warm-up then alternating repeated runs; cold-storage I/O unmeasured",
        },
        "workload": {
            "dataset": "datasets/22022540_imported",
            "sample_ids": list(PROFILE_SAMPLE_IDS),
            "variants": ["source", "cropped"],
            "algorithms": list(ALGORITHMS),
            "task_count": len(dataset.samples) * len(ALGORITHMS),
            "memory_budget_bytes": 768 * 1024 * 1024,
            "batch_size": 2,
            "order": "sample_id then source/cropped",
            "samples": samples,
        },
        "modes": modes,
        "correctness": {
            "identical_results_sha256": True,
            "results_sha256": next(iter(hashes)),
        },
        "io_conditions": {
            "measured": "logical file-read/decode latency in warm-cache runs",
            "unmeasured": "cold-storage physical I/O; no cache dropping performed",
            "interpretation": "zero or unavailable physical counters do not imply no logical reads",
        },
    }


def publish(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=REPORT_PATH.parent, prefix=f".{REPORT_PATH.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        Path(tmp).replace(REPORT_PATH)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.write and not args.check:
        parser.print_help()
        return 0
    report = build_report()
    if args.write:
        publish(report)
        print(f"Acceptance report written: {REPORT_PATH}")
    else:
        print(
            json.dumps(
                {
                    "modes": {
                        k: {
                            "wall_seconds_median": v["wall_seconds_median"],
                            "tasks_per_second_median": v["tasks_per_second_median"],
                            "input_starvation_ratio_median": v[
                                "input_starvation_ratio_median"
                            ],
                        }
                        for k, v in report["modes"].items()
                    },
                    "correctness": report["correctness"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
