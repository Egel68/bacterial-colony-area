"""Lightweight structured runtime telemetry for algorithm evaluation."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

log = logging.getLogger(__name__)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class TelemetryCollector:
    """Collect process/resource counters without making a run fail."""

    def __init__(
        self,
        enabled: bool = False,
        output_path: str | Path | None = None,
        interval: float = 5.0,
        debug_tasks: bool = False,
    ) -> None:
        self.enabled = enabled
        self.output_path = Path(output_path) if output_path else None
        self.interval = max(0.0, interval)
        self.debug_tasks = debug_tasks
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._tasks = defaultdict(int)
        self._work = defaultdict(int)
        self._cache = defaultdict(int)
        self._serial_calls = 0
        self._errors: list[str] = []
        self._started_at = 0.0
        self._last_event_at = 0.0
        self._start_cpu = 0.0
        self._finished = False
        self._start_resources = self._resources()
        self._peak_rss = self._start_resources["memory"]["rss"]

    @staticmethod
    def _resources() -> dict[str, Any]:
        resources: dict[str, Any] = {
            "cpu": {"process_percent": None, "system_percent": None},
            "memory": {"rss": None},
            "io": {
                "read_count": None,
                "write_count": None,
                "read_bytes": None,
                "write_bytes": None,
            },
            "capabilities": {"psutil": False},
        }
        try:
            import psutil

            process = psutil.Process(os.getpid())
            io = process.io_counters()
            resources["cpu"] = {
                "process_percent": process.cpu_percent(None),
                "system_percent": psutil.cpu_percent(None),
            }
            resources["memory"]["rss"] = process.memory_info().rss
            resources["io"] = {
                "read_count": io.read_count,
                "write_count": io.write_count,
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
            }
            resources["capabilities"]["psutil"] = True
        except (ImportError, OSError, AttributeError):
            # psutil is optional at import time so the CLI remains usable.
            pass
        return resources

    @staticmethod
    def _cpu_time() -> float:
        return time.process_time()

    def start(self) -> None:
        if not self.enabled:
            return
        self._started_at = time.perf_counter()
        self._start_cpu = self._cpu_time()
        self._start_resources = self._resources()
        self._peak_rss = self._start_resources["memory"]["rss"]
        self.event("run_started", resources=self._start_resources)

    def event(self, event_type: str, **payload: Any) -> None:
        if not self.enabled:
            return
        event = {
            "timestamp": time.time(),
            "event": event_type,
            **payload,
        }
        with self._lock:
            self._events.append(event)
            if self.output_path:
                self._write_event(event)

    def maybe_snapshot(self, **payload: Any) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if self.interval and now - self._last_event_at < self.interval:
            return
        self._last_event_at = now
        resources = self._resources()
        rss = resources["memory"]["rss"]
        if rss is not None:
            self._peak_rss = max(self._peak_rss or 0, rss)
        self.event("snapshot", resources=resources, **payload)

    def record_stage(self, stage: str, duration: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._latencies[stage].append(max(0.0, duration))

    def record_task(
        self,
        *,
        submitted: int = 0,
        completed: int = 0,
        failed: int = 0,
        retried: int = 0,
        active: int | None = None,
        queue_depth: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._tasks["submitted"] += submitted
            self._tasks["completed"] += completed
            self._tasks["failed"] += failed
            self._tasks["retried"] += retried
            if active is not None:
                self._tasks["active"] = active
            if queue_depth is not None:
                self._tasks["queue_depth"] = queue_depth

    def record_work(
        self,
        *,
        objects: int = 0,
        variants: int = 0,
        algorithm_calls: int = 0,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._work["objects"] += objects
            self._work["variants"] += variants
            self._work["algorithm_calls"] += algorithm_calls

    def record_cache(self, hit: bool) -> None:
        if self.enabled:
            self._cache["hits" if hit else "misses"] += 1

    def record_serial_call(self) -> None:
        if self.enabled:
            self._serial_calls += 1

    def record_error(self, message: str) -> None:
        if self.enabled:
            with self._lock:
                self._errors.append(str(message))

    def record_task_event(self, **payload: Any) -> None:
        if self.enabled and self.debug_tasks:
            self.event("task", **payload)

    def _write_event(self, event: dict[str, Any]) -> None:
        if not self.output_path:
            return
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            event_path = self.output_path.with_suffix(".jsonl")
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Unable to write telemetry event: %s", exc)

    def summary(self) -> dict[str, Any]:
        now = time.perf_counter()
        resources = self._resources()
        rss = resources["memory"]["rss"]
        if rss is not None:
            self._peak_rss = max(self._peak_rss or 0, rss)
        start_io = self._start_resources["io"]
        end_io = resources["io"]
        io_delta = {
            key: (
                end_io[key] - start_io[key]
                if end_io[key] is not None and start_io[key] is not None
                else None
            )
            for key in ("read_count", "write_count", "read_bytes", "write_bytes")
        }
        elapsed = max(0.0, now - self._started_at) if self._started_at else 0.0
        process_cpu_time = max(0.0, self._cpu_time() - self._start_cpu)
        cpu = dict(resources["cpu"])
        if elapsed:
            cpu["process_percent"] = min(
                100.0,
                100.0 * process_cpu_time / elapsed / max(os.cpu_count() or 1, 1),
            )
        latency = {}
        for stage, values in self._latencies.items():
            latency[stage] = {
                "count": len(values),
                "median": median(values) if values else 0.0,
                "p50": _percentile(values, 50),
                "p95": _percentile(values, 95),
                "p99": _percentile(values, 99),
                "total": sum(values),
            }
        throughput = {
            key: (value / elapsed if elapsed else 0.0)
            for key, value in self._work.items()
        }
        io_rates = {
            key: (value / elapsed if value is not None and elapsed else None)
            for key, value in io_delta.items()
        }
        io_rates_explicit = {
            "read_bytes_per_second": io_rates["read_bytes"],
            "write_bytes_per_second": io_rates["write_bytes"],
            "read_operations_per_second": io_rates["read_count"],
            "write_operations_per_second": io_rates["write_count"],
        }
        throughput_explicit = {
            "objects_per_second": throughput.get("objects", 0.0),
            "variants_per_second": throughput.get("variants", 0.0),
            "algorithm_calls_per_second": throughput.get("algorithm_calls", 0.0),
        }
        return {
            "duration_seconds": elapsed,
            "resources": {
                "cpu": cpu,
                "process_cpu_time": process_cpu_time,
                "memory": {"rss": rss, "peak_rss": self._peak_rss},
                "io": io_delta,
                "io_rates": io_rates,
                "rates": io_rates_explicit,
                "capabilities": resources["capabilities"],
            },
            "latency": latency,
            "tasks": dict(self._tasks),
            "work": dict(self._work),
            "throughput": throughput,
            "throughput_rates": throughput_explicit,
            "cache": dict(self._cache),
            "serial_lane_calls": self._serial_calls,
            "errors": list(self._errors),
        }

    def finish(self, **payload: Any) -> dict[str, Any]:
        if not self.enabled:
            return {}
        summary = self.summary()
        summary.update(payload)
        if not self._finished:
            self.event("run_finished", summary=summary)
            self._finished = True
        if self.output_path:
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning("Unable to write telemetry summary: %s", exc)
        return summary
