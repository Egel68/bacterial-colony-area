"""Профиль хоста для воспроизводимых замеров производительности (задача 1.1).

Фиксирует характеристики окружения реализации: ОС, CPU (модель, affinity,
квота), RAM, версии OpenCV/ONNX Runtime, идентичность снимков датасета и их
размеры, состояние тёплого кеша файловой системы. Профиль нужен, чтобы
бенчмарки (задача 1.3) и acceptance-замеры (задача 8.2) были привязаны к
конкретному хосту и воспроизводимы.

Использование::

    from analysis.host_profile import capture_host_profile, write_host_profile

    profile = capture_host_profile(sample_ids=[("s1", 5927, 5968)])
    write_host_profile(profile, Path("tests/baseline/host_profile.json"))

Профиль — документ-снимок состояния хоста в момент вызова; `warm_cache` и
`io_counters_available` помечают, какие условия измерены, а какие — нет.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

SCHEMA_VERSION = 1


def _cpu_model() -> Optional[str]:
    """Модель CPU: /proc/cpuinfo (Linux) или platform (fallback)."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _cpu_affinity() -> Optional[list]:
    """Маска CPU-affinity процесса (None, если не поддерживается)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _cpu_quota() -> Optional[dict]:
    """CPU-квота cgroup (None вне Linux или при недоступности)."""
    for version, path in (
        (2, Path("/sys/fs/cgroup/cpu.max")),
        (1, Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")),
    ):
        try:
            if version == 2:
                raw = path.read_text(encoding="utf-8").strip()
                quota, period = raw.split()
                if quota == "max":
                    return {"version": 2, "quota_us": None, "period_us": int(period)}
                return {
                    "version": 2,
                    "quota_us": int(quota),
                    "period_us": int(period),
                }
            quota_us = int(path.read_text(encoding="utf-8").strip())
            period_path = path.parent / "cpu.cfs_period_us"
            period_us = int(period_path.read_text(encoding="utf-8").strip())
            return {"version": 1, "quota_us": quota_us, "period_us": period_us}
        except (OSError, ValueError):
            continue
    return None


def _memory_info() -> Dict[str, Any]:
    """RAM: total/available в байтах (через psutil или /proc/meminfo)."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {"total_bytes": memory.total, "available_bytes": memory.available}
    except (ImportError, AttributeError):
        pass
    info: Dict[str, int] = {}
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields:
                info[key] = int(fields[0]) * 1024  # kB -> bytes
        return {
            "total_bytes": info.get("MemTotal"),
            "available_bytes": info.get("MemAvailable"),
        }
    except (OSError, ValueError):
        return {"total_bytes": None, "available_bytes": None}


def _runtime_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "opencv": None,
        "onnxruntime": None,
        "numpy": None,
        "python": platform.python_version(),
    }
    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except ImportError:
        pass
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    return versions


def _git_revision() -> Optional[str]:
    """HEAD репозитория (None, если git недоступен)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def detect_io_counters() -> Dict[str, Any]:
    """Доступность системных I/O-счётчиков (неизвестное != ноль)."""
    try:
        import psutil

        counters = psutil.disk_io_counters()
        if counters is None:
            return {"available": False, "read_bytes": None}
        return {"available": True, "read_bytes": counters.read_bytes}
    except (ImportError, AttributeError):
        return {"available": False, "read_bytes": None}


def capture_host_profile(
    sample_ids: Optional[Sequence[Tuple[str, int, int]]] = None,
    warm_cache: Optional[bool] = None,
    dataset_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Собирает профиль хоста.

    `sample_ids` — список `(идентификатор, ширина, высота)` замеренных снимков.
    `warm_cache` — состояние тёплого кеша ФС: True/False, либо None,
    если условие не установлено (будет явно помечено как «не измерено»).
    """
    cpu_count = os.cpu_count()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except (ImportError, AttributeError):
        physical = None

    samples = [
        {"id": sample_id, "width": int(width), "height": int(height)}
        for sample_id, width, height in (sample_ids or [])
    ]
    io_counters = detect_io_counters()

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_count": cpu_count,
            "physical_count": physical,
            "affinity": _cpu_affinity(),
            "quota": _cpu_quota(),
        },
        "memory": _memory_info(),
        "runtimes": _runtime_versions(),
        "git_revision": _git_revision(),
        "dataset": {
            "root": dataset_root,
            "samples": samples,
        },
        "warm_cache": warm_cache,
        "io_counters": {
            "available": io_counters["available"],
            "read_bytes_at_capture": io_counters["read_bytes"],
        },
    }


def write_host_profile(
    profile: Dict[str, Any], output_path: Path, indent: int = 2
) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=indent, ensure_ascii=False), encoding="utf-8"
    )


def load_host_profile(input_path: Path) -> Dict[str, Any]:
    import json

    return json.loads(input_path.read_text(encoding="utf-8"))
