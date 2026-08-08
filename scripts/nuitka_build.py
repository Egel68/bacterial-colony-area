import os
import subprocess
import sys
from pathlib import Path

from nuitka_flags import build_flags

MAX_JOBS = 8


def compute_jobs() -> int:
    """Вычисляет число параллельных задач: env NUITKA_JOBS или max(1, min(ядра, 8))."""
    env_value = os.environ.get("NUITKA_JOBS")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            raise SystemExit(
                f"NUITKA_JOBS должен быть целым числом, получено: {env_value!r}"
            )
    available = os.cpu_count() or 1
    return max(1, min(available, MAX_JOBS))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    flags = build_flags(ROOT)
    flags.append(f"--jobs={compute_jobs()}")

    cmd = [sys.executable, "-m", "nuitka", *flags, *sys.argv[1:]]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
