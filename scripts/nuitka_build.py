import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDE = {
    ".venv",
    ".venv-dev",
    "__pycache__",
    "bacterial_colony_analyzer.egg-info",
    ".git",
    ".github",
    "scripts",
    "test_images",
    "test_data",
    "train",
}

packages = sorted(
    e.name
    for e in ROOT.iterdir()
    if e.is_dir()
    and e.name not in EXCLUDE
    and not e.name.startswith(".")
    and (e / "__init__.py").exists()
)

flags = [f"--include-package={p}" for p in packages]
flags.append("--enable-plugin=pyqt6")

cmd = [sys.executable, "-m", "nuitka", *flags, *sys.argv[1:]]
print(f"Running: {' '.join(cmd)}")
subprocess.check_call(cmd)
