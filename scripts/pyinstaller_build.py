import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "scripts" / "bacteria_analyzer.spec"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"

# Определяем Python с --enable-shared (необходимо для PyInstaller)
for candidate in [
    "/tmp/pyinstaller-venv/bin/python3",
    "/usr/sbin/python3.13",
]:
    if Path(candidate).exists():
        PYTHON = candidate
        break
else:
    print("Ошибка: не найден Python с --enable-shared.")
    print("Установи python3-devel или создай venv через system python.")
    sys.exit(1)

if DIST_DIR.exists():
    shutil.rmtree(DIST_DIR)
if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)

cmd = [
    PYTHON,
    "-m",
    "PyInstaller",
    "--clean",
    "--noconfirm",
    str(SPEC_FILE),
]
print(f"Running: {' '.join(cmd)}")
subprocess.check_call(cmd, cwd=ROOT)
