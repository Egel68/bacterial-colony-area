#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Проверка зависимостей ==="
if ! command -v g++ &>/dev/null; then
  echo "Ошибка: установи gcc-c++ (sudo dnf install gcc-c++ patchelf python3-devel)"
  exit 1
fi

if ! uv run python -c "import nuitka" 2>/dev/null; then
  echo "Устанавливаю Nuitka..."
  uv pip install nuitka zstandard
fi

echo "=== Запуск Nuitka ==="
uv run python scripts/nuitka_build.py \
  --standalone \
  --onefile \
  --show-progress \
  --icon=icon.ico \
  --output-filename=BacteriaAnalyzer \
  main.py

echo "=== Готово ==="
ls -lh BacteriaAnalyzer
