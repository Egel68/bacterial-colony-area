#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Проверка зависимостей ==="
if ! command -v g++ &>/dev/null; then
  echo "Ошибка: установи gcc-c++ (sudo dnf install gcc-c++ patchelf python3-devel)"
  exit 1
fi

# Сборка идёт из изолированного build-окружения .venv-build,
# чтобы НЕ загрязнять .venv / .venv-dev dev- и ML-пакетами.
# .venv-build содержит только runtime-зависимости + инструменты сборки (nuitka, zstandard).
echo "=== Создание чистого build-окружения .venv-build ==="
rm -rf .venv-build
UV_PROJECT_ENVIRONMENT=.venv-build uv sync
UV_PROJECT_ENVIRONMENT=.venv-build uv pip install --python .venv-build nuitka zstandard

echo "=== Запуск Nuitka ==="
UV_PROJECT_ENVIRONMENT=.venv-build uv run --no-sync python scripts/nuitka_build.py \
  --standalone \
  --onefile \
  --show-progress \
  --output-filename=BacteriaAnalyzer \
  main.py

echo "=== Готово ==="
ls -lh BacteriaAnalyzer
