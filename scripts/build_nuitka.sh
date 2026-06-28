#!/usr/bin/env bash
set -euo pipefail

# Сборка BacteriaAnalyzer через Nuitka (один бинарник)
# Запуск: bash scripts/build_nuitka.sh
# Результат: ./BacteriaAnalyzer (~120–150 МБ)

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
uv run nuitka \
  --standalone \
  --onefile \
  --show-progress \
  --enable-plugin=pyqt6 \
  --include-package=ui \
  --include-package=analysis \
  --include-package=utils \
  --include-package=labeling \
  --include-module=cv2 \
  --include-module=albumentations \
  --include-module=tqdm \
  --output-filename=BacteriaAnalyzer \
  main.py

echo "=== Готово ==="
ls -lh BacteriaAnalyzer
