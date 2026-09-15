#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# PyInstaller требует Python, собранный с --enable-shared.
# В Fedora/RHEL это /usr/sbin/python3.13, в других дистрах может быть другое место.
SYSTEM_PYTHON=""
for candidate in /usr/sbin/python3.1[3-9] /usr/bin/python3.1[3-9] /usr/local/bin/python3.1[3-9]; do
    if [ -x "$candidate" ] && "$candidate" -c "import sysconfig; exit(0 if sysconfig.get_config_var('Py_ENABLE_SHARED') else 1)" 2>/dev/null; then
        SYSTEM_PYTHON="$candidate"
        break
    fi
done

if [ -z "$SYSTEM_PYTHON" ]; then
    echo "Ошибка: не найден Python с --enable-shared."
    echo ""
    echo "Fedora:  sudo dnf install python3-devel"
    echo "Ubuntu:  sudo apt install python3-dev"
    echo "macOS:   brew install python@3.13"
    echo ""
    echo "Или собери Python вручную:"
    echo "  ./configure --enable-shared --prefix=/opt/python3.13"
    echo "  make -j\$(nproc) && sudo make install"
    exit 1
fi

echo "=== Использую Python: $SYSTEM_PYTHON ==="
echo "=== Создаю временное окружение ==="

TMP_VENV=$(mktemp -d)
"$SYSTEM_PYTHON" -m venv "$TMP_VENV"
source "$TMP_VENV/bin/activate"

echo "=== Устанавливаю зависимости ==="
pip install -q pyinstaller
pip install -q PyQt6 opencv-python-headless numpy albumentations tqdm onnxruntime
pip install -q -e .

echo "=== Запуск PyInstaller ==="
python scripts/pyinstaller_build.py

echo "=== Очистка ==="
deactivate
rm -rf "$TMP_VENV"

echo "=== Готово ==="
ls -lh dist/BacteriaAnalyzer
