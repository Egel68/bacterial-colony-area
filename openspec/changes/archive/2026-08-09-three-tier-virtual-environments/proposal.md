## Why

Сейчас в проекте два окружения: `.venv` (базовые зависимости, ~0.7 ГБ) и `.venv-dev` (всё вместе с torch, ~5 ГБ). Пользователю для собранного бинарника и разработчику без ML-задач не нужны torch/torchvision/onnxruntime и прочие тяжёлые пакеты, но их приходится ставить в единственное dev-окружение. Требуется три уровня окружений: минимальный для runtime/сборки бинарника, средний для разработки без обучения NN и полный для обучения нейросетей.

## What Changes

- **Три виртуальных окружения** вместо двух:
  - `.venv` — runtime: **только** зависимости, реально импортируемые приложением (PyQt6, opencv-python-headless, numpy). `tqdm` отсюда убирается — runtime его не использует.
  - `.venv-dev` — разработка **без** обучения NN: runtime + pytest-стек + инструменты сборки (nuitka, zstandard).
  - `.venv-full` — полная разработка: всё из `.venv-dev` + ML-стек (torch, torchvision, tensorboard, plotly, fastapi, uvicorn, websockets, rich, albumentations, onnxruntime).
- **Реструктуризация extras в `pyproject.toml`**: из текущего `dev`-extra выносится ML-стек в отдельный `full`-extra; `dev`-extra остаётся «лёгким» (тесты + сборка, без torch). Пакет `tqdm` переносится из базовых `dependencies` в `full`-extra (используется только в `train/augment.py`).
- **Чистота runtime-окружения**: `.venv` создаётся только командой `uv sync` без extras и SHALL NOT содержать пакетов уровней `test`/`dev`/`full` (pytest, nuitka, onnxruntime, scipy и т.д.).
- **Минимизация бинарника**: сборка Nuitka выполняется из изолированного build-окружения `.venv-build` (runtime-зависимости + инструменты сборки nuitka/zstandard), чтобы в `main.dist` не попадали зависимости загрязнённого окружения (например, `libscipy_openblas`, подтянутый numpy из-за установленного scipy). Локальный `scripts/build_nuitka.sh` и CI пересоздают `.venv-build` ad-hoc и запускают Nuitka через `uv run --no-sync`, не трогая `.venv`/`.venv-dev`/`.venv-full`.
- **BREAKING**: команда создания dev-окружения меняется — `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev` теперь не тянет torch. Для ML-разработки используется `.venv-full`.
- Обновляются: `AGENTS.md`, `README.md`, `.gitignore` (добавить `.venv-full/`), `scripts/nuitka_flags.py` и спека `nu-packaging` (исключать `.venv-full` из сборки).

## Capabilities

### New Capabilities
- `development-environments`: спецификация трёхуровневой структуры окружений — `.venv` (runtime), `.venv-dev` (разработка без NN), `.venv-full` (полная разработка), включая состав пакетов каждого уровня, команды создания и назначение.

### Modified Capabilities
- `nu-packaging`: требование о списке исключаемых из Nuitka-сборки каталогов обновляется — к `.venv`, `.venv-dev` добавляется `.venv-full`.

## Impact

- **Код**: `pyproject.toml` (extras `dev`/`full`, удаление `tqdm` из `dependencies`), `scripts/nuitka_flags.py` (исключить `.venv-full`), `.github/workflows/build.yaml` (CI использует `--extra dev` для тестов/сборки).
- **Документация**: `AGENTS.md`, `README.md` (раздел «Окружения» — три уровня, таблица команд).
- **Конфигурация**: `.gitignore` (новый каталог `.venv-full/`).
- **Спеки**: новая `development-environments`, модификация `nu-packaging`.
- **Бинарник**: сборка из изолированного `.venv-build` исключает лишние библиотеки (scipy/openblas), что уменьшает `main.dist`.
- **Риск**: существующие dev-окружения (`.venv-dev` с torch) несовместимы с новой структурой — их нужно пересоздать; CI-пайплайн переводится на `--extra dev`.
