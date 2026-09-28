## 1. Перераспределение зависимостей в pyproject.toml

- [x] 1.1 Убрать `tqdm>=4.65.0` из базовых `dependencies` (runtime его не использует — только `train/augment.py`)
- [x] 1.2 Из `[dev]` extra убрать ML-пакеты: `torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`, `onnxruntime`
- [x] 1.3 `[dev]` extra оставить только: `nuitka`, `zstandard` + ссылка `bacterial-colony-analyzer[test]`
- [x] 1.4 Добавить новый `[full]` extra: ML-пакеты из 1.2 + `tqdm` + ссылка `bacterial-colony-analyzer[dev]`

## 2. Исключение .venv-full из сборки и VCS

- [x] 2.1 `scripts/nuitka_flags.py`: добавить `.venv-full` в список исключаемых каталогов
- [x] 2.2 `.gitignore`: добавить строку `.venv-full/`
- [x] 2.3 Обновить спеки: в `openspec/specs/nu-packaging/spec.md` список исключений дополнить `.venv-full` (синхронизация delta-спеки)

## 3. CI

- [x] 3.1 `.github/workflows/build.yaml`: заменить `uv sync --extra test` на `uv sync --extra dev`
- [x] 3.2 Убрать из CI шаг «Install Nuitka» (`uv pip install nuitka zstandard`) — эти пакеты уже в `dev`-extra
- [x] 3.3 Убедиться, что ruff-шаг и pytest (`-m "not slow and not gui"`) остаются рабочими с `--extra dev`
- [x] 3.4 Убедиться, что сборка бинарника в CI идёт из чистого `.venv` (добавлен шаг с `.venv-build`)

## 4. Документация

- [x] 4.1 `AGENTS.md`: таблица окружений → три строки (`.venv`, `.venv-dev`, `.venv-full`) с командами, назначением, размерами
- [x] 4.2 `AGENTS.md`: раздел «Сборка» — список исключённых каталогов дополнить `.venv-full`
- [x] 4.3 `AGENTS.md`: раздел «Конвенции» — «Два виртуальных окружения» → «Три виртуальных окружения»
- [x] 4.4 `AGENTS.md`/`README.md`: предупреждение о пересоздании старых `.venv`/`.venv-dev` после обновления
- [x] 4.5 `AGENTS.md`: раздел «Сборка» — указать, что бинарник собирается из чистого runtime-окружения, и зафиксировать ожидаемый размер бинарника с факторами влияния

## 5. Верификация

- [x] 5.1 Локально пересоздать окружения: `uv sync`, `uv sync --extra dev` (в `.venv-dev`), `uv sync --extra full` (не создан — требуется torch ~5 ГБ)
- [x] 5.2 Проверить состав: в `.venv` нет `tqdm`/`torch`/`pytest`, в `.venv-dev` нет `torch`
- [x] 5.3 Прогнать `pytest` — 130/130 тестов пройдены (все маркеры)
- [x] 5.4 Прогнать `ruff check .` и `ruff format --check .`
- [x] 5.5 Проверить `openspec validate --all` после синхронизации спек
- [x] 5.6 Собрать бинарник из чистого `.venv` — успешно, onefile 86 МБ; `libscipy_openblas64` — встроенная BLAS numpy 2.x, не от scipy

## 6. Верификация после изменения runtime-зависимостей

- [x] 6.1 Runtime `.venv` (475 МБ) содержит только PyQt6, opencv-python-headless, numpy
- [x] 6.2 `.venv-dev` (531 МБ) содержит pytest + nuitka + zstandard, без torch
- [x] 6.3 `pytest tests/` — 130/130 тестов пройдено (все маркеры: gui, slow, hypothesis)
- [x] 6.4 Сборка Nuitka из чистого `.venv` — успешна, onefile 86 МБ
- [x] 6.5 Бинарник запускается: `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` — без ошибок
- [x] 6.6 В main.dist нет python-модулей scipy/onnxruntime/torch/pytest/nuitka; `libscipy_openblas64` — встроенная BLAS numpy, допускается

## 7. Изоляция сборки в .venv-build (дополнение по результатам аудита 09.08.2026)

- [x] 7.1 Аудит: локальная сборка `build_nuitka.sh` использовала `.venv` и доустанавливала в него `nuitka`/`zstandard` — загрязняла runtime (фактически `.venv` содержал nuitka+zstandard)
- [x] 7.2 `scripts/build_nuitka.sh`: сборка переведена в изолированный `.venv-build` (пересоздание через `uv sync` + `uv pip install --python .venv-build nuitka zstandard`), запуск Nuitka через `UV_PROJECT_ENVIRONMENT=.venv-build uv run --no-sync`
- [x] 7.3 CI `build.yaml`: исправлен шаг создания `.venv-build` (`uv pip install` теперь с `--python .venv-build`, а не `UV_PROJECT_ENVIRONMENT`); шаги сборки Linux/Windows переведены на `uv run --no-sync` (иначе `uv sync` удалит вручную установленные nuitka/zstandard)
- [x] 7.4 `.venv` пересоздан чистым (`uv sync`): только PyQt6, opencv-python-headless, numpy — nuitka/zstandard/pytest/torch отсутствуют
- [x] 7.5 Проверено: `uv run` (без `--no-sync`) удаляет nuitka/zstandard из `.venv-build`; `uv run --no-sync` сохраняет
- [x] 7.6 `scripts/nuitka_flags.py`: добавлен `.venv-build` в список исключений
- [x] 7.7 Delta-спеки `development-environments` и `nu-packaging` дополнены требованием «Isolated build environment»; main-спека `development-environments` создана, `nu-packaging` обновлена
- [x] 7.8 `AGENTS.md`, `README.md`: добавлена строка `.venv-build` в таблицу окружений, раздел о минимальном окружении сборки
- [x] 7.9 `openspec validate --all` — 10/10 валидны
