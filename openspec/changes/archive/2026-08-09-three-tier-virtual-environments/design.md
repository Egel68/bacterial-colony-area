## Context

Сейчас в проекте два окружения: `.venv` (runtime, базовые зависимости, ~0.7 ГБ) и `.venv-dev` (все зависимости включая torch, ~5 ГБ). Проблема: разработчику, работающему с UI/анализом/разметкой, не нужен ML-стек, но `dev`-extra в `pyproject.toml` тянет `torch`, `torchvision`, `onnxruntime` и прочее. Это увеличивает время установки и занимает диск без пользы для большинства задач.

Текущая структура `pyproject.toml`:
- `dependencies` — PyQt6, opencv-python-headless, numpy, tqdm
- `[test]` extra — pytest-стек
- `[dev]` extra — torch, torchvision, nuitka, zstandard, tensorboard, plotly, fastapi, uvicorn, websockets, rich, albumentations, onnxruntime, и ссылка на `[test]`

Окружения создаются через `uv sync` с переменной `UV_PROJECT_ENVIRONMENT`. Сборка бинарника требует `nuitka` + `zstandard`, которые сейчас лежат в `dev`-extra. CI (`build.yaml`) использует `uv sync --extra test`, а `scripts/build_nuitka.sh` доустанавливает `nuitka`/`zstandard` напрямую.

**Аудит runtime-состава (проведён):** приложение (analysis, ui, utils, labeling, testing, main.py) импортирует только `PyQt6` (QtCore/QtGui/QtWidgets), `cv2`, `numpy`. `tqdm` используется исключительно в `train/augment.py`. Текущий `.venv` загрязнён dev/ML-пакетами (scipy, onnxruntime, albumentations, hypothesis, pytest, nuitka, coverage — 312 dist-info).

**Верификация после изменений (проведена):**
- Runtime `.venv` (475 МБ) содержит только PyQt6, opencv-python-headless, numpy. Без tqdm, scipy, onnxruntime, pytest, nuitka.
- `.venv-dev` (531 МБ) создан с `--extra dev`: pytest + nuitka + zstandard, без torch/ML.
- `pytest tests/` — 130/130 тестов пройдено (все маркеры, включая gui и slow).
- Сборка Nuitka из чистого `.venv` (с доустановкой nuitka/zstandard) — успешна. Onefile-бинарник 86 МБ.
- Запуск `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` — успешен (нет краша за 30 с).
- `libscipy_openblas64*.so` присутствует в `main.dist` — это встроенная BLAS библиотека numpy 2.x, не от scipy. Python-модули scipy/onnx/torch/pytest/nuitka отсутствуют.

## Goals / Non-Goals

**Goals:**
- Три независимых окружения с чётким назначением: runtime, dev без ML, full с ML.
- `dev`-extra становится «лёгким» — pytest + инструменты сборки, без torch.
- ML-зависимости изолированы в `full`-extra.
- Runtime-зависимости минимальны: убрать `tqdm` из `dependencies` (перенести в `full`).
- Runtime-окружение `.venv` чистое — содержит только базовые пакеты; сборка бинарника идёт из него, что минимизирует `main.dist`.
- Документация (AGENTS.md, README.md) описывает все три уровня и размер бинарника.
- `.venv-full` исключён из сборки и git.

**Non-Goals:**
- Не добавляем новые пакеты, не обновляем версии.
- Не оптимизируем сам код/плагины Qt внутри бинарника (QtPdf, кодеки OpenCV) — только исключаем затянутые загрязнённым окружением библиотеки.
- Не меняем логику обучения/аугментации — только размещение окружений.
- Не трогаем архитектуру Nuitka-сборки (только список исключений и чистота окружения сборки).

## Decisions

### D1. Три каталога: `.venv`, `.venv-dev`, `.venv-full`

Сохраняем существующие имена `.venv` и `.venv-dev`, добавляем новый `.venv-full`. Альтернативы: `.venv-ml`, `.venv-train` — отклонены, т.к. `.venv-full` передаёт «полная разработка», а ML — лишь часть. Выбор пользователя подтверждён.

### D2. ML-стек выносится в `full`-extra, `dev`-extra остаётся лёгким

В `pyproject.toml`:
- `dependencies` — только PyQt6, opencv-python-headless, numpy (убираем tqdm).
- `[dev]` — `nuitka`, `zstandard`, ссылка на `[test]` (без torch-пакетов).
- `[full]` — torch, torchvision, tensorboard, plotly, fastapi, uvicorn, websockets, rich, albumentations, onnxruntime, tqdm, ссылка на `[dev]`.

Альтернатива: оставить `dev` как есть и добавить отдельный extra «ML-только» поверх. Отклонено: два команды создания (`--extra dev --extra ml`) сложнее, а единый `full`-extra даёт одношаговое создание полного окружения.

Альтернатива: назвать extra `train`. Отклонено: `full` шире, чем обучение — сюда входят и инференс (onnxruntime), и дашборд (fastapi/plotly).

Альтернатива: оставить `tqdm` в `dependencies`. Отклонено: аудит показал, что runtime его не импортирует — `tqdm` нужен только `train/augment.py`, поэтому его место в `full`.

### D3. CI переводится на `--extra dev`

`build.yaml` использует `uv sync --extra test` для качества/сборки. Обновляем на `--extra dev`: dev-extra включает pytest-стек (через ссылку на `[test]`) и добавляет `nuitka`/`zstandard`, что убирает ручную установку Nuitka в CI. Альтернатива: оставить `--extra test` и продолжать доустанавливать nuitka вручную — отклонено, дублирование зависимостей.

### D4. `.venv-full` исключается из сборки и git

- `scripts/nuitka_flags.py`: добавить `.venv-full` в список исключений.
- `.gitignore`: добавить `.venv-full/`.

### D5. Сборка бинарника из изолированного `.venv-build`

Аудит показал: загрязнённый `.venv` (с установленным scipy) заставляет numpy линковаться с `libscipy_openblas64`, которая затем попадает в `main.dist` (+24 МБ). Чтобы бинарник был минимальным, сборка Nuitka SHALL выполняться из изолированного build-окружения `.venv-build`, создаваемого ad-hoc: `uv sync` без extras + `uv pip install --python .venv-build nuitka zstandard`. Это позволяет не загрязнять `.venv`/`.venv-dev`/`.venv-full` инструментами сборки: `.venv` остаётся чистым runtime (только приложение), `.venv-dev` — без ML, `.venv-full` — полный.

Первоначальный вариант D5 (сборка из чистого `.venv` с доустановкой nuitka/zstandard в него) отклонён после аудита: доустановка в `.venv` загрязняет runtime, а `uv run --no-sync` удаляет вручную установленные пакеты при следующем `uv sync`. Альтернатива: разрешить сборку из `.venv-dev`. Отклонено: dev-extra содержит pytest-стек и прочее, что не нужно в бинарнике и рискует подтянуть лишние библиотеки.

`scripts/build_nuitka.sh` и CI (`build.yaml`) пересоздают `.venv-build`, запускают Nuitka через `UV_PROJECT_ENVIRONMENT=.venv-build uv run --no-sync python scripts/nuitka_build.py`. `.venv-build` исключён из сборки (`nuitka_flags.py`) и из git (`.gitignore`).

### D6. Обновление документации

- `AGENTS.md`: таблица окружений → три строки с командами, назначением и размерами; предупреждение о пересоздании старых окружений.
- `README.md`: синхронизировать команды dev-окружения.
- `AGENTS.md`: раздел «Сборка» — список исключённых каталогов бинарника дополнить `.venv-full`; указать ожидаемый размер бинарника и требование чистого runtime-окружения для сборки.

## Risks / Trade-offs

- **BREAKING для существующих dev-окружений** → После обновления старый `.venv-dev` (с torch) не соответствует новой структуре. Митигация: документация предупреждает удалить `.venv`/`.venv-dev`/`.venv-full` и пересоздать; разницы в коде нет.
- **CI может не заметить новую структуру extras** → Проверка: в CI прогоняются pytest (через `--extra dev`) и сборка; если `dev`-extra ссылается на `[test]` корректно, тесты проходят. Митигация: в задачах — верификация `uv sync --extra dev` локально.
- **Ложное ощущение экономии размера** → `full`-extra всё ещё ~5 ГБ с torch; средний `.venv-dev` станет заметно легче (~1–1.5 ГБ). Риск неверных ожиданий снимаем таблицей размеров в документации.
- **Удаление `tqdm` из runtime-зависимостей может быть замечено как несовместимость** → tqdm используется только в `train/augment.py`, а модуль `train` исключён из бинарника; в `full`-extra tqdm присутствует. Митигация: документируем перенос.
- **Загрязнение `.venv` вручную (`uv pip install`) вернёт scipy_openblas в бинарник** → Митигация: документация запрещает ручную установку в `.venv`; CI-сборка создаёт чистое окружение из `uv.lock`.

## Migration Plan

1. Обновить `pyproject.toml`: убрать `tqdm` из `dependencies`, перераспределить зависимости между `[dev]` и `[full]` (tqdm → `full`).
2. Обновить `scripts/nuitka_flags.py`, `.gitignore`.
3. Обновить `.github/workflows/build.yaml` на `--extra dev`, убрать ручную установку nuitka.
4. Обновить `AGENTS.md`, `README.md`.
5. Верификация: `uv sync`, `uv sync --extra dev`, `uv sync --extra full` (новые окружения), прогон pytest и ruff.
6. Проверка минимальности: после сборки из изолированного `.venv-build` убедиться, что в `main.dist` нет python-модулей scipy/onnx/torch; зафиксировать размер бинарника.
7. После любых изменений runtime-зависимостей повторно выполнить шаги 5–6 (сборка + все тесты + запуск бинарника offscreen).

## Результаты верификации (08.08.2026)

| Проверка | Статус | Детали |
|---|---|---|
| Runtime `.venv` чист | ✓ | 475 МБ, только PyQt6+opencv+numpy (после сборки из `.venv-build` — без nuitka/zstandard) |
| Размер `.venv-dev` (без ML) | ✓ | 531 МБ (было 4.9 ГБ) |
| Все тесты (130) | ✓ | `pytest tests/` 130/130 |
| Сборка Nuitka из `.venv-build` | ✓ | Onefile, 86 МБ (было 327 МБ со старым загрязнённым `.venv`) |
| Запуск бинарника | ✓ | `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` — успех |
| Чистота main.dist | ✓ | Нет python-модулей scipy/onnx/torch/pytest/nuitka; `libscipy_openblas64` — встроенная BLAS numpy |

## Open Questions

- Нужно ли добавлять проверку в CI, что `.venv` не содержит torch? (Сейчас нет тестов на состав окружения — см. секцию 7 AGENTS.md.)
