## Context

CI-сборка Nuitka в `.github/workflows/build.yaml`: матрица `[ubuntu-latest, windows-latest]`, шаги — checkout, uv, системные зависимости, `uv sync --extra dev`, ruff check/fmt, pytest, создание чистого `.venv-build`, сборка Nuitka (`scripts/nuitka_build.py`), загрузка артефактов. Анализ run #31299365276 показал следующие предупреждения:

| Источник | Платформа | Причина |
|---|---|---|
| `actions/checkout@v4`, `actions/upload-artifact@v4` | обе | рантайм Node 20 (deprecated), вынужденно использует Node 24 → `##[warning] Node.js 20 is deprecated`, `[DEP0040] punycode`, `[DEP0169] url.parse` |
| git (внутри checkout) | linux | `hint: init.defaultBranch=master` при `git init` |
| Nuitka-Scons | linux | `Nuitka-Scons:WARNING: not using ccache` (2 раза: main + onefile) |
| pytest | обе | `PytestCollectionWarning: cannot collect test class 'TestDataset'` (класс с `__init__`) |

Ограничения: результат сборки (флаги Nuitka, артефакты, структура `.venv-build`) менять нельзя — только чистота логов.

## Goals / Non-Goals

**Goals:**
- Устранить все 4 класса предупреждений в логах CI без изменения результата сборки.
- Обеспечить, что pytest и полный пайплайн CI завершаются с «0 warning».

**Non-Goals:**
- Не меняем флаги Nuitka, структуру артефактов, версии `ruff`/`nuitka`, не переходим на PySide6, не добавляем новые триггеры, не затрагиваем локальные скрипты сборки (`build_nuitka.sh`).

## Decisions

### D1. Обновить `actions/checkout` и `actions/upload-artifact` на node24-версии

- `actions/checkout@v4` → `@v7` (проверен `using: node24`).
- `actions/upload-artifact@v4` → `@v7` (проверен `using: node24`); параметры `name`, `path`, `if-no-files-found`, `compression-level`, `overwrite` совместимы.

_Альтернатива:_ `@v5`/`@v6` (частично ещё node20) — выбраны последние v7 как полностью устраняющие deprecation Node.js 20 и производные `[DEP0040]`/`[DEP0169]`.

### D2. Подавить git-hint про `init.defaultBranch`

Перед `actions/checkout` добавить отдельный шаг, который выполняет `git config --global init.defaultBranch main` в HOME runner-а. Это устраняет hint «Using 'master' as the name for the initial branch», который checkout генерирует при `git init` для клона с `fetch-depth=1`.

_Альтернатива:_ оставить hint (это не error и не warning, а hint), но требование «идеального лога» включает и его.

### D3. Установить `ccache` на Linux

В шаг `Install system dependencies (Linux)` добавить `ccache` в `apt-get install`. Nuitka автоматически обнаружит `ccache` и перестанет выводить `Nuitka-Scons:WARNING: not using ccache` (дважды: основная и onefile-компиляция). На Windows Nuitka использует свой `clcache`, предупреждение не возникает.

### D4. Устранить `PytestCollectionWarning`

Добавить `__test__ = False` в класс `TestDataset` (`testing/dataset.py`), чтобы pytest не пытался коллектировать его как тестовый класс (у него `__init__`). API и поведение не меняются: `TestDataset()` продолжает использоваться в `runner.py`, `__main__.py` и тестах.

_Альтернатива:_ переименовать класс в `Dataset` — больше правок во всех референсах; `__test__ = False` — минимальное и стандартное решение.

## Risks / Trade-offs

- **R1: node24-actions могут изменить поведение артефактов** → Mitigation: версии v7 проверены на совместимость параметров; после прогона проверить, что артефакты `BacteriaAnalyzer-Linux`/`BacteriaAnalyzer-Windows` загружены.
- **R2: `git config --global` в шаге перед checkout может не подействовать, если checkout использует изолированный HOME** → Mitigation: проверка после прогона; fallback — не критично (hint не является warning).
- **R3: `__test__ = False` может быть воспринято как скрытие тестов** → Mitigation: класс не является тестовым, а рабочим загрузчиком; тесты для него уже покрыты `test_classic_algorithms.py`.
- **R4: полное «0 warning» сложно гарантировать** → Mitigation: финальный прогон + автоматический grep логов на `warning|error|deprecat` как gate.