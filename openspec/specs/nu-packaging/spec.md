# nu-packaging Specification

## Purpose
TBD - created by archiving change nu-packaging-build. Update Purpose after archive.
## Requirements
### Requirement: Build entry point and outputs

Система SHALL обеспечивать единый исполняемый файл из точки входа `main.py` через Nuitka. Локальная сборка SHALL запускаться скриптом `scripts/build_nuitka.sh`, который настраивает окружение и вызывает `scripts/nuitka_build.py`. Выходной файл SHALL называться `BacteriaAnalyzer` (Linux) или `BacteriaAnalyzer.exe` (Windows) в корне проекта.

#### Scenario: Build produces executable output
- **КОГДА** установлены `g++`, `patchelf` и пользователь запускает `scripts/build_nuitka.sh`
- **ТОГДА** скрипт SHALL собрать бинарник `BacteriaAnalyzer` в корне проекта

#### Scenario: Output filename configurable
- **КОГДА** сборка передаёт `--output-filename=BacteriaAnalyzer.exe`
- **ТОГДА** итоговый бинарник SHALL получить переданное имя файла

### Requirement: Single source of package flags

Система SHALL формировать флаги `--include-package=<pkg>` и `--enable-plugin=pyqt6` единственным скриптом `scripts/nuitka_build.py` (или `scripts/nuitka_flags.py`), перебирающим пакеты проекта и исключающим: `.venv`, `.venv-dev`, `.venv-full`, `.venv-build`, `__pycache__`, `*.egg-info`, `.git`, `.github`, `scripts`, `test_images`, `test_data`, `train`. Локальные и CI-сборки SHALL NOT дублировать эти флаги вручную.

#### Scenario: Flags generated for project packages
- **КОГДА** `nuitka_build.py` перебирает каталоги проекта
- **ТОГДА** он SHALL вернуть по одному `--include-package=<pkg>` на каждый пакет (например, `analysis`, `ui`) и добавить `--enable-plugin=pyqt6`

#### Scenario: Train directory excluded from build
- **КОГДА** скрипт перебирает каталоги проекта
- **ТОГДА** `train/` SHALL NOT попадать в флаги `--include-package`

#### Scenario: venv directories excluded from build flags
- **КОГДА** скрипт перебирает каталоги проекта
- **ТОГДА** `.venv`, `.venv-dev`, `.venv-full`, `.venv-build` SHALL NOT попадать в флаги `--include-package`

### Requirement: Parallelism configuration via --jobs

Система SHALL передавать Nuitka флаг `--jobs=N` для параллельной C-компиляции, где `N = max(1, min(available_cores, 8))`, а `available_cores` вычисляется через `os.cpu_count()`. Если задана переменная окружения `NUITKA_JOBS`, система SHALL использовать её значение с приоритетом. Жёсткая фиксация `--jobs` в CI-коде (например, `--jobs=2`) SHALL NOT быть.

#### Scenario: Jobs derived from core count
- **КОГДА** сборка запущена на машине с 6 ядрами и `NUITKA_JOBS` не задана
- **ТОГДА** `nuitka_build.py` SHALL передать `--jobs=6` (минимум из 6 и 8)

#### Scenario: NUITKA_JOBS overrides default
- **КОГДА** переменная окружения `NUITKA_JOBS=3` задана явно
- **ТОГДА** `nuitka_build.py` SHALL передать `--jobs=3` независимо от числа ядер

#### Scenario: Core count floor at one
- **КОГДА** у системы только 1 ядро
- **ТОГДА** `N` SHALL быть не меньше 1 (`--jobs=1`)

### Requirement: Standalone onefile output

Система SHALL строить конечный бинарник с одновременным использованием `--standalone` и `--onefile`. Windows-сборка в CI SHALL дополнительно использовать `--msvc=latest`, `--assume-yes-for-downloads`, `--windows-console-mode=disable`, `--windows-icon-from-ico=icon.ico`.

#### Scenario: Linux build flags
- **КОГДА** CI собирает под Linux
- **ТОГДА** сборка SHALL использовать `--standalone --onefile --output-filename=BacteriaAnalyzer`

#### Scenario: Windows build flags
- **КОГДА** CI собирает под Windows
- **ТОГДА** сборка SHALL использовать `--standalone --onefile` с флагами `--msvc=latest`, `--assume-yes-for-downloads`, `--windows-console-mode=disable`, `--windows-icon-from-ico=icon.ico`, `--output-filename=BacteriaAnalyzer.exe`

### Requirement: CI build matrix and quality gates

Система SHALL запускать сборку Nuitka в CI на матрице `ubuntu-latest` и `windows-latest` при push в `develop`/`main`/`feature/*` и при открытии PR в `develop`. Перед сборкой SHALL выполняться проверка качества на зафиксированной версии `ruff` (не плавающий `latest`), и SHALL запускаться тесты с маркировкой `not slow and not gui`. Артефакты собранного бинарника SHALL загружаться для каждого runner-а. Шаги workflow SHALL использовать actions на рантайме `node24` (например, `actions/checkout@v7`, `actions/upload-artifact@v7`), Linux-runner SHALL включать `ccache` среди системных зависимостей, а перед checkout SHALL выполняться `git config --global init.defaultBranch main`, чтобы сборка SHALL завершаться без deprecation-warning Node.js 20, без `Nuitka-Scons: not using ccache` и без git-hint про `master`.

#### Scenario: Verified by ruff and tests before build
- **КОГДА** выполняется шаг качества в `build.yaml`
- **ТОГДА** SHALL использоваться зафиксированная в `build.yaml` версия `ruff` (например, `ruff==0.15.20`) для `check` и `format --check`, после чего запускаться pytest с `-m "not slow and not gui"`

#### Scenario: Output artifact upload
- **КОГДА** сборка обоих runner-ах завершилась успешно
- **ТОГДА** SHALL быть опубликованы артефакты `BacteriaAnalyzer-Linux` и `BacteriaAnalyzer-Windows`

#### Scenario: Node24 actions without deprecation warnings
- **КОГДА** `build.yaml` использует `actions/checkout@v7` и `actions/upload-artifact@v7`
- **ТОГДА** в логах и аннотациях run SHALL NOT быть «Node.js 20 is deprecated» и `[DEP0040] punycode`/`[DEP0169] url.parse`

#### Scenario: ccache installed on linux runner
- **КОГДА** Linux-runner выполняет шаг `Install system dependencies (Linux)`
- **ТОГДА** `ccache` SHALL быть установлен вместе с `g++` и `patchelf`, и лог Nuitka SHALL NOT содержать `Nuitka-Scons:WARNING: not using ccache`

### Requirement: Binary built from clean environment

Система SHALL выполнять сборку бинарника Nuitka из изолированного чистого build-окружения `.venv-build`, содержащего только базовые зависимости (`PyQt6`, `opencv-python-headless`, `numpy`) и инструменты сборки (`nuitka`, `zstandard`), чтобы в `main.dist` не попадали python-модули dev/ML-пакетов (`pyproject.toml extras`). Окружение сборки SHALL NOT содержать dev/ML-пакеты; библиотека `libscipy_openblas64*.so` является встроенной BLAS-реализацией numpy 2.x и её наличие в `main.dist` допускается. Локальный скрипт `scripts/build_nuitka.sh` и CI (`build.yaml`) SHALL создавать/воссоздавать `.venv-build` командами `uv sync` без extras с последующей установкой `nuitka`/`zstandard`, и запускать Nuitka с флагом `--no-sync`, чтобы `uv run` не удалил эти инструменты.

#### Scenario: Clean build excludes dev/ML python modules
- **КОГДА** сборка запущена из `.venv-build`, созданного `uv sync` без extras
- **ТОГДА** в `main.dist` SHALL NOT присутствовать python-каталоги и .dist-info пакетов `scipy`, `onnxruntime`, `torch`, `pytest`, `nuitka`, `albumentations`

#### Scenario: Build script builds from isolated build env
- **КОГДА** `scripts/build_nuitka.sh` запускается в проекте
- **ТОГДА** он SHALL пересоздать чистый `.venv-build` (runtime-зависимости + `nuitka`/`zstandard`) и собрать бинарник именно из него, SHALL NOT устанавливать ML/dev-пакеты, SHALL NOT изменять `.venv`

#### Scenario: Build tools survive uv run sync
- **КОГДА** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **ТОГДА** команде SHALL передаваться флаг `--no-sync`, чтобы вручную установленные `nuitka`/`zstandard` (отсутствующие в lockfile) не были удалены автоматическим `uv sync`

### Requirement: Binary build verification

Система SHALL верифицировать сборку бинарника после изменения runtime-зависимостей: сборка из чистого `.venv-build`, запуск бинарника offscreen не должен завершаться ошибкой, размер onefile-бинарника должен находиться в ожидаемом диапазоне (85–125 МБ).

#### Scenario: Binary launches successfully
- **КОГДА** бинарник собран из чистого `.venv-build` (только базовые зависимости + инструменты сборки)
- **ТОГДА** команда `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` SHALL не возвращать ошибку за 30 секунд работы

#### Scenario: Binary size is monitored
- **КОГДА** бинарник собран и упакован в onefile
- **ТОГДА** его размер SHALL быть в диапазоне 85–125 МБ

### Requirement: Binary size monitored and reported

Система SHALL отслеживать размер собранного бинарника (`BacteriaAnalyzer` / `BacteriaAnalyzer.exe`) и фиксировать его в документации (`AGENTS.md`) как контрольную метрику, чтобы регрессии размера были заметны.

#### Scenario: Binary size documented in AGENTS.md
- **КОГДА** документация описывает сборку Nuitka
- **ТОГДА** в неё SHALL быть указан ожидаемый размер бинарника и факторы, влияющие на него (чистота runtime-окружения, плагины Qt, кодеки OpenCV)

