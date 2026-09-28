# nu-packaging Specification

## Purpose
Модификация требования о едином источнике флагов Nuitka: к списку исключаемых каталогов добавляются `.venv-full` и `.venv-build`. Добавление требования о сборке из чистого runtime-окружения для минимального размера бинарника.

## MODIFIED Requirements
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

## ADDED Requirements
### Requirement: Binary built from clean runtime environment

Система SHALL выполнять сборку бинарника Nuitka из изолированного чистого build-окружения `.venv-build`, содержащего только базовые зависимости (`PyQt6`, `opencv-python-headless`, `numpy`) и инструменты сборки (`nuitka`, `zstandard`), чтобы в `main.dist` не попадали python-модули dev/ML-пакетов (`scipy`, `onnxruntime`, `torch` и т.д.). Окружение сборки SHALL NOT содержать эти пакеты; библиотека `libscipy_openblas64*.so` является встроенной BLAS-реализацией numpy 2.x и её наличие в `main.dist` допускается. Локальный скрипт `scripts/build_nuitka.sh` и CI (`build.yaml`) SHALL создавать/воссоздавать `.venv-build` командой `uv sync` без extras с последующей установкой `nuitka`/`zstandard`, и запускать Nuitka с флагом `--no-sync`, чтобы `uv run` не удалил эти инструменты.

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
- **ТОГДА** его размер SHALL быть в диапазоне 85–125 МБ (флуктуации из-за версий numpy/opencv/PyQt6)

### Requirement: Binary size monitored and reported

Система SHALL отслеживать размер собранного бинарника (`BacteriaAnalyzer` / `BacteriaAnalyzer.exe`) и фиксировать его в документации (`AGENTS.md`) как контрольную метрику, чтобы регрессии размера (например, из-за попадания лишних библиотек) были заметны.

#### Scenario: Binary size documented in AGENTS.md
- **КОГДА** документация описывает сборку Nuitka
- **ТОГДА** она SHALL указывать ожидаемый размер бинарника и факторы, влияющие на него (чистота runtime-окружения, плагины Qt, кодеки OpenCV)
