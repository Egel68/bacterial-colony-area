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

Система SHALL формировать флаги `--include-package=<pkg>` и `--enable-plugin=pyqt6` единственным скриптом `scripts/nuitka_build.py` (или `scripts/nuitka_flags.py`), перебирающим пакеты проекта и исключающим: `.venv`, `.venv-dev`, `__pycache__`, `*.egg-info`, `.git`, `.github`, `scripts`, `test_images`, `test_data`, `train`. Локальные и CI-сборки SHALL NOT дублировать эти флаги вручную.

#### Scenario: Flags generated for project packages
- **КОГДА** `nuitka_build.py` перебирает каталоги проекта
- **ТОГДА** он SHALL вернуть по одному `--include-package=<pkg>` на каждый пакет (например, `analysis`, `ui`) и добавить `--enable-plugin=pyqt6`

#### Scenario: Train directory excluded from build
- **КОГДА** скрипт перебирает каталоги проекта
- **ТОГДА** `train/` SHALL NOT попадать в флаги `--include-package`

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

Система SHALL запускать сборку Nuitka в CI на матрице `ubuntu-latest` и `windows-latest` при push в `develop`/`main`/`feature/*` и при открытии PR в `develop`. Перед сборкой SHALL выполняться проверка качества на зафиксированной версии `ruff` (не плавающий `latest`), и SHALL запускаться тесты с маркировкой `not slow and not gui`. Артефакты собранного бинарника SHALL загружаться для каждого runner-а.

#### Scenario: Verified by ruff and tests before build
- **КОГДА** выполняется шаг качества в `build.yaml`
- **ТОГДА** SHALL использоваться зафиксированная в `build.yaml` версия `ruff` (например, `ruff==0.15.20`) для `check` и `format --check`, после чего запускаться pytest с `-m "not slow and not gui"`

#### Scenario: Output artifact upload
- **КОГДА** сборка обоих runner-ах завершилась успешно
- **ТОГДА** SHALL быть опубликованы артефакты `BacteriaAnalyzer-Linux` и `BacteriaAnalyzer-Windows`

