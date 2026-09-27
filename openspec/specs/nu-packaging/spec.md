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

Система SHALL запускать проверку изменений при открытии или обновлении PR в `develop` и при push в `main` или `master` (включая merge изменений из `develop`), а также по ручному `workflow_dispatch`. Полная Linux/Windows матрица тестов и сборки SHALL выполняться, если изменён хотя бы один файл за пределами `openspec/`; при изменениях только в `openspec/` build matrix SHALL NOT создаваться. Отдельная стабильная проверка `required-ci` SHALL запускаться всегда, завершаться успешно для OpenSpec-only изменений и требовать успешной матрицы для любых изменений вне `openspec/`; ошибка определения изменённых файлов SHALL блокировать merge. Ruleset ветки `develop` SHALL требовать check `required-ci` вместо отдельных checks matrix jobs. Ручной `workflow_dispatch` SHALL всегда запускать полную матрицу. Push в `develop` или `feature/*` SHALL NOT самостоятельно запускать workflow: commits feature-ветки SHALL проверяться через события PR, включая новые commits (`synchronize`), чтобы один push в открытую PR-ветку не создавал дублирующую CI-матрицу. Перед сборкой SHALL выполняться проверка качества на зафиксированной версии `ruff` (не плавающий `latest`), и SHALL запускаться тесты с маркировкой `not slow and not gui`. Артефакты собранного бинарника SHALL загружаться для каждого runner-а. Шаги workflow SHALL использовать actions на рантайме `node24` (например, `actions/checkout@v7`, `actions/upload-artifact@v7`), Linux-runner SHALL включать `ccache` среди системных зависимостей, а перед checkout SHALL выполняться `git config --global init.defaultBranch main`, чтобы сборка SHALL завершаться без deprecation-warning Node.js 20, без `Nuitka-Scons: not using ccache` и без git-hint про `master`.

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

#### Scenario: Feature PR pushes trigger exactly one matrix
- **WHEN** commit отправляется в `feature/*` с открытым PR в `develop`
- **THEN** для этого commit SHALL запускаться одна CI-матрица по событию PR и SHALL NOT запускаться вторая матрица по событию `push`

#### Scenario: Merge from develop into the primary branch triggers CI
- **WHEN** изменения из `develop` попадают в `main` или `master`
- **THEN** SHALL запускаться CI-матрица `ubuntu-latest` и `windows-latest`

#### Scenario: Feature branch can be checked before opening a PR
- **WHEN** пользователь вручную запускает workflow через `workflow_dispatch` для feature-ветки
- **THEN** SHALL запускаться та же CI-матрица Linux/Windows

#### Scenario: OpenSpec-only changes skip expensive matrix without blocking merge
- **WHEN** все изменения PR или push находятся только в `openspec/`
- **THEN** Linux/Windows matrix jobs SHALL NOT создаваться, а обязательная проверка `required-ci` SHALL завершиться успешно

#### Scenario: Changes outside OpenSpec run the full matrix
- **WHEN** PR или push включает изменение хотя бы одного файла вне `openspec/`
- **THEN** SHALL выполняться полные lint, test, Linux/Windows packaging и packaged CPU smoke checks, и `required-ci` SHALL требовать успешное завершение всех matrix jobs

### Requirement: Binary built from clean environment

Система SHALL выполнять сборку Nuitka из изолированного build-окружения `.venv-build`, содержащего только runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`, ONNX Runtime CPU) и инструменты сборки (`nuitka`, `zstandard`), чтобы в `main.dist` не попадали Python-модули dev/ML-пакетов. Окружение сборки SHALL NOT содержать dev/ML-пакеты; библиотека `libscipy_openblas64*.so` является встроенной BLAS-реализацией numpy 2.x и её наличие в `main.dist` допускается. Локальный скрипт `scripts/build_nuitka.sh` и CI (`build.yaml`) SHALL создавать/воссоздавать `.venv-build` командами `uv sync` без extras с последующей установкой `nuitka`/`zstandard`, и запускать Nuitka с флагом `--no-sync`, чтобы `uv run` не удалил эти инструменты.

#### Scenario: Clean build includes CPU ONNX inference but not PyTorch
- **КОГДА** сборка запущена из `.venv-build`, созданного `uv sync` без extras
- **ТОГДА** бинарник SHALL содержать ONNX Runtime CPU и bundled ONNX-модели и SHALL NOT содержать Python-модули `torch`, `torchvision`, `pytest` или `albumentations`

#### Scenario: Clean build excludes dev/ML python modules
- **КОГДА** сборка запущена из `.venv-build`, созданного `uv sync` без extras
- **ТОГДА** бинарник SHALL содержать ONNX Runtime CPU и SHALL NOT содержать Python-модули `torch`, `pytest`, `nuitka` или `albumentations`

#### Scenario: Local and CI builds use clean runtime build env
- **КОГДА** запускается локальная или CI-сборка
- **ТОГДА** сборка SHALL использовать `.venv-build` с runtime-зависимостями и Nuitka-инструментами, SHALL NOT устанавливать PyTorch в него и SHALL NOT изменять другие окружения

#### Scenario: Build script builds from isolated build env
- **КОГДА** `scripts/build_nuitka.sh` запускается в проекте
- **ТОГДА** он SHALL пересоздать чистый `.venv-build` (runtime-зависимости, ONNX Runtime CPU и `nuitka`/`zstandard`) и собрать бинарник именно из него, SHALL NOT устанавливать ML/dev-пакеты, SHALL NOT изменять `.venv`

#### Scenario: Build tools survive uv run sync
- **КОГДА** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **ТОГДА** команде SHALL передаваться флаг `--no-sync`, чтобы вручную установленные `nuitka`/`zstandard` (отсутствующие в lockfile) не были удалены автоматическим `uv sync`

### Requirement: Binary build verification

Система SHALL верифицировать Nuitka-сборку после изменения runtime-зависимостей или состава bundled-моделей: бинарник SHALL быть собран из чистого `.venv-build`, запускаться без ошибки и выполнять bundled CPU ONNX-инференс на тестовом изображении без PyTorch. Размер onefile SHALL измеряться и фиксироваться в документации с учётом ONNX Runtime и bundled-весов; для Linux-сборки с обеими моделями ожидаемый диапазон составляет 155–175 MiB.

#### Scenario: Packaged smoke runs a model without GUI
- **КОГДА** бинарник собран из чистого `.venv-build` и запущен с `--smoke-test-model` для bundled ONNX-модели
- **ТОГДА** приложение SHALL зарегистрировать модель и выполнить CPU-инференс на тестовом изображении без создания GUI и без внешнего Python

#### Scenario: Binary launches successfully
- **КОГДА** бинарник собран из чистого `.venv-build` (только базовые зависимости + инструменты сборки)
- **ТОГДА** команда `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` SHALL не возвращать ошибку за 30 секунд работы

#### Scenario: Binary size baseline is documented
- **КОГДА** выполнена сборка с ONNX Runtime и моделями
- **ТОГДА** фактический размер SHALL быть зафиксирован в документации; контрольное Linux-значение 170,221,056 bytes (162.3 MiB) SHALL попадать в диапазон 155–175 MiB

#### Scenario: Binary size is monitored
- **КОГДА** бинарник собран и упакован в onefile
- **ТОГДА** его размер SHALL быть измерен, проверен относительно диапазона 155–175 MiB для Linux-сборки с ONNX Runtime и двумя bundled-моделями и отражён в документации; размеры других платформ SHALL фиксироваться отдельно

### Requirement: Binary size monitored and reported

Система SHALL отслеживать размер собранного бинарника (`BacteriaAnalyzer` / `BacteriaAnalyzer.exe`) и фиксировать его в документации (`AGENTS.md`) как контрольную метрику, чтобы регрессии размера были заметны.

#### Scenario: Binary size documented in AGENTS.md
- **КОГДА** документация описывает сборку Nuitka
- **ТОГДА** в неё SHALL быть указан ожидаемый размер бинарника и факторы, влияющие на него (чистота runtime-окружения, плагины Qt, кодеки OpenCV)
