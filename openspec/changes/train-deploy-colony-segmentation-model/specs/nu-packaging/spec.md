## MODIFIED Requirements

### Requirement: CI build matrix and quality gates

Система SHALL запускать сборку Nuitka в CI на матрице `ubuntu-latest` и `windows-latest` при открытии или обновлении PR в `develop`, при каждом push в `main` или `master` (включая merge изменений из `develop`), а также по ручному `workflow_dispatch`. Push в `develop` или `feature/*` SHALL NOT самостоятельно запускать workflow: commits feature-ветки SHALL проверяться через события PR, включая новые commits (`synchronize`), чтобы один push в открытую PR-ветку не создавал дублирующую CI-матрицу. Перед сборкой SHALL выполняться проверка качества на зафиксированной версии `ruff` (не плавающий `latest`), и SHALL запускаться тесты с маркировкой `not slow and not gui`. Артефакты собранного бинарника SHALL загружаться для каждого runner-а. Шаги workflow SHALL использовать actions на рантайме `node24` (например, `actions/checkout@v7`, `actions/upload-artifact@v7`), Linux-runner SHALL включать `ccache` среди системных зависимостей, а перед checkout SHALL выполняться `git config --global init.defaultBranch main`, чтобы сборка SHALL завершаться без deprecation-warning Node.js 20, без `Nuitka-Scons: not using ccache` и без git-hint про `master`.

#### Scenario: Feature PR pushes trigger exactly one matrix
- **WHEN** commit отправляется в `feature/*` с открытым PR в `develop`
- **THEN** для этого commit SHALL запускаться одна CI-матрица по событию PR и SHALL NOT запускаться вторая матрица по событию `push`

#### Scenario: Merge from develop into the primary branch triggers CI
- **WHEN** изменения из `develop` попадают в `main` или `master`
- **THEN** SHALL запускаться CI-матрица `ubuntu-latest` и `windows-latest`

#### Scenario: Feature branch can be checked before opening a PR
- **WHEN** пользователь вручную запускает workflow через `workflow_dispatch` для feature-ветки
- **THEN** SHALL запускаться та же CI-матрица Linux/Windows

#### Scenario: Verified by ruff and tests before build
- **WHEN** выполняется шаг качества в `build.yaml`
- **THEN** SHALL использоваться зафиксированная в `build.yaml` версия `ruff` (например, `ruff==0.15.20`) для `check` и `format --check`, после чего запускаться pytest с `-m "not slow and not gui"`

#### Scenario: Output artifact upload
- **WHEN** сборка обоих runner-ах завершилась успешно
- **THEN** SHALL быть опубликованы артефакты `BacteriaAnalyzer-Linux` и `BacteriaAnalyzer-Windows`

#### Scenario: Node24 actions without deprecation warnings
- **WHEN** `build.yaml` использует `actions/checkout@v7` и `actions/upload-artifact@v7`
- **THEN** в логах и аннотациях run SHALL NOT быть «Node.js 20 is deprecated» и `[DEP0040] punycode`/`[DEP0169] url.parse`

#### Scenario: ccache installed on linux runner
- **WHEN** Linux-runner выполняет шаг `Install system dependencies (Linux)`
- **THEN** `ccache` SHALL быть установлен вместе с `g++` и `patchelf`, и лог Nuitka SHALL NOT содержать `Nuitka-Scons:WARNING: not using ccache`

### Requirement: Binary built from clean environment

Система SHALL выполнять сборку Nuitka из изолированного build-окружения `.venv-build`, содержащего только runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`, `onnxruntime` CPU) и инструменты сборки (`nuitka`, `zstandard`), чтобы PyTorch, тестовые и training-пакеты не попадали в бинарник. Локальный скрипт и CI SHALL создавать `.venv-build` командами `uv sync` без extras с последующей установкой `nuitka`/`zstandard` и запускать Nuitka с `--no-sync`.

#### Scenario: Clean build includes CPU ONNX inference but not PyTorch
- **WHEN** сборка запущена из `.venv-build`, созданного `uv sync` без extras
- **THEN** бинарник SHALL содержать ONNX Runtime CPU и bundled ONNX-модели и SHALL NOT содержать Python-модули `torch`, `torchvision`, `pytest` или `albumentations`

#### Scenario: Clean build excludes dev/ML python modules
- **WHEN** сборка запущена из `.venv-build`, созданного `uv sync` без extras
- **THEN** бинарник SHALL содержать ONNX Runtime CPU и SHALL NOT содержать Python-модули `torch`, `pytest`, `nuitka` или `albumentations`

#### Scenario: Local and CI builds use clean runtime build env
- **WHEN** запускается локальная или CI-сборка
- **THEN** сборка SHALL использовать `.venv-build` с runtime-зависимостями и Nuitka-инструментами, SHALL NOT устанавливать PyTorch в него и SHALL NOT изменять другие окружения

#### Scenario: Build script builds from isolated build env
- **WHEN** `scripts/build_nuitka.sh` запускается в проекте
- **THEN** он SHALL пересоздать чистый `.venv-build` с runtime-зависимостями, ONNX Runtime CPU и `nuitka`/`zstandard`, SHALL NOT изменять `.venv`

#### Scenario: Build tools survive uv run sync
- **WHEN** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **THEN** команде SHALL передаваться `--no-sync`, чтобы вручную установленные `nuitka`/`zstandard` не были удалены

#### Scenario: Build tools survive uv run sync
- **WHEN** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **THEN** команде SHALL передаваться `--no-sync`, чтобы вручную установленные `nuitka`/`zstandard` не были удалены

### Requirement: Binary build verification

После изменения runtime-зависимостей или состава bundled-моделей система SHALL проверять чистую Nuitka-сборку и запуск packaged-приложения. Проверка SHALL подтверждать запуск бинарника, регистрацию новой ONNX-модели и успешный CPU-инференс на тестовом изображении без PyTorch; размер бинарника SHALL измеряться и отражаться в документации с учётом ONNX Runtime и bundled-весов. Для Linux onefile с обеими ONNX-моделями и текущим чистым runtime build-набором контрольный диапазон составляет 155–175 MiB.

#### Scenario: Packaged smoke runs a model without GUI
- **WHEN** бинарник собран из чистого `.venv-build` и запущен с `--smoke-test-model` для bundled ONNX-модели
- **THEN** приложение SHALL зарегистрировать модель и выполнить CPU-инференс на тестовом изображении без создания GUI и без внешнего Python

#### Scenario: Binary launches successfully
- **WHEN** бинарник собран из чистого `.venv-build`
- **THEN** команда `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` SHALL не возвращать ошибку за 30 секунд работы

#### Scenario: Binary size baseline is documented
- **WHEN** выполнена сборка с ONNX Runtime и моделями
- **THEN** фактический размер SHALL быть зафиксирован в документации; контрольное Linux-значение 170,221,056 bytes (162.3 MiB) SHALL попадать в диапазон 155–175 MiB

#### Scenario: Binary size is monitored
- **WHEN** бинарник собран и упакован в onefile
- **THEN** его размер SHALL быть измерен, проверен относительно диапазона 155–175 MiB для Linux-сборки с ONNX Runtime и двумя bundled-моделями и отражён в документации; размеры других платформ SHALL фиксироваться отдельно
