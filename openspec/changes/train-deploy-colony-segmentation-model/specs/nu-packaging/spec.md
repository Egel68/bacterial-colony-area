## MODIFIED Requirements

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
