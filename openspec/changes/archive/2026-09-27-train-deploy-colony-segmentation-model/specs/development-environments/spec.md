## MODIFIED Requirements

### Requirement: Three-tier environment structure

Система SHALL поддерживать три виртуальных окружения, каждое со строго определённым составом пакетов и назначением:

- `.venv` — runtime: `PyQt6`, `opencv-python-headless`, `numpy`, `onnxruntime` (CPU provider) и их runtime-зависимости. Создаётся командой `uv sync`.
- `.venv-dev` — всё из `.venv` + pytest-стек + инструменты сборки (`nuitka`, `zstandard`), без PyTorch и пакетов обучения. Создаётся командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`.
- `.venv-full` — всё из `.venv-dev` + ML-стек обучения (`torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`). Создаётся командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`.

#### Scenario: Runtime environment contains inference runtime but not training stack
- **WHEN** создано окружение `.venv` командой `uv sync`
- **THEN** в нём SHALL присутствовать `PyQt6`, `opencv-python-headless`, `numpy` и `onnxruntime` CPU, SHALL NOT присутствовать `torch`, `pytest` или `nuitka`

#### Scenario: Runtime environment contains only app dependencies
- **WHEN** создано окружение `.venv` командой `uv sync`
- **THEN** в нём SHALL присутствовать `PyQt6`, `opencv-python-headless`, `numpy` и `onnxruntime` CPU, и SHALL NOT присутствовать `torch`, `pytest`, `nuitka`, `tqdm`

#### Scenario: Dev environment includes tests and CPU inference
- **WHEN** создано окружение `.venv-dev` командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`
- **THEN** в нём SHALL присутствовать `pytest`, `nuitka`, `zstandard` и базовые runtime-зависимости, включая `onnxruntime`, SHALL NOT присутствовать `torch`, `torchvision` или `albumentations`

#### Scenario: Dev environment without ML stack
- **WHEN** создано окружение `.venv-dev` командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`
- **THEN** в нём SHALL присутствовать `pytest`, `nuitka`, `zstandard` и ONNX Runtime CPU, SHALL NOT присутствовать `torch`, `torchvision` или `albumentations`

#### Scenario: Full environment includes training stack
- **WHEN** создано окружение `.venv-full` командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`
- **THEN** в нём SHALL присутствовать `onnxruntime`, `torch`, `torchvision`, `albumentations`, `plotly`, `tensorboard` и `fastapi`

#### Scenario: Full environment includes ML stack
- **WHEN** создано окружение `.venv-full` командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`
- **THEN** в нём SHALL присутствовать `torch`, `torchvision`, `albumentations`, `onnxruntime`, `plotly`, `tensorboard`, `fastapi`

### Requirement: ML stack isolated in full extra

Система SHALL определять только обучение/тяжёлые ML-пакеты (`torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`) в `full`-extra. `onnxruntime` SHALL быть CPU runtime-зависимостью базового приложения, необходимой для запуска bundled ONNX-моделей, и SHALL NOT требовать установки PyTorch.

#### Scenario: Runtime inference does not install PyTorch
- **WHEN** пользователь устанавливает базовые зависимости командой `uv sync`
- **THEN** SHALL быть установлен `onnxruntime` CPU, но SHALL NOT устанавливаться `torch`/`torchvision`/`albumentations`

#### Scenario: Dev extra excludes training packages
- **WHEN** установлены зависимости через `--extra dev`
- **THEN** пакеты `torch`, `torchvision`, `albumentations` SHALL NOT устанавливаться, а `onnxruntime` CPU SHALL быть доступен как runtime dependency

#### Scenario: Full extra adds training packages on top of runtime
- **WHEN** пользователь устанавливает `.venv-full` с `--extra full`
- **THEN** SHALL быть установлены и runtime ONNX Runtime, и полный PyTorch training stack

#### Scenario: Full extra includes dev tooling and ML stack
- **WHEN** пользователь устанавливает `.venv-full` с `--extra full`
- **THEN** SHALL установиться pytest-стек с инструментами сборки, runtime ONNX Runtime и полный ML stack

### Requirement: Runtime environment purity

Система SHALL гарантировать, что runtime-окружение `.venv` содержит только базовые зависимости приложения (`PyQt6`, `opencv-python-headless`, `numpy`, `onnxruntime` CPU) и их runtime-зависимости, но не пакеты уровней test/dev/full. `.venv` SHALL создаваться командой `uv sync` без extras; установка дополнительных dev/training-пакетов в `.venv` вручную SHALL NOT допускаться.

#### Scenario: Fresh sync produces lean inference runtime
- **WHEN** пользователь удалил `.venv` и выполнил `uv sync` заново
- **THEN** в окружении SHALL присутствовать ONNX Runtime CPU и базовые пакеты приложения, SHALL NOT присутствовать `torch`, `pytest`, `nuitka` или `albumentations`

#### Scenario: Fresh sync produces lean runtime env
- **WHEN** пользователь удалил `.venv` и выполнил `uv sync` заново
- **THEN** в окружении SHALL присутствовать `PyQt6`, `opencv-python-headless`, `numpy` и ONNX Runtime CPU, SHALL NOT присутствовать `scipy`, `pytest`, `nuitka`, `albumentations` или `torch`

#### Scenario: Manual install into runtime env rejected by docs
- **WHEN** документация описывает установку пакета для разработки или обучения
- **THEN** SHALL предлагаться `.venv-dev` или `.venv-full`, а не ручная установка dev/training-пакетов в `.venv`

### Requirement: Isolated build environment for binary builds

Система SHALL собирать Nuitka-бинарник из изолированного `.venv-build`, содержащего runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`, `onnxruntime` CPU) плюс `nuitka`/`zstandard`. `.venv-build` SHALL создаваться через `uv sync` без extras и последующую установку инструментов сборки; локальный build-скрипт и CI SHALL использовать именно это окружение, не загрязняя `.venv`/`.venv-dev`/`.venv-full`.

#### Scenario: Local build uses isolated inference-capable build env
- **WHEN** пользователь запускает `scripts/build_nuitka.sh`
- **THEN** скрипт SHALL создать чистый `.venv-build` с runtime ONNX Runtime CPU и инструментами сборки, SHALL NOT устанавливать PyTorch и SHALL NOT изменять `.venv`

#### Scenario: Local build uses isolated build env
- **WHEN** пользователь запускает `scripts/build_nuitka.sh`
- **THEN** скрипт SHALL создать чистый `.venv-build` с runtime-зависимостями, ONNX Runtime CPU и `nuitka`/`zstandard`, SHALL NOT изменять `.venv`

#### Scenario: Build and VCS exclude all virtual environments
- **WHEN** выполняются `nuitka_flags.py` и `git status`
- **THEN** `.venv`, `.venv-dev`, `.venv-full` и `.venv-build` SHALL NOT попадать в package flags и SHALL NOT отображаться как untracked

#### Scenario: venv and build dirs excluded from build and git
- **WHEN** выполняются `nuitka_flags.py` и `git status`
- **THEN** `.venv`, `.venv-dev`, `.venv-full`, `.venv-build` SHALL NOT попадать в package flags и SHALL NOT отображаться как untracked

#### Scenario: uv preserves build tools
- **WHEN** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **THEN** SHALL использоваться `--no-sync`, чтобы вручную установленные `nuitka`/`zstandard` не удалялись

#### Scenario: uv run preserves build tools in build env
- **WHEN** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **THEN** SHALL использоваться `--no-sync`, чтобы вручную установленные `nuitka`/`zstandard` не удалялись

### Requirement: Minimal binary footprint

Система SHALL собирать финальный бинарник из изолированного `.venv-build`, содержащего только базовые runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`, ONNX Runtime CPU) и инструменты сборки (`nuitka`, `zstandard`), чтобы в binary не попадали пакеты full/dev-окружений. Бинарник SHALL включать ONNX Runtime CPU provider и `.onnx`-модели из `models/`, но SHALL NOT включать Python-модули `torch`, `torchvision`, `albumentations`, `tensorboard` или веса `.pt`.

#### Scenario: Binary built from clean build env
- **WHEN** Nuitka-сборка выполняется из `.venv-build` с runtime-зависимостями и ONNX Runtime CPU
- **THEN** бинарник SHALL содержать приложения, runtime и bundled ONNX assets, но SHALL NOT содержать Python-модули `torch`, `onnxruntime`-training dependencies, `pytest` или PyTorch

#### Scenario: Build env does not install dev packages
- **WHEN** выполняется `scripts/build_nuitka.sh` или CI-сборка
- **THEN** в `.venv-build` SHALL присутствовать runtime ONNX Runtime CPU и инструменты сборки, SHALL NOT присутствовать PyTorch/full-extra зависимости

### Requirement: Runtime environment change verification

При изменении состава runtime зависимостей система SHALL проверять чистую установку runtime/dev/build-окружений, прохождение тестов и работу packaged CPU ONNX-инференса. Runtime/build SHALL содержать ONNX Runtime CPU, но SHALL NOT содержать PyTorch или dev-only ML training пакеты.

#### Scenario: All tests pass with minimal runtime env
- **WHEN** создано чистое `.venv` с ONNX Runtime CPU и `.venv-dev` с тестовым стеком
- **THEN** `pytest tests/` SHALL пройти, CPU ONNX Runtime SHALL загружаться, а PyTorch SHALL отсутствовать в runtime/build окружениях

#### Scenario: Binary builds and launches from clean build env
- **WHEN** сборка Nuitka выполнена из `.venv-build` с runtime ONNX Runtime CPU
- **THEN** приложение SHALL запускаться и SHALL выполнить bundled-модель на CPU без PyTorch/CUDA

#### Scenario: No dev/ML packages leak into runtime venv
- **WHEN** выполнен `uv sync` без extras
- **THEN** в `.venv` SHALL присутствовать базовые runtime-пакеты и ONNX Runtime CPU, SHALL NOT присутствовать `scipy`, `pytest`, `nuitka`, `albumentations` или `torch`
