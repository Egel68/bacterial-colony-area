# development-environments Specification

## Purpose
Трёхуровневая структура виртуальных окружений проекта плюс изолированное build-окружение: `.venv` (runtime), `.venv-dev` (разработка без ML), `.venv-full` (полная разработка с обучением NN), `.venv-build` (ad-hoc сборка бинарника).

## Requirements
### Requirement: Three-tier environment structure

Система SHALL поддерживать три виртуальных окружения, каждое со строго определённым составом пакетов и назначением:

- `.venv` — runtime: `PyQt6`, `opencv-python-headless`, `numpy`, `onnxruntime` (CPU provider) и их runtime-зависимости. Создаётся командой `uv sync`.
- `.venv-dev` — разработка без обучения NN: всё из `.venv` + pytest-стек + инструменты сборки (`nuitka`, `zstandard`). Создаётся командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`.
- `.venv-full` — полная разработка: всё из `.venv-dev` + ML-стек (`torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`). Создаётся командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`.

#### Scenario: Runtime environment contains inference runtime but not training stack
- **КОГДА** создано окружение `.venv` командой `uv sync`
- **ТОГДА** в нём SHALL присутствовать `PyQt6`, `opencv-python-headless`, `numpy` и `onnxruntime` CPU, SHALL NOT присутствовать `torch`, `torchvision`, `pytest`, `nuitka`, `albumentations` или `tqdm`

#### Scenario: Dev environment includes tests and CPU inference
- **КОГДА** создано окружение `.venv-dev` командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`
- **ТОГДА** в нём SHALL присутствовать `pytest`, `nuitka`, `zstandard` и ONNX Runtime CPU, SHALL NOT присутствовать `torch`, `torchvision` или `albumentations`

#### Scenario: Full environment includes ML stack
- **КОГДА** создано окружение `.venv-full` командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`
- **ТОГДА** в нём SHALL присутствовать ONNX Runtime CPU, pytest-стек и инструменты сборки (`nuitka`, `zstandard`), а также `torch`, `torchvision`, `albumentations`, `plotly`, `tensorboard` и `fastapi`

### Requirement: ML stack isolated in full extra

Система SHALL определять только обучение/тяжёлые ML-пакеты (`torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`) в `full`-extra. `onnxruntime` SHALL быть CPU runtime-зависимостью базового приложения, необходимой для запуска bundled ONNX-моделей, и SHALL NOT требовать установки PyTorch.

#### Scenario: Dev extra excludes training packages
- **КОГДА** установлены зависимости через `--extra dev`
- **ТОГДА** пакеты `torch`, `torchvision`, `albumentations` SHALL NOT устанавливаться, а ONNX Runtime CPU SHALL быть доступен как runtime dependency

#### Scenario: Full extra adds training packages on top of runtime
- **КОГДА** пользователь устанавливает `.venv-full` с `--extra full`
- **ТОГДА** SHALL быть установлены runtime ONNX Runtime, pytest-стек с инструментами сборки и полный PyTorch training stack

### Requirement: Environment lifecycle documentation

Система SHALL документировать команды создания и назначение всех трёх окружений в `AGENTS.md` и `README.md` (раздел «Окружения»). Документация SHALL указывать размер каждого окружения и для какой задачи оно используется.

#### Scenario: Documentation lists all three environments
- **КОГДА** открыт `AGENTS.md` или `README.md`
- **ТОГДА** раздел «Окружения» SHALL содержать таблицу из трёх строк (`.venv`, `.venv-dev`, `.venv-full`) с командой создания и назначением

#### Scenario: Existing dev environments need recreation
- **КОГДА** пользователь ранее создал `.venv-dev` со старой структурой (включая torch)
- **ТОГДА** документация SHALL предупреждать о необходимости удалить и пересоздать окружения после обновления

### Requirement: Runtime environment purity

Система SHALL гарантировать, что runtime-окружение `.venv` содержит только базовые зависимости приложения (`PyQt6`, `opencv-python-headless`, `numpy`, ONNX Runtime CPU) и их runtime-зависимости, но не пакеты уровней test/dev/full. `.venv` SHALL создаваться командой `uv sync` без extras; установка дополнительных dev/training-пакетов в `.venv` вручную SHALL NOT допускаться.

#### Scenario: Fresh sync produces lean inference runtime
- **КОГДА** пользователь удалил `.venv` и выполнил `uv sync` заново
- **ТОГДА** в окружении SHALL присутствовать `PyQt6`, `opencv-python-headless`, `numpy` и ONNX Runtime CPU, SHALL NOT присутствовать `scipy`, `torch`, `torchvision`, `pytest`, `nuitka` или `albumentations`

#### Scenario: Manual install into runtime env rejected by docs
- **КОГДА** документация описывает, как доустановить пакет в dev-окружение
- **ТОГДА** она SHALL указывать использование `.venv-dev` (`UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`) вместо ручной установки в `.venv`

### Requirement: Isolated build environment for binary builds

Система SHALL собирать Nuitka-бинарник из изолированного `.venv-build`, содержащего runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`, ONNX Runtime CPU) плюс `nuitka`/`zstandard`. `.venv-build` SHALL создаваться через `uv sync` без extras и последующую установку инструментов сборки; локальный build-скрипт и CI SHALL использовать именно это окружение, не загрязняя `.venv`/`.venv-dev`/`.venv-full`. `.venv-build` SHALL исключаться из Nuitka package flags и системы контроля версий вместе с остальными окружениями.

#### Scenario: Local build uses isolated build env
- **КОГДА** пользователь запускает `scripts/build_nuitka.sh`
- **ТОГДА** скрипт SHALL создать чистый `.venv-build` с runtime-зависимостями, ONNX Runtime CPU и `nuitka`/`zstandard`, SHALL NOT устанавливать PyTorch и SHALL NOT изменять `.venv`

#### Scenario: venv and build dirs excluded from build and git
- **КОГДА** выполняются `nuitka_flags.py` и `git status`
- **ТОГДА** `.venv`, `.venv-dev`, `.venv-full`, `.venv-build` SHALL NOT попадать в флаги `--include-package` и SHALL NOT отображаться как untracked

#### Scenario: uv run preserves build tools in build env
- **КОГДА** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **ТОГДА** SHALL использоваться флаг `--no-sync`, чтобы `uv sync` не удалил вручную установленные `nuitka`/`zstandard` (отсутствующие в lockfile)

### Requirement: Minimal binary footprint

Система SHALL собирать финальный бинарник из изолированного `.venv-build`, содержащего только базовые runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`, ONNX Runtime CPU) и инструменты сборки (`nuitka`, `zstandard`), чтобы в binary не попадали пакеты full/dev-окружений. Бинарник SHALL включать ONNX Runtime CPU provider и `.onnx`-модели из `models/`, но SHALL NOT включать Python-модули `torch`, `torchvision`, `albumentations`, `tensorboard` или веса `.pt`.

#### Scenario: Binary built from clean build env
- **КОГДА** Nuitka-сборка выполняется из `.venv-build` с runtime-зависимостями и ONNX Runtime CPU
- **ТОГДА** бинарник SHALL содержать приложение, runtime и bundled ONNX assets, но SHALL NOT содержать Python-модули `torch`, `pytest` или PyTorch training-зависимости

#### Scenario: Build env does not install dev packages
- **КОГДА** выполняется `scripts/build_nuitka.sh` или CI-сборка
- **ТОГДА** в `.venv-build` SHALL присутствовать runtime ONNX Runtime CPU и инструменты сборки, SHALL NOT присутствовать PyTorch/full-extra зависимости

### Requirement: Runtime environment change verification

При изменении состава runtime-зависимостей система SHALL проверять чистую установку runtime/dev/build-окружений, прохождение тестов и работу packaged CPU ONNX-инференса. Runtime/build SHALL содержать ONNX Runtime CPU, но SHALL NOT содержать PyTorch или dev-only ML training пакеты.

#### Scenario: All tests pass with minimal runtime env
- **КОГДА** создано чистое `.venv` с ONNX Runtime CPU и `.venv-dev` с тестовым стеком
- **ТОГДА** `pytest tests/` SHALL пройти, CPU ONNX Runtime SHALL загружаться, а PyTorch SHALL отсутствовать в runtime/build окружениях

#### Scenario: Binary builds and launches from clean build env
- **КОГДА** сборка Nuitka выполнена из `.venv-build` с runtime ONNX Runtime CPU
- **ТОГДА** приложение SHALL запускаться и SHALL выполнить bundled-модель на CPU без PyTorch/CUDA

#### Scenario: No dev/ML packages leak into runtime venv
- **КОГДА** выполнен `uv sync` без extras
- **ТОГДА** в `.venv` SHALL присутствовать базовые runtime-пакеты и ONNX Runtime CPU, SHALL NOT присутствовать `scipy`, `pytest`, `nuitka`, `albumentations` или `torch`
