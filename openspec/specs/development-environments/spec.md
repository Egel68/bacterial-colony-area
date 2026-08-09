# development-environments Specification

## Purpose
Трёхуровневая структура виртуальных окружений проекта плюс изолированное build-окружение: `.venv` (runtime), `.venv-dev` (разработка без ML), `.venv-full` (полная разработка с обучением NN), `.venv-build` (ad-hoc сборка бинарника).

## Requirements
### Requirement: Three-tier environment structure

Система SHALL поддерживать три виртуальных окружения, каждое со строго определённым составом пакетов и назначением:

- `.venv` — runtime: только зависимости, необходимые для запуска приложения и сборки финального бинарника (`PyQt6`, `opencv-python-headless`, `numpy`). Создаётся командой `uv sync`.
- `.venv-dev` — разработка без обучения NN: всё из `.venv` + pytest-стек + инструменты сборки (`nuitka`, `zstandard`). Создаётся командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`.
- `.venv-full` — полная разработка: всё из `.venv-dev` + ML-стек (`torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`, `onnxruntime`). Создаётся командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`.

#### Scenario: Runtime environment contains only app dependencies
- **КОГДА** создано окружение `.venv` командой `uv sync`
- **ТОГДА** в нём SHALL присутствовать `PyQt6`, `opencv-python-headless`, `numpy`, и SHALL NOT присутствовать `torch`, `pytest`, `nuitka`, `tqdm`

#### Scenario: Dev environment without ML stack
- **КОГДА** создано окружение `.venv-dev` командой `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`
- **ТОГДА** в нём SHALL присутствовать `pytest`, `nuitka`, `zstandard`, и SHALL NOT присутствовать `torch`, `torchvision`, `albumentations`, `onnxruntime`

#### Scenario: Full environment includes ML stack
- **КОГДА** создано окружение `.venv-full` командой `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full`
- **ТОГДА** в нём SHALL присутствовать `torch`, `torchvision`, `albumentations`, `onnxruntime`, `plotly`, `tensorboard`, `fastapi`

### Requirement: ML stack isolated in full extra

Система SHALL определять ML-зависимости (`torch`, `torchvision`, `tensorboard`, `plotly`, `fastapi`, `uvicorn`, `websockets`, `rich`, `albumentations`, `onnxruntime`) в отдельном `full`-extra в `pyproject.toml`. `dev`-extra SHALL NOT содержать ML-зависимостей и SHALL содержать pytest-стек и инструменты сборки.

#### Scenario: Dev extra excludes training packages
- **КОГДА** установлены зависимости через `--extra dev`
- **ТОГДА** пакеты `torch`, `torchvision`, `albumentations`, `onnxruntime` SHALL NOT устанавливаться

#### Scenario: Full extra includes dev tooling and ML stack
- **КОГДА** установлены зависимости через `--extra full`
- **ТОГДА** установится как pytest-стек с инструментами сборки, так и полный ML-стек

### Requirement: Environment lifecycle documentation

Система SHALL документировать команды создания и назначение всех трёх окружений в `AGENTS.md` и `README.md` (раздел «Окружения»). Документация SHALL указывать размер каждого окружения и для какой задачи оно используется.

#### Scenario: Documentation lists all three environments
- **КОГДА** открыт `AGENTS.md` или `README.md`
- **ТОГДА** раздел «Окружения» SHALL содержать таблицу из трёх строк (`.venv`, `.venv-dev`, `.venv-full`) с командой создания и назначением

#### Scenario: Existing dev environments need recreation
- **КОГДА** пользователь ранее создал `.venv-dev` со старой структурой (включая torch)
- **ТОГДА** документация SHALL предупреждать о необходимости удалить и пересоздать окружения после обновления

### Requirement: Runtime environment purity

Система SHALL гарантировать, что runtime-окружение `.venv` содержит только пакеты из базовых `dependencies` (`PyQt6`, `opencv-python-headless`, `numpy`) и не загрязняется пакетами уровней `test`/`dev`/`full`. `.venv` SHALL создаваться только командой `uv sync` без extras; установка дополнительных пакетов в `.venv` вручную (`uv pip install`) SHALL NOT допускаться для dev/ML-пакетов.

#### Scenario: Fresh sync produces lean runtime env
- **КОГДА** пользователь удалил `.venv` и выполнил `uv sync` заново
- **ТОГДА** в `.venv` SHALL присутствовать только `PyQt6`, `opencv-python-headless`, `numpy` и SHALL NOT присутствовать `scipy`, `onnxruntime`, `pytest`, `nuitka`, `albumentations`, `torch`

#### Scenario: Manual install into runtime env rejected by docs
- **КОГДА** документация описывает, как доустановить пакет в dev-окружение
- **ТОГДА** она SHALL указывать использование `.venv-dev` (`UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev`) вместо ручной установки в `.venv`

### Requirement: Isolated build environment for binary builds

Система SHALL собирать бинарник Nuitka из изолированного build-окружения `.venv-build`, создаваемого ad-hoc и содержащего только runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`) плюс инструменты сборки (`nuitka`, `zstandard`). `.venv-build` SHALL создаваться командой `uv sync` (без extras) с последующей установкой `nuitka`/`zstandard`, и оба способа сборки — `scripts/build_nuitka.sh` (локально) и CI (`build.yaml`) — SHALL использовать именно это окружение, а не загрязнять `.venv`/`.venv-dev`/`.venv-full` dev/ML-пакетами. `.venv-build` SHALL быть исключён из сборки и из системы контроля версий (`.gitignore`).

#### Scenario: Local build uses isolated build env
- **КОГДА** пользователь запускает `scripts/build_nuitka.sh`
- **ТОГДА** скрипт SHALL пересоздать чистый `.venv-build` (runtime-зависимости + `nuitka`/`zstandard`) и собрать бинарник именно из него; `.venv` SHALL NOT быть изменён

#### Scenario: venv and build dirs excluded from build and git
- **КОГДА** выполняются `nuitka_flags.py` и `git status`
- **ТОГДА** `.venv`, `.venv-dev`, `.venv-full`, `.venv-build` SHALL NOT попадать в флаги `--include-package` и SHALL NOT отображаться как untracked

#### Scenario: uv run preserves build tools in build env
- **КОГДА** сборка запускается через `UV_PROJECT_ENVIRONMENT=.venv-build uv run`
- **ТОГДА** SHALL использоваться флаг `--no-sync`, чтобы `uv sync` не удалил вручную установленные `nuitka`/`zstandard` (отсутствующие в lockfile)

### Requirement: Minimal binary footprint

Система SHALL собирать финальный бинарник из изолированного build-окружения `.venv-build`, содержащего только runtime-зависимости (`PyQt6`, `opencv-python-headless`, `numpy`) и инструменты сборки (`nuitka`, `zstandard`), чтобы в `main.dist` не попадали пакеты dev/ML-окружений. Бинарник SHALL содержать только код приложения и библиотеки, реально импортируемые на runtime (PyQt6, opencv-python-headless, numpy).

#### Scenario: Binary built from clean build env
- **КОГДА** сборка Nuitka выполняется из `.venv-build` (только `PyQt6`, `opencv-python-headless`, `numpy`, `nuitka`, `zstandard`)
- **ТОГДА** в `main.dist` SHALL NOT присутствовать python-модули `scipy`, `onnxruntime`, `torch`, `pytest` (их .so/.dist-info); допускается наличие `libscipy_openblas64*.so`, так как это встроенная BLAS-библиотека numpy wheel

#### Scenario: Build env does not install dev packages
- **КОГДА** выполняется `scripts/build_nuitka.sh` или CI-сборка
- **ТОГДА** перед запуском Nuitka в `.venv-build` SHALL присутствовать только базовые зависимости и инструменты сборки (dev/full пакеты инициализируются в `.venv-dev`/`.venv-full`)

### Requirement: New virtual environment excluded from build and VCS

Система SHALL исключать каталог `.venv-build` из Nuitka-сборки (`scripts/nuitka_flags.py`) и из системы контроля версий (`.gitignore`), наравне с `.venv`, `.venv-dev` и `.venv-full`.

#### Scenario: all virtual dirs excluded from build flags
- **КОГДА** **nuitka_flags.py** перебирает каталоги проекта
- **ТОГДА** `.venv`, `.venv-dev`, `.venv-full`, `.venv-build` SHALL NOT попадать в флаги `--include-package`

### Requirement: Runtime environment change verification

После изменения состава базовых зависимостей `dependencies` (удаление `tqdm`, потенциальные будущие изменения) система SHALL проходить верификацию: пересоздание чистого `.venv-build` и сборка бинарника из него, запуск приложения offscreen, прогон полного набора pytest (все маркеры), проверка отсутствия dev/ML-пакетов в `.venv` и `.venv-build`. Только после успешного прохождения верификации изменения SHALL считаться готовыми к коммиту.

#### Scenario: All tests pass with minimal runtime env
- **КОГДА** создано чистое `.venv` (только runtime-зависимости) и `.venv-dev` (с тестовым стеком)
- **ТОГДА** `pytest tests/` (все 130 тестов, любой маркер) SHALL проходить без ошибок

#### Scenario: Binary builds and launches from clean build env
- **КОГДА** сборка Nuitka запущена из `.venv-build`, содержащего только `PyQt6`, `opencv-python-headless`, `numpy` + инструменты сборки (`nuitka`/`zstandard`)
- **ТОГДА** сборка SHALL завершиться успешно, бинарник SHALL запускаться без краша (проверка: `QT_QPA_PLATFORM=offscreen ./BacteriaAnalyzer` не возвращает ошибку за 30 с работы)

#### Scenario: No dev/ML packages leak into runtime venv
- **КОГДА** выполнен `uv sync` без extras
- **ТОГДА** в `.venv/lib/python3.13/site-packages/` SHALL присутствовать только `PyQt6`, `cv2`, `numpy`, `opencv_python_headless.libs`, `numpy.libs` и их dist-info; SHALL NOT присутствовать `scipy`, `onnxruntime`, `torch`, `pytest`, `nuitka`, `albumentations`