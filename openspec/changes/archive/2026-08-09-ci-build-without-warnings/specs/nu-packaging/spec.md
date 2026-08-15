## MODIFIED Requirements

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