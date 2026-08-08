## Context

Сборка бинарника (Nuitka) не имеет спеки в `openspec/specs/` — необходимо обратной документации. Конфигурация разбросана: `scripts/build_nuitka.sh` (локальная обёртка), `scripts/nuitka_build.py` (генератор флагов + вызов Nuitka), `scripts/nuitka_flags.py` (генератор флагов `--include-package`), `.github/workflows/build.yaml` (CI-матрица). Расхождение: Windows-CI жёстко задаёт `--jobs=2`, локально и Linux-CI используют дефолт Nuitka (все ядра). Сессия утверждает опцию «все ядра с верхней границей 8» и возможность задать через env `NUITKA_JOBS`.

## Goals / Non-Goals

**Goals:**
- Специфицировать `nu-packaging`: команды сборки, обязательные флаги, единый источник флагов, параметризация `--jobs`, CI-матрица и quality gates.
- Синхронизировать код (build-скрипты и CI) с спекой: убрать `--jobs=2` из Windows-CI, добавить вычисление `N = max(1, min(cores, 8))` в `nuitka_build.py`, поддержать `NUITKA_JOBS`.
- Документировать фиксацию ruff в CI (не плавающий latest).

**Non-Goals:**
- Не ускоряем сам компилятор (LTO выкл/вкл не рассматриваем).
- Не переписываем PyInstaller-флоу (`scripts/pyinstaller_build.py`, `build_pyinstaller.sh`) — он вне этой сферы (спека Nuitka).
- Не меняем структуру CLI Nuitka.

## Decisions

1. **Вычисление `--jobs` в `nuitka_build.py`** — единственная точка логики: 
   `N = int(os.environ.get("NUITKA_JOBS") or 0)`; если `0` → `N = max(1, min(os.cpu_count() or 1, 8))`. Все сборки (локальная и CI) передают `--jobs=N` через скрипт, а не вручную в рабочих файлах/CI.

2. **Убрать `--jobs=2` из Windows-CI (`build.yaml`)** — заменяем на единый скриптовый путь (перефлаг генерится). Windows-CI по умолчанию получит `--jobs` из ядер (GitHub windows-runner = 4 vCPU → `min(4,8)=4`).

3. **`nuitka_flags.py` остаётся единственным генератором** флагов пакетов (для тестов и диагностики), `nuitka_build.py` его использует или дублирует логику вызова — решаем: держим один источник (`nuitka_flags.py`) и `nuitka_build.py` импортирует его.

4. **Фиксация ruff в CI**: `uv pip install ruff==0.15.20` (уже есть); специфицируем это требование в quality gate (не latest).

## Risks / Trade-offs

- **[верхняя граница 8]** — на CI-раннерах ≤8 ядер loss функции нет; на мощных машинах (64 cores) сборка ограничится 8, что не замедление, а защита памяти. → *принимаем; Имагент override через `NUITKA_JOBS`.*
- **[CUDA/тайм]** — `os.cpu_count` может возвращать `None` → плюс `or 1`.
- **[переменная `NUITKA_JOBS`]** — если задана нецелым числом, парсинг `int()` выбросит; скрипт SHALL логировать и падать с понятной ошибкой.

## Migration Plan

1. Внести спекы и артефакты change.
2. Обновить `nuitka_build.py`: вычисление `--jobs`, добавление `--jobs` в `cmd`.
3. Обновить `.github/workflows/build.yaml`: убрать явный `--jobs=2` (Windows).
4. Обновить `build.yaml` шаг quality: оставить `ruff==0.15.20` (фиксация уже сделана).
5. Тест: `test_nuitka_flags.py`/`test_nuitka_build.py` — проверить вычисление `--jobs` (mock `os.cpu_count`, `NUITKA_JOBS`).
6. `openspec archive nu-packaging-build` (sync main specs) + validate --all.

## Open Questions

- Нужен ли out-of-scope PyInstaller spec? Не в этой change (оставим как историческая утилита).