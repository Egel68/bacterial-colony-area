## 1. Спека `nu-packaging` (артефакты change)

- [x] 1.1 proposal.md: цель, Why/What, Impact (новый capability `nu-packaging`)
- [x] 1.2 design.md: вычисление `--jobs`, единый источник флагов, риски
- [x] 1.3 Spec: 6 требований (entry point, флаги, `--jobs`, standalone/onefile, CI-матрица, quality gates)
- [x] 1.4 `openspec validate --type change nu-packaging-build` → valid

## 2. Реализация: вычисление `--jobs` в скрипте сборки

- [x] 2.1 `scripts/nuitka_build.py` — добавить `--jobs=N` с `N = max(1, min(os.cpu_count(), 8))` и приоритет env `NUITKA_JOBS`
- [x] 2.2 `.github/workflows/build.yaml` — убрать жёсткий `--jobs=2` из Windows-шага (передаётся скриптом)
- [x] 2.3 Убедиться, что локальный `build_nuitka.sh` и Linux-CI идут через `nuitka_build.py` (единый путь)

## 3. Тесты сценариев

- [x] 3.1 Unit-тест вычисления `--jobs`: mock `os.cpu_count()` (6→`--jobs=6`), флор-кейс (1→`--jobs=1`)
- [x] 3.2 Unit-тест `NUITKA_JOBS=3` переопределяет дефолт
- [x] 3.3 Прогон `pytest` (без gui/slow) и `ruff` на change

## 4. Верификация и публикация

- [x] 4.1 `openspec validate --type change nu-packaging-build` → valid (после правок)
- [x] 4.2 `openspec archive nu-packaging-build --yes` → `nu-packaging` в main specs
- [x] 4.3 `openspec validate --all` → всё валидно
- [x] 4.4 Коммит и push (ветка `feature/testing`)

## 5. Итог

- [x] 5.1 Код синхронизирован со спекой (CI не дублирует флаги, `--jobs` вычисляется)
- [x] 5.2 `scripts/nuitka_flags.py`/`nuitka_build.py` — единый источник флагов