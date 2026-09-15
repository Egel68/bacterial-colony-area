## 1. YAML config parser

- [x] 1.1 Создать `testing/evaluator.py` с функцией `load_config(config_path: str | None, cli_args: argparse.Namespace) -> dict`, которая загружает YAML-конфиг (или JSON fallback) и объединяет с CLI-аргументами. Проверить: `load_config("nonexistent.yaml", ...)` не падает, `load_config(None, args)` возвращает словарь из CLI-аргументов.

## 2. Run directory and metadata

- [x] 2.1 Реализовать функцию `create_run_dir(base_dir: str = "evaluations") -> str`, которая создаёт `evaluations/YYYY-MM-DD_HH-MM-SS/` и возвращает run-id (строку временой метки). Проверить: повторный вызов создаёт разные директории.
- [x] 2.2 Реализовать функцию `get_git_info() -> dict`, которая выполняет `git rev-parse HEAD`, `git log -1 --format=%s` и возвращает `commit_hash`, `commit_message`. Проверить: вызов внутри репозитория возвращает непустой commit_hash из 40 hex-символов.

## 3. Pipeline orchestration

- [x] 3.1 Реализовать функцию `run_evaluate(config: dict) -> str`, которая: (a) опционально запускает `CocoBboxImporter().build()` если указан `import_root`; (b) загружает датасет через `BaselineDataset` или `TestDataset`; (c) выполняет `run_baseline()`; (d) дорабатывает JSON-результат — добавляет `run_id`, `git_commit`; (e) генерирует HTML через `generate_report()`; (f) сохраняет всё в `evaluations/<run-id>/`. Проверить: `uv run evaluate --data-root test_images` создаёт `evaluations/<run-id>/` с 4 файлами.

## 4. CLI entry point

- [x] 4.1 Добавить в `testing/__main__.py` режим `--mode evaluate` с параметрами `--config`, `--data-root`, `--output`, `--no-cropped`, `--import-root`. Проверить: `uv run python -m testing --mode evaluate --data-root test_images` завершается без ошибок.
- [x] 4.2 Добавить entry point `evaluate` в `pyproject.toml`: `evaluate = "testing.__main__:main_evaluate"`. Проверить: `uv run evaluate --data-root test_images` работает.

## 5. Integration test on real dataset

- [x] 5.1 Запустить `uv run evaluate --data-root test_images` и проверить, что в `evaluations/` создалась директория с корректным JSON и HTML. Проверить: `report.json` читается `json.load()`, `report.html` содержит Chart.js скрипт, `run_info.json` содержит git-commit.
- [x] 5.2 Проверить, что повторный прогон создаёт новую директорию (не перезаписывает). Проверить: `ls -1 evaluations/ | wc -l` увеличился на 1.