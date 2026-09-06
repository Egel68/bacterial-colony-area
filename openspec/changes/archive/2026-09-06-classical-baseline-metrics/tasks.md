## 1. Dataset loader for importer-output and legacy structures

- [x] 1.1 Создать `testing/baseline.py` с классом `BaselineDataset`, который авто-детектит структуру: приоритет `dataset.json` (ManifestAdapter-совместимый), затем `source/` + `masks/` (легаси test_images), затем `source/*_mask.png` (CocoBboxImporter-output). Загружать пары лениво (iterable). Проверить: `BaselineDataset(root)` без ошибок на всех трёх типах структур.
- [x] 1.2 Реализовать метод `BaselineDataset.samples()`, возвращающий список кортежей `(image_path, mask_path, variant)` для source и (опционально) cropped. Проверить: для CocoBboxImporter-output количество пар совпадает с `len(listdir(source/)) - len(listdir(source/))` (исключая маски).
- [x] 1.3 Реализовать отчёт о размере датасета (число source/cropped пар, пустой датасет → ошибка). Проверить: пустая папка вызывает `RuntimeError("No valid image-mask pairs found")`.

## 2. Baseline runner

- [x] 2.1 Реализовать функцию `run_baseline(dataset: BaselineDataset, use_cropped: bool = True) -> dict`, которая прогоняет 4 классических алгоритма (через `get_algorithm` из registry) и вычисляет метрики через `compute_segmentation_metrics`. Проверить: на датасете из 1 пары результат содержит 4 алгоритма × 2 варианта (source/cropped).
- [x] 2.2 Использовать `_mean_metrics` из `testing/runner.py` для агрегации source и cropped отдельно, группируя по variant. Проверить: mean/std поля присутствуют в выходном словаре.
- [x] 2.3 Добавить прогресс-бар через `tqdm` (опциональная зависимость, fallback на простой print). Проверить: при запуске на 10+ парах прогресс отображается.

## 3. JSON report export

- [x] 3.1 Реализовать функцию `export_json(result: dict, output_path: str)`, которая формирует JSON со структурой: `{dataset_name, image_count, timestamp, algorithms: [{name, description, source: {mean_iou, std_iou, ...}, cropped: {...}}]}`. Проверить: `json.load(open(output_path))` читается без ошибок, поля соответствуют контракту.
- [x] 3.2 Добавить `dataset_name` (имя корневой папки датасета) и `timestamp` (ISO 8601). Проверить: поле timestamp парсится `datetime.fromisoformat()`.

## 4. CLI entry point

- [x] 4.1 Добавить в `testing/__main__.py` режим `--mode baseline` с флагами `--data-root`, `--output`, `--no-cropped`. Проверить: `uv run python -m testing --mode baseline --data-root test_images --output /tmp/baseline.json` завершается без ошибок, JSON создан.
- [x] 4.2 Добавить entry point `baseline` в `pyproject.toml` (секция `[project.scripts]`): `baseline = "testing.__main__:main"`. Проверить: `uv run baseline --data-root test_images --output /tmp/baseline.json` даёт тот же результат.
- [x] 4.3 Проверить интеграционно: запустить baseline на существующем `test_images/` датасете, убедиться что JSON содержит 4 алгоритма с ненулевыми метриками. Проверить: в JSON есть все 4 алгоритма, mean_iou > 0 для каждого.