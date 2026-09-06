## Why

Текущий фреймворк тестирования (`testing/runner.py`) позволяет прогонять классические алгоритмы только на структуре `test_images/` (source/ + masks/). Для загрузки импортированного датасета 22022540 (369 изображений, 56 865 колоний) не хватает конвейера: CocoBboxImporter пишет маски в той же папке, что и исходники (source/*_mask.png), а TestDataset ожидает отдельную папку masks/. При этом метрики классических алгоритмов на всём датасете нужны как baseline для сравнения с будущей нейросетевой моделью — без них нельзя объективно оценить, улучшило ли обучение качество распознавания.

## What Changes

- Добавить скрипт/команду для массового прогона 4 классических алгоритмов (ClassicDefault, ClassicHighSensitivity, ClassicSolidFill, ClassicLowSensitivity) на датасете 22022540 (как на source-, так и на cropped-вариантах).
- Адаптировать загрузчик датасета (TestDataset) или создать прослойку, чтобы он принимал структуру, порождаемую CocoBboxImporter (source/*.jpg + source/*_mask.png) — без её изменения.
- Сохранять результаты метрик (IoU, Dice, F1, Precision, Recall, Accuracy) в структурированном формате (JSON / CSV) для последующего сравнения с нейросетевыми моделями.
- Все параметры классических алгоритмов — те же, что используются в UI через AnalysisParams (sensitivity, margin_percent, min_colony_size, contrast, solid_fill).

## Capabilities

### New Capabilities
- `classical-baseline-evaluation`: Пакетный прогон классических алгоритмов на импортированном датасете (CocoBboxImporter-output), вычисление метрик сегментации, экспорт baseline-отчёта в JSON.

### Modified Capabilities
*(Нет изменений требований существующих specs — поведение тестового фреймворка и алгоритмов не меняется.)*

## Impact

- `testing/` — новый модуль (например, `testing/baseline.py`) или расширение `testing/__main__.py` с флагом `--baseline`.
- `testing/dataset.py` — опциональная адаптация TestDataset для чтения импортированной структуры.
- `testing/metrics.py` — без изменений (метрики уже есть).
- `testing/runner.py` — без изменений (runner уже умеет считать).
- Выходной файл (JSON-отчёт) — создаётся в указанной директории.