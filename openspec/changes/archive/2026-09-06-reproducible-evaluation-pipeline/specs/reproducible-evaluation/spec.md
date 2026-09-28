## Purpose

Единый воспроизводимый pipeline оценки алгоритмов распознавания колоний: конфигурация (YAML) → импорт данных → прогон алгоритмов → расчёт метрик → сохранение результатов в git-tracked директории с run-id → HTML-визуализация.

## ADDED Requirements

### Requirement: One-command evaluation pipeline

Система SHALL предоставлять CLI-команду `uv run evaluate`, которая за один вызов выполняет полный pipeline: загрузка/импорт датасета, прогон выбранных алгоритмов, расчёт метрик, генерация HTML-отчёта и сохранение результатов в структурированную директорию. Команда SHALL принимать конфигурационный файл YAML (`--config`) и/или прямые аргументы.

#### Scenario: Evaluate with default config
- **WHEN** команда запущена как `uv run evaluate --data-root test_images`
- **THEN** pipeline SHALL выполнить прогон, создать директорию `evaluations/YYYY-MM-DD_HH-MM-SS/` и записать туда отчёт

#### Scenario: Evaluate with config file
- **WHEN** команда запущена как `uv run evaluate --config eval_config.yaml`
- **THEN** pipeline SHALL использовать параметры из YAML-файла (data-root, algorithms, models, опции)

### Requirement: Run identifier and result directory

Каждый прогон SHALL получать уникальный идентификатор на основе временной метки: `YYYY-MM-DD_HH-MM-SS`. Директория `evaluations/<run-id>/` SHALL содержать: `report.json` (метрики), `report.html` (визуализация), `config.yaml` (копия конфигурации), `run_info.json` (git-commit, алгоритмы, параметры). Система SHALL NOT перезаписывать существующие прогоны.

#### Scenario: Run directory created
- **WHEN** pipeline выполнен
- **THEN** директория `evaluations/<run-id>/` существует и содержит `report.json`, `report.html`, `config.yaml`, `run_info.json`

#### Scenario: No overwrite
- **WHEN** второй прогон следует сразу за первым
- **THEN** директории SHALL иметь разные временные метки

### Requirement: Run info metadata

Каждый результат прогона SHALL содержать метаданные: `commit_hash` (результат `git rev-parse HEAD`), `commit_message`, `algorithms_run` (список имён алгоритмов), `dataset_path`, `dataset_size`, `timestamp`, `config` (копия параметров запуска).

#### Scenario: Run info populated
- **WHEN** `run_info.json` открыт
- **THEN** он SHALL содержать commit_hash (строка 40 символов), algorithms_run (массив), dataset_path, timestamp

### Requirement: Configuration file format

Система SHALL поддерживать конфигурационные YAML-файлы со структурой: `data_root` (путь), `output` (путь, опционально), `algorithms` (список или null для всех), `models` (список путей к .onnx), `compare` (пара A,B для сравнения), `use_cropped` (bool), `per_snapshot` (bool). Отсутствующие поля SHALL использовать значения по умолчанию.

#### Scenario: Config with subset of algorithms
- **WHEN** YAML-конфиг содержит `algorithms: [ClassicDefault, ClassicHighSensitivity]` и не содержит `models`
- **THEN** pipeline SHALL запустить только указанные 2 алгоритма, без ONNX-моделей

#### Scenario: Empty config uses defaults
- **WHEN** YAML-конфиг пуст (только `data_root: test_images`)
- **THEN** pipeline SHALL использовать значения по умолчанию: все классические алгоритмы, use_cropped=true

### Requirement: HTML visualization

Система SHALL генерировать HTML-отчёт с Chart.js графиками (сводная таблица, bar charts для IoU/Dice/F1/Precision/Recall, парное сравнение) при каждом прогоне в `evaluations/<run-id>/report.html`, используя существующий `testing/dashboard.py:generate_report`.

#### Scenario: HTML report in run directory
- **WHEN** pipeline завершён
- **THEN** `evaluations/<run-id>/report.html` SHALL существовать и содержать сводную таблицу и графики

#### Scenario: Report mentions run ID
- **WHEN** HTML-отчёт открыт
- **THEN** заголовок SHALL содержать run-id и timestamp прогона

### Requirement: Import dataset as pipeline step

Pipeline SHALL опционально выполнять импорт датасета через CocoBboxImporter перед прогоном, если указаны `--import-root` или в конфиге `import_root`. Если `import_root` указан вместе с `data_root`, импорт SHALL выполниться, а `data_root` SHALL указывать на output импорта. Если импорт не указан, pipeline SHALL работать с уже готовым датасетом.

#### Scenario: Import before evaluate
- **WHEN** конфиг содержит `import_root: datasets/22022540` и `data_root: datasets/22022540_imported`
- **THEN** pipeline SHALL сначала выполнить import-22022540, затем baseline-прогон

#### Scenario: Skip import if exists
- **WHEN** `data_root` уже существует и содержит валидный датасет
- **THEN** pipeline SHALL пропустить импорт и сразу перейти к прогону