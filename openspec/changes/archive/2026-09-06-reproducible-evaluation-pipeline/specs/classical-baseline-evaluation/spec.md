## MODIFIED Requirements

### Requirement: JSON baseline report export

Система SHALL экспортировать результаты baseline-прогона в JSON-файл со следующей структурой: корневой объект с полями `dataset_name`, `image_count`, `timestamp`, и `algorithms` — массив объектов, каждый из которых содержит `name`, `description`, и вложенные `source` и (опционально) `cropped` с полями `mean_iou`, `std_iou`, `mean_dice`, `std_dice`, `mean_f1`, `std_f1`, `mean_precision`, `std_precision`, `mean_recall`, `std_recall`, `mean_accuracy`, `std_accuracy`, `num_samples`. Файл SHALL быть валидным JSON и читаться стандартным `json.load()`. При запуске через `uv run evaluate` JSON-файл SHALL дополнительно содержать поля `run_id` (строка временной метки) и `git_commit` (SHA коммита).

#### Scenario: JSON report contains all algorithms
- **WHEN** baseline-прогон завершён
- **THEN** JSON-файл SHALL содержать по одному объекту для каждого из 4 классических алгоритмов с агрегированными метриками

#### Scenario: JSON report is machine-readable
- **WHEN** JSON-файл открыт python-скриптом через `json.load()`
- **THEN** структура SHALL соответствовать описанному контракту без дополнительных вложенностей

#### Scenario: JSON from pipeline contains run_id
- **WHEN** прогон выполнен через `uv run evaluate`
- **THEN** JSON-файл SHALL содержать `run_id` и `git_commit`

### Requirement: CLI entry point for baseline

Система SHALL предоставлять CLI-команду (или расширение существующего CLI `test-algorithms`) для запуска baseline-прогона с параметрами: `--data-root` (путь к датасету), `--output` (путь к JSON-отчёту), `--no-cropped` (опционально, отключить cropped-варианты). Команда `uv run evaluate` SHALL также принимать те же параметры и дополнительно `--config` (YAML-файл конфигурации). При запуске через `uv run evaluate` без `--output` путь сохранения SHALL автоматически определяться как `evaluations/<run-id>/report.json`.

#### Scenario: Baseline CLI accepts data-root
- **WHEN** команда запущена с `--data-root /path/to/dataset --output baseline.json`
- **THEN** система SHALL загрузить датасет по указанному пути и записать отчёт в baseline.json

#### Scenario: Baseline CLI with cropped disabled
- **WHEN** команда запущена с `--no-cropped`
- **THEN** система SHALL НЕ загружать и НЕ обрабатывать cropped-варианты

#### Scenario: Evaluate auto-names output
- **WHEN** `uv run evaluate --data-root test_images` без `--output`
- **THEN** система SHALL сохранить результаты в `evaluations/<run-id>/report.json`