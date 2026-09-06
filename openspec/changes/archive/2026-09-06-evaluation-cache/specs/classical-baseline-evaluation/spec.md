## MODIFIED Requirements

### Requirement: Batch run with algorithm parameters matching UI

Система SHALL прогонять каждый из 4 зарегистрированных классических алгоритмов (ClassicDefault, ClassicHighSensitivity, ClassicSolidFill, ClassicLowSensitivity) на всём датасете, используя те же параметры AnalysisParams, что определены в `testing/classic_algorithms.py` и используются в UI-окне анализа. Количество прогонов SHALL равняться `(количество алгоритмов) × (количество source-образцов + количество cropped-образцов)`. Система SHALL проверять наличие кэшированных метрик перед каждым прогоном алгоритма и пропускать алгоритм, если его метрики для данного датасета уже существуют в кэше и сигнатура датасета не изменилась.

#### Scenario: Four algorithms run on dataset
- **WHEN** baseline-прогон запущен на датасете с 50 source-парами
- **THEN** каждый из 4 алгоритмов SHALL обработать 50 source-снимков, метрики SHALL быть вычислены для каждого снимка

#### Scenario: Algorithms use production parameters
- **WHEN** ClassicDefault.detect() вызывается в baseline-прогоне
- **THEN** параметры детекции SHALL быть sensitivity=0.5, margin=10%, min_size=50, contrast=1.0, solid_fill=False

#### Scenario: Cached algorithm is skipped
- **WHEN** baseline-прогон запущен повторно на том же датасете и для алгоритма `ClassicDefault` уже есть кэшированные метрики
- **THEN** алгоритм SHALL быть пропущен, а его метрики SHALL быть взяты из кэша без повторного детектирования

#### Scenario: Dataset change invalidates cache
- **WHEN** датасет изменён (добавлены/удалены/изменены файлы изображений или масок)
- **THEN** при следующем baseline-прогоне система SHALL пересчитать все алгоритмы с нуля, так как сигнатура датасета изменилась

#### Scenario: Partial cache is reused
- **WHEN** из 4 алгоритмов 2 уже имеют кэшированные метрики
- **THEN** система SHALL пропустить 2 алгоритма и пересчитать только 2 оставшихся

### Requirement: JSON baseline report export

Система SHALL экспортировать результаты baseline-прогона в JSON-файл со следующей структурой: корневой объект с полями `dataset_name`, `image_count`, `timestamp`, и `algorithms` — массив объектов, каждый из которых содержит `name`, `description`, и вложенные `source` и (опционально) `cropped` с полями `mean_iou`, `std_iou`, `mean_dice`, `std_dice`, `mean_f1`, `std_f1`, `mean_precision`, `std_precision`, `mean_recall`, `std_recall`, `mean_accuracy`, `std_accuracy`, `num_samples`. Файл SHALL быть валидным JSON и читаться стандартным `json.load()`. Система SHALL также сохранять кэш метрик по алгоритмам в отдельный файл `.cache/<dataset_signature>.json` и читать его при повторных запусках.

#### Scenario: JSON report contains all algorithms
- **WHEN** baseline-прогон завершён
- **THEN** JSON-файл SHALL содержать по одному объекту для каждого из 4 классических алгоритмов с агрегированными метриками

#### Scenario: JSON report is machine-readable
- **WHEN** JSON-файл открыт python-скриптом через `json.load()`
- **THEN** структура SHALL соответствовать описанному контракту без дополнительных вложенностей

#### Scenario: Cache file contains algorithm entries
- **WHEN** алгоритм `ClassicDefault` успешно отработал на датасете
- **THEN** его метрики SHALL быть сохранены в `.cache/<dataset_signature>.json` под ключом `ClassicDefault`

#### Scenario: Cache file is read on subsequent run
- **WHEN** baseline-прогон запущен на датасете с существующим файлом `.cache/<dataset_signature>.json`
- **THEN** система SHALL загрузить кэш и для каждого алгоритма, присутствующего в кэше, SHALL NOT запускать детектирование

### Requirement: Cache management CLI

Система SHALL предоставлять CLI-флаг `--clear-cache` для принудительной очистки кэша перед запуском. Флаг SHALL удалять файл кэша для данного датасета, заставляя систему пересчитать все алгоритмы с нуля.

#### Scenario: Cache cleared via CLI
- **WHEN** команда запущена с `--clear-cache --data-root /path/to/dataset`
- **THEN** система SHALL удалить `.cache/<dataset_signature>.json` (если существует) и пересчитать все алгоритмы

#### Scenario: Cache cleared for specific dataset only
- **WHEN** команда запущена с `--clear-cache`
- **THEN** система SHALL удалить кэш только для датасета, указанного в `--data-root`, не затрагивая кэши других датасетов