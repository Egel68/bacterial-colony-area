# classical-baseline-evaluation Specification

## Purpose

Пакетный прогон классических (OpenCV) алгоритмов распознавания колоний на импортированном датасете (CocoBboxImporter-output) с вычислением метрик сегментации и экспортом baseline-отчёта в JSON для сравнения с будущими нейросетевыми моделями.

## Requirements

### Requirement: Importer-output dataset loading

Система SHALL загружать датасет в структуре, порождаемой CocoBboxImporter: `source/*.{jpg,png}` и соответствующие маски `source/*_mask.png` в одной директории, а также опционально `cropped/*_cropped.{jpg,png}` + `cropped/*_cropped_mask.png`. Система SHALL поддерживать загрузку как через имеющийся TestDataset (при совместимой структуре папок), так и через ManifestAdapter + dataset.json для датасетов с манифестом.

#### Scenario: Load importer source-only dataset
- **WHEN** датасет содержит `source/X.jpg` + `source/X_mask.png` (без dataset.json)
- **THEN** система SHALL загрузить пару с source-изображением и source-маской

#### Scenario: Load importer dataset with cropped variants
- **WHEN** датасет содержит также `cropped/X_cropped.jpg` + `cropped/X_cropped_mask.png`
- **THEN** система SHALL дополнительно загрузить cropped-пару для того же снимка

### Requirement: Dataset size reporting

Перед прогоном система SHALL выводить количество найденных пар изображение-маска отдельно для source и cropped вариантов. Если датасет пуст, система SHALL завершаться с сообщением об ошибке.

#### Scenario: Non-empty dataset
- **WHEN** загружен датасет с 100 source-парами
- **THEN** система SHALL сообщить "Found 100 source samples" и начать прогон

#### Scenario: Empty dataset
- **WHEN** в указанном корне нет ни одной валидной пары
- **THEN** система SHALL вывести сообщение "No valid image-mask pairs found" и завершиться

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

### Requirement: Metrics computation

Для каждого прогона алгоритма система SHALL вычислять те же метрики сегментации, что определены в `testing/metrics.py`: IoU, Dice, Precision, Recall, F1, Accuracy с smooth=1e-6, на основе попиксельного сравнения TP/FP/FN/TN (порог > 0). Для каждого алгоритма SHALL вычисляться средние (mean) и стандартные отклонения (std) по всем снимкам отдельно для source и cropped вариантов.

#### Scenario: Mean metrics computed
- **WHEN** алгоритм обработал N source-снимков
- **THEN** результат SHALL содержать mean_iou, mean_dice, mean_f1, mean_precision, mean_recall, mean_accuracy и соответствующие std_* по всем N снимкам

#### Scenario: Source and cropped aggregated separately
- **WHEN** датасет содержит и source, и cropped варианты
- **THEN** агрегированные метрики SHALL вычисляться отдельно для source-выборки и для cropped-выборки

### Requirement: JSON baseline report export

Система SHALL экспортировать результаты baseline-прогона в JSON-файл со следующей структурой: корневой объект с полями `dataset_name`, `image_count`, `timestamp`, и `algorithms` — массив объектов, каждый из которых содержит `name`, `description`, и вложенные `source` и (опционально) `cropped` с полями `mean_iou`, `std_iou`, `median_iou`, `q1_iou`, `q3_iou`, `p5_iou`, `p95_iou`, `min_iou`, `max_iou`, `mean_dice`, `std_dice`, `median_dice`, `q1_dice`, `q3_dice`, `p5_dice`, `p95_dice`, `min_dice`, `max_dice`, `mean_f1`, `std_f1`, `median_f1`, `q1_f1`, `q3_f1`, `p5_f1`, `p95_f1`, `min_f1`, `max_f1`, `mean_precision`, `std_precision`, `median_precision`, `q1_precision`, `q3_precision`, `p5_precision`, `p95_precision`, `min_precision`, `max_precision`, `mean_recall`, `std_recall`, `median_recall`, `q1_recall`, `q3_recall`, `p5_recall`, `p95_recall`, `min_recall`, `max_recall`, `mean_accuracy`, `std_accuracy`, `median_accuracy`, `q1_accuracy`, `q3_accuracy`, `p5_accuracy`, `p95_accuracy`, `min_accuracy`, `max_accuracy`, `num_samples`. Файл SHALL быть валидным JSON и читаться стандартным `json.load()`. Система SHALL также сохранять кэш метрик по алгоритмам в отдельный файл `.cache/<dataset_signature>.json` и читать его при повторных запусках.

#### Scenario: JSON report contains all algorithms
- **WHEN** baseline-прогон завершён
- **THEN** JSON-файл SHALL содержать по одному объекту для каждого из 4 классических алгоритмов с агрегированными метриками, включая median, q1, q3, p5, p95, min, max

#### Scenario: JSON report is machine-readable
- **WHEN** JSON-файл открыт python-скриптом через `json.load()`
- **THEN** структура SHALL соответствовать описанному контракту без дополнительных вложенностей

#### Scenario: Cache file contains algorithm entries
- **WHEN** алгоритм `ClassicDefault` успешно отработал на датасете
- **THEN** его метрики SHALL быть сохранены в `.cache/<dataset_signature>.json` под ключом `ClassicDefault`

#### Scenario: Cache file is read on subsequent run
- **WHEN** baseline-прогон запущен на датасете с существующим файлом `.cache/<dataset_signature>.json`
- **THEN** система SHALL загрузить кэш и для каждого алгоритма, присутствующего в кэше, SHALL NOT запускать детектирование

### Requirement: CLI entry point for baseline

Система SHALL предоставлять CLI-команду (или расширение существующего CLI `test-algorithms`) для запуска baseline-прогона с параметрами: `--data-root` (путь к датасету), `--output` (путь к JSON-отчёту), `--no-cropped` (опционально, отключить cropped-варианты).

#### Scenario: Baseline CLI accepts data-root
- **WHEN** команда запущена с `--data-root /path/to/dataset --output baseline.json`
- **THEN** система SHALL загрузить датасет по указанному пути и записать отчёт в baseline.json

#### Scenario: Baseline CLI with cropped disabled
- **WHEN** команда запущена с `--no-cropped`
- **THEN** система SHALL НЕ загружать и НЕ обрабатывать cropped-варианты

### Requirement: Cache management CLI

Система SHALL предоставлять CLI-флаг `--clear-cache` для принудительной очистки кэша перед запуском. Флаг SHALL удалять файл кэша для данного датасета, заставляя систему пересчитать все алгоритмы с нуля.

#### Scenario: Cache cleared via CLI
- **WHEN** команда запущена с `--clear-cache --data-root /path/to/dataset`
- **THEN** система SHALL удалить `.cache/<dataset_signature>.json` (если существует) и пересчитать все алгоритмы

#### Scenario: Cache cleared for specific dataset only
- **WHEN** команда запущена с `--clear-cache`
- **THEN** система SHALL удалить кэш только для датасета, указанного в `--data-root`, не затрагивая кэши других датасетов