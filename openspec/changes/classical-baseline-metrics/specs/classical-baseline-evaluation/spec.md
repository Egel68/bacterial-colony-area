## Purpose

Пакетный прогон классических (OpenCV) алгоритмов распознавания колоний на импортированном датасете (CocoBboxImporter-output) с вычислением метрик сегментации и экспортом baseline-отчёта в JSON для сравнения с будущими нейросетевыми моделями.

## ADDED Requirements

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

Система SHALL прогонять каждый из 4 зарегистрированных классических алгоритмов (ClassicDefault, ClassicHighSensitivity, ClassicSolidFill, ClassicLowSensitivity) на всём датасете, используя те же параметры AnalysisParams, что определены в `testing/classic_algorithms.py` и используются в UI-окне анализа. Количество прогонов SHALL равняться `(количество алгоритмов) × (количество source-образцов + количество cropped-образцов)`.

#### Scenario: Four algorithms run on dataset
- **WHEN** baseline-прогон запущен на датасете с 50 source-парами
- **THEN** каждый из 4 алгоритмов SHALL обработать 50 source-снимков, метрики SHALL быть вычислены для каждого снимка

#### Scenario: Algorithms use production parameters
- **WHEN** ClassicDefault.detect() вызывается в baseline-прогоне
- **THEN** параметры детекции SHALL быть sensitivity=0.5, margin=10%, min_size=50, contrast=1.0, solid_fill=False

### Requirement: Metrics computation

Для каждого прогона алгоритма система SHALL вычислять те же метрики сегментации, что определены в `testing/metrics.py`: IoU, Dice, Precision, Recall, F1, Accuracy с smooth=1e-6, на основе попиксельного сравнения TP/FP/FN/TN (порог > 0). Для каждого алгоритма SHALL вычисляться средние (mean) и стандартные отклонения (std) по всем снимкам отдельно для source и cropped вариантов.

#### Scenario: Mean metrics computed
- **WHEN** алгоритм обработал N source-снимков
- **THEN** результат SHALL содержать mean_iou, mean_dice, mean_f1, mean_precision, mean_recall, mean_accuracy и соответствующие std_* по всем N снимкам

#### Scenario: Source and cropped aggregated separately
- **WHEN** датасет содержит и source, и cropped варианты
- **THEN** агрегированные метрики SHALL вычисляться отдельно для source-выборки и для cropped-выборки

### Requirement: JSON baseline report export

Система SHALL экспортировать результаты baseline-прогона в JSON-файл со следующей структурой: корневой объект с полями `dataset_name`, `image_count`, `timestamp`, и `algorithms` — массив объектов, каждый из которых содержит `name`, `description`, и вложенные `source` и (опционально) `cropped` с полями `mean_iou`, `std_iou`, `mean_dice`, `std_dice`, `mean_f1`, `std_f1`, `mean_precision`, `std_precision`, `mean_recall`, `std_recall`, `mean_accuracy`, `std_accuracy`, `num_samples`. Файл SHALL быть валидным JSON и читаться стандартным `json.load()`.

#### Scenario: JSON report contains all algorithms
- **WHEN** baseline-прогон завершён
- **THEN** JSON-файл SHALL содержать по одному объекту для каждого из 4 классических алгоритмов с агрегированными метриками

#### Scenario: JSON report is machine-readable
- **WHEN** JSON-файл открыт python-скриптом через `json.load()`
- **THEN** структура SHALL соответствовать описанному контракту без дополнительных вложенностей

### Requirement: CLI entry point for baseline

Система SHALL предоставлять CLI-команду (или расширение существующего CLI `test-algorithms`) для запуска baseline-прогона с параметрами: `--data-root` (путь к датасету), `--output` (путь к JSON-отчёту), `--no-cropped` (опционально, отключить cropped-варианты).

#### Scenario: Baseline CLI accepts data-root
- **WHEN** команда запущена с `--data-root /path/to/dataset --output baseline.json`
- **THEN** система SHALL загрузить датасет по указанному пути и записать отчёт в baseline.json

#### Scenario: Baseline CLI with cropped disabled
- **WHEN** команда запущена с `--no-cropped`
- **THEN** система SHALL НЕ загружать и НЕ обрабатывать cropped-варианты