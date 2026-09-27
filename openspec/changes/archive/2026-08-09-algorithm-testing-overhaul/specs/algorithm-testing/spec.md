## MODIFIED Requirements

### Requirement: Runner and aggregation

Система SHALL прогонять выбранные алгоритмы по датасету (`run_algorithm` для одного алгоритма, `run_all` для подмножества), собирать метрики для variant `source` и (при наличии) `cropped`, и усреднять (`_mean_metrics`) в `mean_*`/`std_*`, исключая сырые TP/FP/FN/TN. Для пустого списка `_mean_metrics` SHALL возвращать пустой словарь.

#### Scenario: Empty metrics list
- **КОГДА** `_mean_metrics([])` вызывается
- **ТОГДА** результат SHALL быть пустым словарём

#### Scenario: Run algorithm on dataset
- **КОГДА** `run_algorithm(algo, dataset)` вызывается на непустом датасете
- **ТОГДА** результат SHALL содержит по ключу sample-имени словарь с метриками для `source`

#### Scenario: Run subset of algorithms
- **КОГДА** `run_all(dataset, subset)` вызывается с `subset=["ClassicDefault", "ClassicSolidFill"]`
- **ТОГДА** результат SHALL содержать метрики только для выбранных алгоритмов

### Requirement: HTML report

Система SHALL генерировать HTML-отчёт (`generate_report`) со сводной таблицей, графиками Chart.js и опциональным разделом парного сравнения алгоритмов, записывая файл по указанному пути.

#### Scenario: Report generated
- **КОГДА** `generate_report(all_results, output_path)` вызывается с результатами
- **ТОГДА** файл SHALL быть создан и содержать ключевые метрики (`mean_iou`)

#### Scenario: Report with paired comparison
- **КОГДА** `generate_report(all_results, output_path)` вызывается с включённым парным сравнением
- **ТОГДА** файл SHALL дополнительно содержать таблицу «победитель по каждому снимку» для каждой метрики

## ADDED Requirements

### Requirement: Dataset root selection

Система SHALL принимать произвольный корень данных для датасета (CLI `--data-root`, GUI — выбор папки), по умолчанию `test_images`. При отсутствии валидных пар источник+маска запуск SHALL сообщать о пустом датасете и останавливаться.

#### Scenario: Custom data root
- **КОГДА** `TestDataset(root="/path")` создаётся с произвольным путём
- **ТОГДА** загрузка SHALL идти из папок `source/`, `masks/` и т.д. внутри этого пути

#### Scenario: Empty dataset is reported
- **КОГДА** в выбранном корне нет ни одной валидной пары изображение+маска
- **ТОГДА** запуск (CLI и GUI) SHALL показать сообщение «No test samples found» и прерваться

### Requirement: Per-snapshot metrics toggle

Система SHALL optionally включать/выключать детализацию метрик по каждому снимку в отчёте. Сводные `mean_*`/`std_*` считаются всегда; детальные по-сним метрики включаются только при включённой опции.

#### Scenario: Per-snapshot enabled
- **КОГДА** запуск выполнен с включённой детализацией по снимкам
- **ТОГДА** отчёт SHALL содержать таблицу «снимок × variant × метрика» для каждого алгоритма

#### Scenario: Per-snapshot disabled
- **КОГДА** запуск выполнен без детализации по снимкам
- **ТОГДА** отчёт SHALL содержать только агрегированные метрики по выборке