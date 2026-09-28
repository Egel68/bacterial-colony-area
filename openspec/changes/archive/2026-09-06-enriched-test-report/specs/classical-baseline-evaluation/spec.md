## MODIFIED Requirements

### Requirement: JSON baseline report export

Система SHALL экспортировать результаты baseline-прогона в JSON-файл со следующей структурой: корневой объект с полями `dataset_name`, `image_count`, `timestamp`, и `algorithms` — массив объектов, каждый из которых содержит `name`, `description`, и вложенные `source` и (опционально) `cropped` с полями `mean_iou`, `std_iou`, `median_iou`, `q1_iou`, `q3_iou`, `p5_iou`, `p95_iou`, `min_iou`, `max_iou`, `mean_dice`, `std_dice`, `median_dice`, `q1_dice`, `q3_dice`, `p5_dice`, `p95_dice`, `min_dice`, `max_dice`, `mean_f1`, `std_f1`, `median_f1`, `q1_f1`, `q3_f1`, `p5_f1`, `p95_f1`, `min_f1`, `max_f1`, `mean_precision`, `std_precision`, `median_precision`, `q1_precision`, `q3_precision`, `p5_precision`, `p95_precision`, `min_precision`, `max_precision`, `mean_recall`, `std_recall`, `median_recall`, `q1_recall`, `q3_recall`, `p5_recall`, `p95_recall`, `min_recall`, `max_recall`, `mean_accuracy`, `std_accuracy`, `median_accuracy`, `q1_accuracy`, `q3_accuracy`, `p5_accuracy`, `p95_accuracy`, `min_accuracy`, `max_accuracy`, `num_samples`. Файл SHALL быть валидным JSON и читаться стандартным `json.load()`. Система SHALL также сохранять кэш метрик по алгоритмам в отдельный файл `.cache/<dataset_signature>.json` и читать его при повторных запусках.

#### Scenario: JSON report contains all algorithms

- **WHEN** baseline-прогон завершён
- **THEN** JSON-файл SHALL содержать по одному объекту для каждого из 4 классических алгоритмов с агрегированными метриками, включая median, q1, q3, p5, p95, min, max

#### Scenario: JSON report is machine-readable

- **WHEN** JSON-файл открыт python-скриптом через `json.load()`
- **THEN** структура SHALL соответствовать описанному контракту без дополнительных вложенностей