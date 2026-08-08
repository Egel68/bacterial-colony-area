## ADDED Requirements

### Requirement: Algorithm registry

Система SHALL предоставлять реестр алгоритмов детекции. Алгоритм SHALL регистрироваться через декоратор `@register_algorithm` (по имени класса), запрашиваться через `get_algorithm(name)` и перечисляться через `list_algorithms()` (сортированный список). Для неизвестного имени система SHALL выбрасывать `ValueError` с перечнем доступных алгоритмов.

#### Scenario: Registered algorithm is listed
- **КОГДА** класс помечен `@register_algorithm`
- **ТОГДА** `list_algorithms()` SHALL содержать его имя, `get_algorithm(name)` SHALL вернуть экземпляр

#### Scenario: Unknown algorithm raises
- **КОГДА** `get_algorithm("nope")` с неизвестным именем
- **ТОГДА** система SHALL выбросить `ValueError` с сообщением "Unknown algorithm"

### Requirement: Algorithm interface

Каждый алгоритм SHALL реализовывать `BaseDetectionAlgorithm` с атрибутами `name`, `description` и методом `detect(image, is_cropped=False)`, возвращающим бинарную `uint8`-маску того же размера, что и входное изображение.

#### Scenario: Detect returns mask
- **КОГДА** `detect(image)` вызывается для BGR-изображения
- **ТОГДА** возвращается маска `ndim=2`, `dtype=uint8`, shape совпадает с изображением

#### Scenario: Cropped input detected
- **КОГДА** `detect(image, is_cropped=True)` для изображения обрезки чашки
- **ТОГДА** чашка предполагается в центре (`r = min(w,h)/2`), возвращается маска того же shape

### Requirement: Classic algorithm variants

Система SHALL предоставлять 4 зарегистрированных классических алгоритма с фиксированными параметрами: `ClassicDefault` (sensitivity=0.5, margin=10%, min_size=50), `ClassicHighSensitivity` (0.7, 5%, 30), `ClassicSolidFill` (0.3, 15%, 50, solid_fill=true, fill_strength=15), `ClassicLowSensitivity` (0.3, 10%, 100). Каждый SHALL использовать `ColonyDetector` и `AnalysisParams`.

#### Scenario: Default variant runs
- **КОГДА** `ClassicDefault().detect(image)` вызывается для изображения с чашкой
- **ТОГДА** возвращается бинарная маска того же shape

#### Scenario: Fallback when dish not detected
- **КОГДА** для не-cropped изображения `detect_petri_dish` возвращает `None`
- **ТОГДА** алгоритм SHALL использовать окружность по центру изображения (`r = 0.4·min(w,h)`)

### Requirement: Test dataset loading

Система SHALL загружать парные данные из `test_images/`: для каждой пары из `source/` + `masks/` строится sample с `source_image`, `source_mask`; при наличии пары в `cropped/` + `cropped_masks/` — также `cropped_image` и `cropped_mask`. Отсутствующие пары SHALL пропускаться.

#### Scenario: Pair with cropped variant
- **КОГДА** в датасете есть source+mask и cropped+cropped_mask
- **ТОГДА** `TestDataset` SHALL содержать sample с заполненными cropped-полями

### Requirement: Segmentation metrics

Система SHALL вычислять метрики сегментации по TP/FP/FN/TN (пиксели, порог `>0`): IoU, Dice, Precision, Recall, F1, Accuracy с сглаживающим `smooth=1e-6`.

#### Scenario: Perfect prediction
- **КОГДА** pred-маска идентична gt-маске
- **ТОГДА** IoU/Dice/F1/Precision/Recall SHALL равняться 1.0

#### Scenario: Disjoint predictions
- **КОГДА** pred и gt не пересекаются
- **ТОГДА** IoU/Dice/F1/Precision SHALL равняться 0.0

### Requirement: Runner and aggregation

Система SHALL прогонять алгоритм по датасету (`run_algorithm`), собирать метрики для variant `source` и (при наличии) `cropped`, и усреднять (`_mean_metrics`) в `mean_*`/`std_*`, исключая сырые TP/FP/FN/TN. Для пустого списка `_mean_metrics` SHALL возвращать пустой словарь.

#### Scenario: Empty metrics list
- **КОГДА** `_mean_metrics([])` вызывается
- **ТОГДА** результат SHALL быть пустым словарём

#### Scenario: Run algorithm on dataset
- **КОГДА** `run_algorithm(algo, dataset)` на непустом датасете
- **ТОГДА** результат SHALL содержать по ключу sample-имени словарь с метриками для `source`

### Requirement: HTML report

Система SHALL генерировать HTML-отчёт (`generate_report`) со сводной таблицей и графиками Chart.js, записывая файл по указанному пути.

#### Scenario: Report generated
- **КОГДА** `generate_report(all_results, path)` вызывается с результатами
- **ТОГДА** файл SHALL быть создан и содержать ключевые метрики (`mean_iou`)