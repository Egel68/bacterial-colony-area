## ADDED Requirements

### Requirement: Petri dish detection

Система SHALL обнаруживать чашку Петри на изображении. Сначала система SHALL пробовать путь через яркие объекты: `GaussianBlur(9×9, σ=2)`, бинаризация по порогу 200, морфологическое закрытие эллиптическим ядром 30×30, поиск внешних контуров. Контур SHALL быть кандидатом, если его площадь ≥ 10% площади изображения, а расстояние от центра контура до центра изображения ≤ 30% от `min(h, w)`. Среди кандидатов система SHALL выбрать контур с максимальной площадью и построить по нему `minEnclosingCircle` для центра и радиуса.

Если подходящих контуров нет, система SHALL использовать fallback через `HoughCircles` (dp=1.2, minDist=`min_dim/2`, param1=100, param2=30, minRadius=0.3·`min_dim`, maxRadius=0.48·`min_dim`). Если и HoughCircles не нашёл чашку, система SHALL вернуть `(None, None)`.

#### Scenario: Image loads with specular highlights
- **КОГДА** сумма ярких пикселей после порога 200 достаточно велика и найден контур с area ≥ 10% площади и близким к центру
- **ТОГДА** функция SHALL вернуть круговую маску и `PetriInfo`, построенные через `minEnclosingCircle` выбранного контура

#### Scenario: No highlight contour, Hough fallback
- **КОГДА** не найдено ни одного подходящего яркого контура
- **ТОГДА** функция SHALL вызвать `HoughCircles`; при найденной окружности — вернуть маску и `PetriInfo`

#### Scenario: No dish detected at all
- **КОГДА** ни яркие объекты, ни `HoughCircles` не дали чашку
- **ТОГДА** функция SHALL вернуть `(None, None)`

#### Scenario: PetriInfo invariant validation
- **КОГДА** создаётся `PetriInfo(cx, cy, radius, image_shape)`
- **ТОГДА** SHALL выполняться инварианты: радиус положительный, центр неотрицательный и внутри границ `image_shape`

### Requirement: Analysis parameter validation

Система SHALL задавать параметры анализа через `AnalysisParams` (dataclass) и валидировать их в `__post_init__`. Допустимые диапазоны: `sensitivity` [0.01, 1.0], `contrast` [0.5, 3.0], `margin_percent` [0, 30], `min_colony_size` [1, 1000], `fill_strength` [1, 100].

#### Scenario: Default params construct without error
- **КОГДА** создаётся `AnalysisParams()` со значениями по умолчанию
- **ТОГДА** конструктор SHALL не выбрасывать исключение

#### Scenario: Out-of-range parameter is rejected
- **КОГДА** создаётся `AnalysisParams(sensitivity=1.5)` (или другое значение вне диапазона)
- **ТОГДА** конструктор SHALL выбросить `AssertionError` с сообщением о допустимом диапазоне

### Requirement: Colony detection pipeline

Система SHALL детектировать колонии внутри области интереса следующим образом: извлечение зелёного канала, CLAHE с `clip_limit = 2·contrast` и сеткой 8×8, медианное размытие (нечётное ядро `blur_size`, не менее 3), вычитание фона `GaussianBlur(51×51)` с `addWeighted(1.5, −0.5)`, ограничение маской `roi` (внутренний радиус чашки за вычетом `margin_percent`), адаптивный порог для области: `k = 3.0 − 2.5·sensitivity`, порог `mean + k·std` (ограничен диапазоном [mean+5, 254]), морфология OPEN ядром 3×3.

При `solid_fill=true` система SHALL применять морфологическое закрытие ядром `fill_strength` (минимум 3) и заливать внутренние области контуров. При `solid_fill=false` система SHALL применять CLOSE ядром 3×3 в две итерации. Итог SHALL фильтроваться по connected components: сохраняется компонент, только если его площадь ≥ `min_colony_size`.

#### Scenario: ROI restricts detection to inner dish
- **КОГДА** `petri_info` передан и `margin_percent > 0`, а яркие пиксели присутствуют вне внутреннего радиуса
- **ТОГДА** результирующая маска SHALL быть нулевой за пределами `roi`-маски

#### Scenario: Empty ROI yields empty result
- **КОГДА** в `roi_mask` нет ни одного валидного пикселя
- **ТОГДА** функция SHALL возвращать нулевую маску и пустой словарь debug-изображений

#### Scenario: Solid fill fills enclosed annular colonies
- **КОГДА** `solid_fill=true` и колония выглядит как кольцо
- **ТОГДА** финальная маска SHALL содержать заполненную внутренность кольца after заливки контуров

#### Scenario: Small components are removed by size filter
- **КОГДА** маска содержит компоненты с площадью <  `min_colony_size`
- **ТОГДА** финальная маска SHALL не содержать такие компоненты

### Requirement: Area and coverage calculation

Система SHALL вычислять площади через `AreaCalculator.calculate_areas`. Площадь колоний в px — число ненулевых пикселей `colony_mask`. Площадь чашки в px: `π·r_full²` при `margin_percent=0`, иначе `π·r_inner²` с `r_inner = r_full·(100−margin_percent)/100`. Площадь чашки в mm² — `π·(диаметр_мм/2)²` (диаметр 90 мм по умолчанию). Коэффициент `px_to_mm2` SHALL равен площади чашки в mm², делённой на площадь чашки в px. Площадь колоний в mm² — `colony_area_px·px_to_mm2`. Покрытие: `colony_area_px / petri_area_px·100`, при нулевой площади чашки — 0. Число колоний SHALL определяться как число connected components (8-связность) без учёта фона.

#### Scenario: Areas with inner margin
- **КОГДА** `calculate_areas` вызывается с `petri_info` и `margin_percent=10`
- **ТОГДА** `petri_area_px` SHALL быть рассчитан по внутреннему радиусу с учётом 10% отступа

#### Scenario: No Petri info given
- **КОГДА** `petri_info=None`
- **ТОГДА** `petri_area_px` SHALL равняться `count_nonzero(petri_mask)`, `petri_area_mm2=0`, `px_to_mm2=0`

#### Scenario: Coverage percent computed
- **КОГДА** обе площади больше нуля
- **ТОГДА** `coverage_percent` SHALL равняться `colony_area_px / petri_area_px·100`

### Requirement: AnalysisController orchestration

The `AnalysisController` SHALL предоставлять `find_petri_dish(image)` и `analyze(image, petri_mask, params, petri_info=None, blur_size=5)`, не зависеть от Qt, хранить последнюю `colony_mask` и `debug_images`, и возвращать `AnalysisResult` (frozen dataclass).

#### Scenario: analyzed returns typed result
- **КОГДА** вызывается `analyze()` с корректным набором параметров
- **ТОГДА** возвращается `AnalysisResult` со всеми полями и `colony_mask`/`debug_images` доступны через свойства