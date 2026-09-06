## Purpose

Расширяет HTML-отчёт тестирования алгоритмов полным набором дескриптивных статистик (медиана, IQR, процентили, min/max, выбросы), боксплотами и scatter-графиками распределения метрик, статистическим тестом значимости различий (Wilcoxon signed-rank), таблицей доли побед по снимкам и экспортом полной статистики в JSON для машинной обработки.

## ADDED Requirements

### Requirement: Summary table includes median, IQR, percentiles

Сводная таблица в HTML-отчёте SHALL для каждой метрики (IoU, Dice, F1, Precision, Recall, Accuracy) дополнительно отображать: median, Q1 (25-й процентиль), Q3 (75-й процентиль), min, max, P5 (5-й процентиль), P95 (95-й процентиль) — в дополнение к существующим mean и std.

#### Scenario: Summary table contains percentile columns

- **WHEN** HTML-отчёт сгенерирован
- **THEN** сводная таблица SHALL содержать столбцы Median, Q1, Q3, P5, P95, Min, Max для каждой метрики

#### Scenario: Numeric correctness

- **WHEN** для алгоритма есть N значений метрики (N ≥ 1)
- **THEN** median, Q1, Q3, P5, P95 SHALL вычисляться через `numpy.percentile` с линейной интерполяцией, min/max — через `numpy.min/numpy.max`

### Requirement: Box-plot charts

HTML-отчёт SHALL содержать box-plot графики (Chart.js) для каждой метрики — один график со всеми алгоритмами, показывающий медиану, Q1/Q3, усы (whiskers) до 1.5×IQR и выбросы за пределами усов.

#### Scenario: Box-plots rendered

- **WHEN** отчёт сгенерирован
- **THEN** HTML SHALL содержать `<canvas id="boxplot_iou">`, `<canvas id="boxplot_dice">`, `<canvas id="boxplot_f1">`, `<canvas id="boxplot_pr">` с Chart.js box-plot рендерингом

#### Scenario: Data for box-plots correct

- **WHEN** box-plot данных переданы в JavaScript
- **THEN** каждый набор данных SHALL содержать min, q1, median, q3, max, outliers

### Requirement: Per-sample scatter plot

HTML-отчёт SHALL содержать scatter-график (Chart.js scatter), где каждая точка — отдельный снимок (по оси X — алгоритм, по оси Y — значение метрики). Точки-выбросы (за ±3σ от среднего алгоритма) SHALL подсвечиваться другим цветом/маркером.

#### Scenario: Scatter plot rendered

- **WHEN** отчёт сгенерирован
- **THEN** HTML SHALL содержать `<canvas id="scatter_iou">` со scatter-данными

#### Scenario: Outliers highlighted

- **WHEN** значение метрики снимка отклоняется более чем на 3σ от среднего по алгоритму
- **THEN** точка на scatter-графике SHALL быть выделена красным цветом

### Requirement: Statistical significance test (Wilcoxon signed-rank)

Система SHALL выполнять парный Wilcoxon signed-rank test для каждой метрики между каждой парой алгоритмов. Результат SHALL включать p-value и бинарную интерпретацию: «значимо лучше» (p < 0.05), «нет значимых различий» (p ≥ 0.05). Тест SHALL использовать попарно-сопоставленные значения метрик source-варианта на одинаковых снимках.

#### Scenario: Wilcoxon p-value computed

- **WHEN** два алгоритма обработали один и тот же набор source-снимков
- **THEN** `scipy.stats.wilcoxon(metrics_a, metrics_b)` SHALL быть вызван для каждой метрики, результат SHALL содержать p-value

#### Scenario: Significance reported in HTML

- **WHEN** отчёт сгенерирован
- **THEN** раздел «Статистическая значимость» SHALL содержать таблицу «алгоритм A × алгоритм B × метрика × p-value × интерпретация»

#### Scenario: Wilcoxon skipped for identical pairs

- **WHEN** оба алгоритма дали идентичные метрики на всех снимках (разности = 0)
- **THEN** Wilcoxon test SHALL NOT выполняться, p-value SHALL быть указан как `—` с пометкой «идентичные результаты»

### Requirement: Winner fraction table

Система SHALL для каждой метрики подсчитывать, на какой доле снимков (0.0–1.0) алгоритм A превосходит алгоритм B. Результат SHALL выводиться в виде матрицы «алгоритм A × алгоритм B × метрика → доля побед A».

#### Scenario: Winner fraction matrix in report

- **WHEN** отчёт сгенерирован с парным сравнением
- **THEN** HTML SHALL содержать таблицу с долей побед для каждой пары алгоритмов по каждой метрике

#### Scenario: Tie fraction included

- **WHEN** значения метрик на снимке равны (в пределах `abs(a - b) < 1e-9`)
- **THEN** снимок SHALL учитываться как ничья, доля ничьих SHALL быть отдельной колонкой

### Requirement: Outlier detection table

Система SHALL детектировать снимки-выбросы (значение метрики алгоритма выходит за ±3σ от среднего по алгоритму) и выводить их в отдельную таблицу с информацией: алгоритм, метрика, имя снимка, variant (source/cropped), значение метрики, z-score.

#### Scenario: Outlier table in report

- **WHEN** отчёт сгенерирован
- **THEN** HTML SHALL содержать таблицу «Выбросы» со столбцами Algorithm, Metric, Image, Variant, Value, Z-score

### Requirement: JSON export of full statistics

Система SHALL экспортировать полную дескриптивную статистику в JSON-файл, содержащий для каждой метрики каждого алгоритма: mean, std, median, q1, q3, p5, p95, min, max, а также матрицу p-value парных Wilcoxon тестов.

#### Scenario: Stats JSON export

- **WHEN** отчёт сгенерирован с запросом JSON-экспорта
- **THEN** файл `<output_path>.stats.json` SHALL быть создан с полной статистикой

### Requirement: CLI --stats flag

Система SHALL принимать CLI-флаг `--stats / --no-stats` (по умолчанию `--stats`), управляющий включением расширенной статистики в отчёт. При `--no-stats` отчёт возвращается к текущему поведению (mean ± std + существующие графики).

#### Scenario: Stats enabled (default)

- **WHEN** CLI запущен без `--no-stats`
- **THEN** отчёт SHALL содержать расширенную статистику

#### Scenario: Stats disabled

- **WHEN** CLI запущен с `--no-stats`
- **THEN** отчёт SHALL содержать только mean ± std и существующие графики (обратная совместимость)

### Requirement: Backward compatibility

Система SHALL сохранять сигнатуру `generate_report(all_results, output_path, include_per_snapshot, comparison)` рабочей. Новые параметры SHALL иметь значения по умолчанию, обеспечивающие эквивалентное текущему поведение при их отсутствии.

#### Scenario: Compatible call

- **WHEN** `generate_report(all_results, "report.html")` вызывается без новых параметров
- **THEN** отчёт SHALL содержать как минимум существующую сводную таблицу mean±std и графики