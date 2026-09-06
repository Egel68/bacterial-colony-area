## 1. Модуль статистики — `testing/statistics.py`

- [x] 1.1 Создать `testing/statistics.py` с функцией `compute_descriptive(values)` → `{ mean, std, median, q1, q3, p5, p95, min, max }` через `numpy`, проверить на синтетических данных (массив [0.1, 0.5, 0.9] → median=0.5, q1≈0.3, q3≈0.7)
- [x] 1.2 Реализовать `detect_outliers(values, z_thresh=3.0)` → список `{ value, z_score }` для элементов за пределами z_thresh, проверить на массиве с выбросом [0.1, 0.2, 0.1, 0.9, 0.15, 0.12] → выброс 0.9
- [x] 1.3 Реализовать `wilcoxon_signed_rank(a, b)` через чистый numpy (ранжирование разностей, нормальное приближение, Z-score → p-value через erf), проверить: идентичные массивы → p≈1.0; [1,2,3] vs [2,3,4] → p > 0; пустой список разностей → `{ p_value: None, message: "identical" }`
- [x] 1.4 Реализовать `compute_winner_fractions(metrics_algo_a, metrics_algo_b, metric_key)` → `{ a_wins, b_wins, ties }` (доли 0..1), проверить: a > b на всех снимках → a_wins=1.0; равенство везде → ties=1.0

## 2. Расширение `runner.py`

- [x] 2.1 ...
- [x] 2.2 ...
- [x] 2.3 ...
- [x] 2.4 ...
- [x] 2.5 ...

## 3. Переработка `dashboard.py`

- [x] 3.1 Расширить сигнатуру `generate_report`: добавить параметры
- [x] 3.2 Расширить `_build_summary_rows`: добавить колонки Median, Q1, Q3, P5, P95, Min, Max для каждой метрики
- [x] 3.3 Реализовать `_build_boxplot_data(summary_data)` — сформировать JSON для box-plot по каждой метрике
- [x] 3.4 Реализовать `_build_scatter_data(all_results)` — сформировать JSON scatter-данных
- [x] 3.5 Реализовать `_build_significance_section(wilcoxon_table)` — HTML-таблица p-value и интерпретации
- [x] 3.6 Реализовать `_build_winner_table(all_winners)` — HTML-таблица доли побед
- [x] 3.7 Реализовать `_build_outlier_table(outliers)` — свёрнутая HTML-таблица выбросов
- [x] 3.8 Расширить HTML_TEMPLATE: добавить секции box-plot, scatter, significance, winner, outlier
- [x] 3.9 Дополнить JavaScript в HTML_TEMPLATE рендерингом box-plot и scatter с подсветкой выбросов
- [x] 3.10 Реализовать экспорт JSON-статистики при `stats_output` не None
- [x] 3.11 Условное отключение: при `include_stats=False` вернуть к текущему поведению

## 4. CLI — `__main__.py`

- [x] 4.1 Добавить флаг `--stats / --no-stats` (store_true, default=True), передавать в `generate_report` как `include_stats`
- [x] 4.2 Добавить флаг `--stats-output` (default=None), передавать как `stats_output`
- [x] 4.3 Проверить: `--no-stats` отключает все enrichments, отчёт идентичен текущему (проверить визуально)

## 5. JSON-экспорт baseline — `testing/baseline.py`

- [x] 5.1 Расширить функцию для включения медианы, Q1, Q3, P5, P95, min, max в JSON-отчёт
- [x] 5.2 Проверить: сгенерированный JSON содержит поля `median_*`, `q1_*`, `q3_*`, `p5_*`, `p95_*`, `min_*`, `max_*`

## 6. Валидация и тестирование

- [x] 6.1 Запустить `uv run test-algorithms --stats` — HTML генерируется без ошибок, все секции присутствуют
- [x] 6.2 Запустить `uv run test-algorithms --no-stats` — отчёт без enrichments (8 колонок: Algorithm, Description, Images, IoU, Dice, F1, Precision, Recall)
- [x] 6.3 Проверить JSON-экспорт: stats файл читается `json.load()`, содержит median, q1, q3, p5, p95, min, max, Wilcoxon p-value