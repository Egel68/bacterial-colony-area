## MODIFIED Requirements

### Requirement: HTML report

Система SHALL генерировать HTML-отчёт (`generate_report`) со сводной таблицей, графиками Chart.js и опциональным разделом парного сравнения алгоритмов, записывая файл по указанному пути. Сигнатура принимает новые опциональные параметры: `include_stats=True`, `include_boxplots=True`, `include_scatter=True`, `include_significance=True`, `include_outliers=True`, `stats_output=None`. При значениях по умолчанию поведение эквивалентно текущему плюс расширенная статистика.

#### Scenario: Report generated

- **КОГДА** `generate_report(all_results, output_path)` вызывается с результатами
- **ТОГДА** файл SHALL быть создан и содержать ключевые метрики (`mean_iou`)

#### Scenario: Report with paired comparison

- **КОГДА** `generate_report(all_results, output_path)` вызывается с включённым парным сравнением
- **ТОГДА** файл SHALL дополнительно содержать таблицу «победитель по каждому снимку» для каждой метрики

#### Scenario: Report with stats disabled

- **КОГДА** `generate_report(all_results, output_path, include_stats=False)` вызывается
- **ТОГДА** отчёт SHALL содержать только mean ± std + существующие графики (обратная совместимость)