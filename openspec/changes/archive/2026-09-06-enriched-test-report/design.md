## Context

Сейчас HTML-отчёт (`testing/dashboard.py`) выводит сводную таблицу с mean±std (от `runner.compute_summary`) и четыре Column-графика Chart.js. `runner._mean_metrics` агрегирует метрики для каждого алгоритма в mean+std. Новые требования (см. spec) добавляют медиану, IQR, процентили, min/max, box-plots, scatter-plot, Wilcoxon signed-rank test, таблицу выбросов, таблицу доли побед и JSON-экспорт. Всё это вычисляется на уже собранных per-sample метриках — никаких новых прогонов алгоритмов не требуется.

## Goals / Non-Goals

**Goals:**
- Ввести модуль `testing/statistics.py` с функциями дескриптивной статистики, детекции выбросов и Wilcoxon теста
- Расширить `runner.compute_summary` для включения дополнительных статистик
- Расширить `runner` публичными функциями `wilcoxon_test`, `detect_outliers`, `compute_winner_fractions`
- Переработать `dashboard.py`: новые секции HTML, новые Chart.js графики, условное включение через параметры
- Добавить CLI-флаг `--stats`/`--no-stats` в `__main__.py`
- Обратная совместимость: `generate_report(all_results, "r.html")` работает как раньше, но с enrichments
- `--no-stats` возвращает ровно текущее поведение (mean±std + старые графики)

**Non-Goals:**
- Не добавлять scipy как обязательную зависимость: Wilcoxon реализуем через чистый Python (scipy опционально, fallback на приближение)
- Не менять сигнатуру существующих CLI-флагов
- Не переписывать HTML-шаблон целиком — только расширять его

## Decisions

### Decision 1: Отдельный модуль `testing/statistics.py`
- **Решение:** Вынести все статистические функции в `testing/statistics.py`
- **Rationale:** `runner.py` уже 127 строк и растёт; статистика — ортогональная ответственность
- **Alternative:** Добавить в `runner.py` — отвергнуто (смешение responsibilities)
- **Alternative:** Встроить в `dashboard.py` — отвергнуто (логика и представление разделены)

### Decision 2: Wilcoxon signed-rank без scipy
- **Решение:** Реализовать Wilcoxon signed-rank через чистый Python/numpy (ранжирование разностей, нормальное приближение для n > 20, точные таблицы для малых n)
- **Rationale:** scipy — тяжёлая зависимость (~15 MB), `.venv` runtime её не содержит; для reproducibility достаточно нормального приближения
- **Trade-off:** Для n < 20 p-value будет менее точным, чем scipy; на реальных датасетах (сотни снимков) разница пренебрежима
- **Alternative:** Сделать scipy опциональным — отвергнуто (усложнение кода без необходимости: нормальное приближение достаточно для исследовательских целей)

### Decision 3: Box-plot через Chart.js кастомный datalabels/плагин
- **Решение:** Использовать нативный Chart.js bar + error bars (или `chartjs-chart-box-and-violin`), а если плагин не подходит — рисовать box-plot как кастомный chart через stacked bar + scatter
- **Rationale:** Chart.js 4.4.7 уже подключён в отчёте; box-plot не встроен, но рисуется через `errorBars` плагин или ручную разметку canvas
- **Fallback:** Самый надёжный — отрендерить box-plot как комбинацию `bar` (IQR) + `scatter` (медиана, усы, выбросы) без внешних плагинов

### Decision 4: Формат обогащённых данных для Chart.js
- **Решение:** Передавать в JavaScript не только mean, но и `{ q1, median, q3, min, max, outliers }` для каждой метрики каждого алгоритма
- **Rationale:** Один источник данных — меньше мест, где могут рассинхронизироваться значения

### Decision 5: JSON-экспорт отдельным файлом
- **Решение:** Флаг `--stats-output` (опционально, по умолчанию `<output>.stats.json` при `--stats`)
- **Rationale:** JSON отдельно от HTML — можно анализировать скриптами, не парся HTML

## Risks / Trade-offs

- **Риск:** Chart.js без плагина box-plot выглядит менее аккуратно → **митигация:** использовать `chartjs-chart-box-and-violin` если CDN доступен, fallback на кастомный рендер
- **Риск:** Увеличение размера HTML-файла (сотни снимков × 6 метрик × статистики) → **митигация:** outlier-таблица сворачивается (details/summary), scatter данные передаются как сжатый JSON
- **Риск:** `--no-stats` должен давать побитово тот же HTML, что и сейчас → **митигация:** snapshot-тест с эталонным `test_report.html`
- **Риск:** Wilcoxon нормальное приближение может быть неточным при малом числе снимков → **принято:** для исследовательских целей достаточно directional signal
- **Trade-off:** Дополнительные параметры `generate_report` (8 штук) — сложнее API → **принято:** все со значениями по умолчанию, backwards compatible