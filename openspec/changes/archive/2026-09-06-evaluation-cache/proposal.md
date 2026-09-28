## Why

Прогон baseline-метрик на полном датасете (738 снимков × 4 алгоритма) занимает ~2+ часа. При прерывании на 3-м алгоритме или при добавлении новых алгоритмов все предыдущие результаты теряются — перезапуск идёт с нуля. Отсутствие idempotent-прогона делает итеративное тестирование непрактичным.

## What Changes

- В evaluation-пайплайн добавляется persistent-кэш: после успешного прогона каждого алгоритма его усреднённые метрики сохраняются в `.cache/<dataset_signature>.json`
- При повторном запуске `run_evaluate` (или `run_baseline`) на том же датасете уже посчитанные алгоритмы пропускаются
- Сигнатура датасета — SHA256 от конкатенации `relpath:size` всех image+mask-файлов — гарантирует инвалидацию кэша при любом изменении данных
- Старый формат отчёта (`evaluations/<timestamp>/report.json`) сохраняется без изменений — кэш является дополнительным слоем, не заменяет output
- Новый CLI-флаг `--clear-cache` для принудительного пересчёта

## Capabilities

### New Capabilities
- `<skip>`: change does not introduce new spec-level behavior — it adds caching to an existing requirement in `classical-baseline-evaluation`.

### Modified Capabilities
- `classical-baseline-evaluation`: Requirement "JSON baseline report export" расширяется кэшированием результатов по алгоритму; Requirement "Batch run" — пропуском уже посчитанных алгоритмов.

## Impact

- `testing/evaluator.py` — `run_evaluate()`: добавить загрузку/сохранение кэша
- `testing/baseline.py` — `run_baseline()`: принимать опциональный кэш, проверять перед каждым алгоритмом
- `testing/__main__.py` — добавить `--clear-cache`
- CLI и GUI интерфейсы не меняются (кроме опционального флага)