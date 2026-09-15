## Context

См. `proposal.md` — Why. Текущий pipeline уже использует `ProcessPoolExecutor` и плоский fan-out по задачам `algorithm × sample × variant`. Каждая задача отдельно вызывает `cv2.imread()`, поэтому разные процессы повторно читают одни и те же файлы и конкурируют за диск. Увеличение числа процессов не является надёжным ускорением при I/O-bound нагрузке.

Входные изображения и маски представлены NumPy-массивами. `detect()` синхронен, но используемые реализации в основном выполняют OpenCV и ONNX Runtime в native-коде. Это позволяет проверить ограниченный пул потоков, не вводя `asyncio`: asyncio сам по себе не ускорит синхронную CPU-bound функцию.

## Goals / Non-Goals

**Goals:**

- Читать каждый image/GT-файл один раз на batch и повторно использовать массивы для всех выбранных алгоритмов.
- Параллельно запускать алгоритмы на уже загруженных объектах через ограниченный `ThreadPoolExecutor`, не создавая неконтролируемую конкуренцию за CPU или память.
- Сохранять бит-в-бит идентичные предсказания и метрики, включая канонический порядок агрегации.
- Сделать фактическое узкое место измеримым: I/O, загрузка, ожидание очереди, detect, метрики, cache и запись отчёта.
- Выполнять smoke/acceptance-проверки на детерминированном наборе из 5 объектов.

**Non-Goals:**

- Изменение алгоритмов детекции, формата метрик качества или API `BaseDetectionAlgorithm`.
- Использование `asyncio` без изменения синхронного API алгоритмов.
- Возврат к flat process fan-out или введение shared-memory процессов в рамках этого change.
- Параллельная генерация HTML-отчёта.
- Обещание линейного ускорения на любой машине: результат должен подтверждаться telemetry.

## Decisions

### 1. Batch loader вместо process fan-out

Координатор строит детерминированный список ссылок на samples и обрабатывает его bounded batch-ами:

1. batch ограничивается `batch_size` и `memory_budget`;
2. координатор читает image и GT-маску каждого объекта/variant ровно один раз;
3. для batch создаются задачи вида `(algorithm, sample, variant)`, которые получают ссылки на in-memory NumPy-массивы;
4. после завершения всех задач batch освобождается, затем загружается следующий.

Это устраняет повторный `imread()` для каждого алгоритма и ограничивает пиковую память. Внутри batch разные алгоритмы одного объекта могут выполняться параллельно, как и разные варианты/объекты.

Рассматривались:

- flat `ProcessPoolExecutor`: отклонён, потому что повторяет I/O и требует pickling массивов;
- staged I/O + CPU queues: отклонён как избыточно сложный для первой версии при общей памяти потоков;
- shared-memory processes: отклонён, потому что добавляет lifetime/cleanup-сложность и не нужен после устранения повторного I/O;
- последовательная обработка: остаётся fallback для `workers=1` и для сред, где telemetry показывает отсутствие выигрыша.

### 2. Ограниченный пул потоков и настройка concurrency

`--workers` задаёт верхнюю границу количества потоков. При отсутствии явного значения scheduler выбирает консервативный лимит по доступным физическим ядрам, размеру batch, memory budget и числу задач; `os.cpu_count()` не считается безусловно оптимальным значением.

Потоки работают только с массивами текущего batch. Внешний пул не должен умножать внутреннее число потоков OpenCV/ONNX Runtime; runtime-настройки и/или outer limit должны предотвращать oversubscription.

Для проверки стратегии scheduler собирает telemetry первых batch-ей и может уменьшить concurrency при росте queue wait, peak RSS, I/O pressure или отсутствии throughput improvement. Автоматическое увеличение выше заданного лимита не допускается.

### 3. Жизненный цикл алгоритмов и потокобезопасность

Алгоритмы, зарегистрированные как классы, получают thread-local экземпляр, чтобы состояние одного экземпляра не разделялось между потоками. Входные изображения и маски считаются read-only; worker не должен изменять их in-place.

Зарегистрированные готовые экземпляры, включая ONNX-модели, могут не быть безопасными для одновременного `detect()`. Для них runner использует независимые контексты, если это поддерживается адаптером, или serial lane/lock с явным telemetry-событием `serialized_algorithm`. Это сохраняет корректность даже ценой меньшего ускорения.

### 4. Cache без дополнительного чтения изображения

Prediction cache расширяется операциями по digest уже прочитанных image bytes/массивов и параметрам алгоритма. Cache lookup выполняется после batch load и не вызывает повторное чтение исходного image с диска. Cache hit пропускает `detect()`, но метрики по GT считаются обычным образом.

Если прежний path-based cache сохраняется для совместимости, digest вычисляется во время единственного чтения batch и передаётся в cache API; отдельный `open(image_path)` в worker запрещён.

### 5. Telemetry и логирование

Основной режим использует `psutil` и stdlib logging. Сборщик не должен ломать прогон, если отдельный системный counter недоступен: значение записывается как `null`, вместе с capability flag.

Периодические структурированные события пишутся в `performance.jsonl`:

- `run_started`, `run_finished`;
- `batch_started`, `batch_completed`;
- периодические `snapshot` с active workers, queue depth и системными counters;
- `warning`/`error` для fallback, неудач и недоступных counters.

Итоговый `performance.json` содержит:

- wall time и process CPU time;
- process/system CPU utilization;
- RSS и peak RSS;
- read/write bytes, read/write operations, bytes/s и operations/s;
- throughput объектов/s, variants/s и algorithm calls/s;
- latency для `load`, `cache`, `queue_wait`, `detect`, `metrics`, `report` и end-to-end с p50/p95/p99;
- batch size, workers, memory budget, queue depth/active/completed/failed/retried;
- cache hits/misses и число serial-lane вызовов;
- ошибки и причины fallback.

Per-task telemetry-события не включаются по умолчанию. Опциональный debug-профиль может писать событие на каждый `algorithm × sample × variant`, но его overhead и объём логов должны быть видны в итоговом отчёте.

### 6. CLI и smoke-набор

Добавляются опциональные параметры запуска: `--batch-size`, `--memory-budget`, `--workers`, `--telemetry`, `--telemetry-interval`, `--performance-output` и `--sample-limit`. `--sample-limit 5` выбирает первые пять объектов в стабильном отсортированном порядке и используется всеми проверками этого change.

Обычный production/evaluation запуск по умолчанию не ограничивается пятью объектами. GUI сохраняет совместимый вызов runner и получает telemetry через результат/сигналы, не блокируя интерфейс.

### 7. Идентичность результатов

Завершение потоков может происходить в произвольном порядке, поэтому runner сначала складывает результаты по ключу, а затем агрегирует их в каноническом порядке `algorithm → sample → variant`. Сравнение с последовательным запуском выполняется на пяти smoke-объектах с допуском `1e-6` для численных метрик и точным сравнением бинарных масок.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Python GIL уменьшит выигрыш для чисто Python-частей `detect()` | Измерять throughput и CPU в telemetry; использовать `workers=1` как fallback, если потоковый режим не выигрывает |
| OpenCV/ONNX Runtime создают внутренние threads и вызывают oversubscription | Ограничить outer pool, настроить native thread pools и записывать фактические active workers |
| Thread-unsafe готовый algorithm instance | Thread-local context либо serial lane/lock с событием `serialized_algorithm` |
| Batch увеличивает RSS | `batch_size` + `memory_budget`, peak RSS guard и освобождение batch после завершения |
| Один повреждённый файл останавливает весь batch | Ошибка привязывается к sample, остальные задачи продолжаются; итог содержит failed count и причины |
| psutil counter недоступен на платформе | Записывать `null` и capability flag, не подменять измерение нулём |
| Cache может вернуть устаревшее предсказание | Сохранять content digest изображения и параметры алгоритма; несовпадение означает cache miss |
| Подробные логи сами становятся I/O bottleneck | По умолчанию только периодические JSONL-события; per-task режим явно opt-in |
| Потоковый режим не быстрее на конкретном алгоритме | Сравнить с `workers=1`, сохранить telemetry и не считать ускорение гарантированным |

## Migration Plan

1. Реализовать batch loader и новый threaded runner за существующим интерфейсом `run_all()`/`run_baseline()`.
2. Добавить telemetry и опциональные CLI/config параметры с выключенным debug-профилем по умолчанию.
3. Прогнать сравнение sequential vs A3 на smoke-наборе из 5 объектов; только после проверки идентичности включить A3 как default.
4. При регрессии использовать `workers=1` и старый cache/report output как rollback на уровне конфигурации; публичные API не меняются.

## Open Questions

Нет. Размер batch и default concurrency должны быть измерены на этапе реализации на smoke-наборе, но не меняют выбранную архитектуру.
