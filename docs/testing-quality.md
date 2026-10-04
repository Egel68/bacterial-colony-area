# Тестирование и оценка качества алгоритмов распознавания колоний

Метрики, реестр алгоритмов, планировщик прогонов, кэширование, telemetry и
структура отчётов. Смежные документы: [architecture.md](architecture.md) —
модули и их взаимодействие, [algorithms.md](algorithms.md) — справочник
алгоритмов и контракт моделей, [user-guide.md](user-guide.md) — флаги CLI,
[datasets.md](datasets.md) — первичные источники и версии наборов.

## 1. Общая схема

```
        ┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
        │ Изображение  │      │   Алгоритм       │      │   Маска     │
        │  чашки Петри │─────▶│  детекции        │─────▶│  колоний    │
        │  (BGR)       │      │  (detect())       │      │  (uint8)    │
        └─────────────┘      └──────────────────┘      └─────────────┘
                                                              │
                                                              ▼
        ┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
        │ Эталонная   │      │  compute_         │      │ Метрики:    │
        │  маска      │─────▶│  segmentation_    │─────▶│ IoU, Dice,  │
        │  (ground     │      │  metrics()        │      │ F1, Prec,   │
        │   truth)     │      │                   │      │ Rec, Acc    │
        └─────────────┘      └──────────────────┘      └─────────────┘
```

Каждый алгоритм детекции получает на вход цветное изображение чашки Петри (BGR, `numpy.ndarray`) и возвращает бинарную маску колоний (uint8: 0/255). Эта маска попиксельно сравнивается с эталонной (ground truth) маской — вычисляются 6 метрик сегментации. Результаты агрегируются по всему датасету и экспортируются в HTML-отчёт или JSON.

---

## 2. Входные данные

### 2.1. Структура датасета

Датасет может быть организован **тремя способами**. Система авто-детектит структуру в порядке приоритета:

| Приоритет | Структура | Описание | Пример |
|---|---|---|---|
| 1 | **manifest** | Есть `dataset.json` с манифестом (ManifestAdapter-совместимый) | Импортированный датасет 22022540 |
| 2 | **legacy** | `source/` + `masks/` — отдельная папка для масок | `test_images/` |
| 3 | **importer** | `source/*.jpg` + `source/*_mask.png` — маски в той же папке | CocoBboxImporter-output |

**Legacy-структура** (`test_images/`):
```
test_images/
├── source/           # Исходные изображения чашек Петри
│   ├── img001.jpg
│   └── img002.jpg
├── masks/            # Эталонные маски (имена с _mask)
│   ├── img001_mask.jpg
│   └── img002_mask.jpg
├── cropped/          # Обрезки чашек Петри (опционально)
│   ├── img001_cropped.jpg
│   └── img002_cropped.jpg
└── cropped_masks/    # Маски обрезок (опционально)
    ├── img001_cropped_mask.jpg
    └── img002_cropped_mask.jpg
```

**Importer-структура** (CocoBboxImporter, `--crop`):
```
dataset/
├── source/
│   ├── img001.jpg
│   ├── img001_mask.png      # маска в той же папке
│   ├── img002.jpg
│   └── img002_mask.png
├── cropped/
│   ├── img001_cropped.jpg
│   ├── img001_cropped_mask.png
│   ├── img002_cropped.jpg
│   └── img002_cropped_mask.png
└── dataset.json             # манифест
```

**Manifest-структура** (любой датасет с `dataset.json`):
```json
{
  "name": "22022540_imported",
  "origin": "external",
  "mask_mode": "binary",
  "storage": "copy",
  "samples": [
    {
      "id": "sp01_img01",
      "kind": "source",
      "image": "source/sp01_img01.jpg",
      "mask": "source/sp01_img01_mask.png",
      "dish": {"cx": 1000, "cy": 800, "r": 700}
    },
    {
      "id": "sp01_img01_cropped",
      "kind": "cropped",
      "image": "cropped/sp01_img01_cropped.jpg",
      "mask": "cropped/sp01_img01_cropped_mask.png"
    }
  ]
}
```

### 2.2. Варианты снимков

Каждый снимок существует в двух вариантах:

- **source** — полноразмерное изображение чашки Петри. Алгоритм сначала ищет чашку на изображении (через `detect_petri_dish`), затем детектирует колонии внутри неё.
- **cropped** — обрезка изображения по кругу чашки Петри, фон залит чёрным. Алгоритм получает `is_cropped=True` и использует геометрию чашки по центру (`cx=w/2, cy=h/2, r=min(w,h)/2`) вместо поиска.

Метрики считаются **отдельно** для source и cropped вариантов — это позволяет оценить вклад детекции чашки Петри в общее качество распознавания колоний.

---

## 3. Алгоритмы детекции

### 3.1. Единый интерфейс

Все алгоритмы реализуют абстрактный класс `BaseDetectionAlgorithm`:

```python
class BaseDetectionAlgorithm(ABC):
    name: str            # Отображаемое имя
    description: str     # Описание параметров
    detect(image: np.ndarray, is_cropped: bool = False) -> np.ndarray
```

- `image` — BGR-изображение (H×W×3, uint8)
- `is_cropped` — флаг, что чашка уже обрезана по кругу
- Возвращает бинарную маску (H×W, uint8: 0 или 255)

### 3.2. Классические алгоритмы (OpenCV)

Четыре зарегистрированных варианта с фиксированными параметрами:

| Алгоритм | Sensitivity | Margin | Min Size | Contrast | Solid Fill | IoU* |
|---|---|---|---|---|---|---|
| ClassicDefault | 0.5 (50%) | 10% | 50 px | 1.0 | Нет | 0.193 |
| ClassicHighSensitivity | 0.7 (70%) | 5% | 30 px | 1.0 | Нет | **0.211** |
| ClassicSolidFill | 0.3 (30%) | 15% | 50 px | 1.0 | Да (strength=15) | 0.151 |
| ClassicLowSensitivity | 0.3 (30%) | 10% | 100 px | 1.0 | Нет | 0.156 |

*Средний IoU на датасете 22022540 (369 source-изображений).

**Pipeline классического алгоритма** (`ColonyDetector.detect_colonies`):
1. Извлечение зелёного канала (колонии — белые/прозрачные на тёмном фоне)
2. CLAHE с усилением контраста (clip_limit = `2.0 × contrast`)
3. Медианный blur (размер 5) — удаление шума
4. Вычитание фона: GaussianBlur (51×51), `1.5×img − 0.5×bg`
5. Адаптивная бинаризация: `threshold = mean + k × std`, где `k = 3.0 − sensitivity × 2.5`, с клампом в `[mean + 5, 254]`
6. Морфология OPEN (3×3 эллипс, 1 итерация) — удаление мелких шумов
7. Обычный режим: CLOSE (3×3 эллипс, 2 итерации) — склейка разрывов; SOLID FILL: CLOSE с ядром `fill_strength` + заливка контуров `FILLED`
8. Connected Components (связность 8) с фильтром `area >= min_colony_size`

Рабочая зона сужается до `radius · (100 − margin)/100` с практическим минимумом
в 1% (`max(margin_percent, 1.0)`), а статистики порога считаются **только** по
пикселям этой зоны.

**Детекция чашки Петри** (`detect_petri_dish`):
1. Поиск ярких объектов: GaussianBlur(9×9, σ=2), threshold 200, морфология CLOSE (30×30), `cv2.minEnclosingCircle` для самого большого контура
2. Fallback: `cv2.HoughCircles` (dp=1.2, minDist=min_dim/2, param1=100, param2=30)
3. Если чашка не найдена — искусственная окружность по центру (`r = 0.4 × min(w,h)`)

### 3.3. Нейросетевые алгоритмы (ONNX)

Адаптеров два, оба CPU-only (`providers=["CPUExecutionProvider"]`), оба
импортируют `onnxruntime` лениво — приложение работает и без него.

#### `OnnxModelAlgorithm` — режим resize

`testing/onnx_algorithm.py`:

1. BGR → RGB (цветовой формат)
2. Resize до 512×512 (INTER_LINEAR)
3. Нормализация ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
4. Инференс через `onnxruntime.InferenceSession`
5. Resize выхода обратно к исходному размеру (INTER_LINEAR)
6. Бинаризация порогом 0.5 (`raw > threshold`) → маска uint8 (0/255)

Выход модели трактуется как вероятность — сигмоида **не** применяется.

#### `TiledOnnxModelAlgorithm` — полное разрешение

`testing/tiled_onnx_algorithm.py`: кадр режется на тайлы 512×512 с шагом
`stride=384` (перекрытие 128 px). Каждый тайл нормализуется и прогоняется
отдельно, вероятности **усредняются** по перекрытию, затем применяется порог
(`>= threshold`). Сигмоида применяется, если `output_is_logits=True` (по
умолчанию). Метод `detect_with_progress(image, is_cropped, progress_callback)`
сообщает число обработанных тайлов — его использует окно анализа для полосы
прогресса. Параметр `is_cropped` игнорируется: тайлы покрывают весь кадр.

Выбор адаптера делает `scan_bundled_models()` по имени файла в `models/`:
`colony_mobilenet_v3_small` → тайловый, `colony_seg` и всё остальное → resize.
Флаг `--model` и кнопка «Загрузить модель» такого выбора **не** делают —
всегда создаётся `OnnxModelAlgorithm`.

#### Регистрация моделей

Модели из `models/*.onnx` регистрируются как `NN:<stem>`. Вызов
`register_bundled_models()` есть в `main.py` и в `AnalysisWindow`, но **нет**
в `testing/__main__.py` — поэтому в CLI модели `NN:*` появляются только через
`--model` (см. [architecture.md § 3](architecture.md#регистрация-встроенных-моделей)).

Готовая модель `models/colony_seg.onnx`: IoU 0.72, Dice 0.83 на валидации
датасета 22022540. Полный справочник — [algorithms.md](algorithms.md).

---

## 4. Метрики качества

### 4.1. Попиксельное сравнение

Для каждой пары `(pred_mask, gt_mask)` вычисляются TP/FP/FN/TN:

```python
pred = (pred_mask > 0).astype(uint8)      # порог > 0
gt   = (gt_mask > 0).astype(uint8)

tp = sum(pred == 1 AND gt == 1)   # истинно-положительные пиксели
fp = sum(pred == 1 AND gt == 0)   # ложно-положительные
fn = sum(pred == 0 AND gt == 1)   # ложно-отрицательные
tn = sum(pred == 0 AND gt == 0)   # истинно-отрицательные
```

### 4.2. Формулы метрик (сглаживание ε = 1e-6)

| Метрика | Формула | Диапазон | Смысл |
|---|---|---|---|
| **IoU** (Intersection over Union) | `TP / (TP + FP + FN + ε)` | 0–1 | Доля перекрытия предсказанной и эталонной маски |
| **Dice** (F1 по пикселям) | `2·TP / (2·TP + FP + FN + ε)` | 0–1 | Гармоническое среднее Precision и Recall |
| **Precision** | `TP / (TP + FP + ε)` | 0–1 | Доля верно предсказанных колоний среди всех предсказанных |
| **Recall** | `TP / (TP + FN + ε)` | 0–1 | Доля найденных колоний среди всех эталонных |
| **F1** | `2·P·R / (P + R + ε)` | 0–1 | Гармоническое среднее Precision и Recall |
| **Accuracy** | `(TP + TN) / (TP + FP + FN + TN + ε)` | 0–1 | Доля верно классифицированных пикселей |

### 4.3. Агрегация по датасету

Для каждого алгоритма метрики агрегируются **отдельно** по source и cropped вариантам:

```python
def _mean_metrics(metrics_list):
    for key in [iou, dice, f1, precision, recall, accuracy]:
        mean_key = mean(values)
        std_key  = std(values)        # population std
    return {mean_iou, std_iou, mean_dice, std_dice, ...}
```

Сырые счётчики (TP, FP, FN, TN) не агрегируются — только производные метрики.

### 4.4. Парное сравнение алгоритмов

Для двух выбранных алгоритмов A и B на каждом снимке для каждой метрики фиксируется «победитель» — алгоритм с большим значением метрики. Результат — таблица «снимок × variant × метрика → A/B/ничья».

---

## 5. Запуск тестирования

### 5.1. CLI (режим отчёта)

```bash
# Полный прогон всех алгоритмов на test_images/ → HTML-отчёт
uv run test-algorithms

# Выборочные алгоритмы + внешняя модель + парное сравнение
uv run test-algorithms \
    --data-root test_images \
    --output report.html \
    --algorithms ClassicDefault,ClassicSolidFill \
    --model path/to/model.onnx \
    --compare ClassicDefault,ClassicSolidFill \
    --no-per-snapshot
```

### 5.2. CLI (режим baseline — JSON)

```bash
# Прогон 4 классических алгоритмов → JSON для сравнения с нейросетью
uv run baseline \
    --data-root datasets/22022540_imported \
    --output baseline.json

# Без cropped-вариантов
uv run baseline \
    --data-root test_images \
    --output baseline.json \
    --no-cropped
```

Режим `baseline` совпадает с `test-algorithms --mode baseline`; отдельная
консольная команда существует, чтобы не набирать `--mode` каждый раз.

### 5.3. GUI

В приложении: главное окно → «Тестирование алгоритмов» → выбор датасета, чекбоксы алгоритмов, загрузка моделей, запуск, экспорт HTML-отчёта.

Окно использует `BaselineDataset` и принимает manifest (`dataset.json`), legacy
`source/` + `masks/` и importer-структуры. CLI-режим `report` отдельно использует
`TestDataset` и обычные пары в четырёх каталогах. Модели `NN:*` в списке по
умолчанию отсутствуют — их нужно загрузить кнопкой.
Источники датасетов и их опубликованные версии перечислены в
[каталоге датасетов](datasets.md); наличие локальной папки не подтверждает
побайтовое совпадение с опубликованным архивом.

### 5.4. CLI (режим evaluate — каталог прогона)

```bash
uv run evaluate --data-root test_images

# Импортировать сырой COCO-датасет и тут же измерить
uv run evaluate --data-root ./out --import-root datasets/22022540

# Конфиг-файл + телеметрия
uv run evaluate --config eval.yaml --telemetry --performance-output ./perf.json
```

`evaluate` создаёт отдельный каталог прогона и складывает в него всё, что
нужно для воспроизводимости:

```
evaluations/2026-09-15_19-44-10/
├── report.json      полная выгрузка метрик
├── report.html      самодостаточный отчёт с Chart.js
├── run_info.json    run_id, timestamp, путь и размер датасета, алгоритмы, git
├── config.yaml      снимок конфигурации (config.json, если нет PyYAML)
└── performance.json (+ .jsonl)   телеметрия, если включена
```

Приоритет конфигурации: `DEFAULT_CONFIG` ← файл `--config` ← CLI. При этом
значения, которые argparse вернул как дефолты (а не `None`), **перетирают**
файл конфигурации. Чтобы параметр из YAML действительно применялся, не
полагайтесь на дефолт парсера.

`--sample-limit` отключает чтение агрегатного кэша, чтобы прогон на
подмножестве не испортил кэш полного датасета.

---

## 6. Производительность: батчи, потоки, кэш, телеметрия

Все режимы оценки, включая GUI, выполняются одним и тем же
`scheduler.execute_pipeline`.

### Батчи и память

`iter_batches(dataset, batch_size=8, memory_budget=None, ...)` накапливает
сэмплы, пока не выполнится **любое** из условий:

- набралось `batch_size` сэмплов;
- сумма `image.nbytes + mask.nbytes` превысила `memory_budget`.

Превышение бюджета **одним** сэмплом прогон не останавливает — оно
фиксируется событием `memory_budget_exceeded` в телеметрии. Нечитаемая пара
(битый файл) не роняет прогон: ошибка попадает в лог и телеметрию, сэмпл
пропускается.

### Ресурсная политика и число потоков

Число workers координируется через `testing/resource_policy.py` и
`testing/scheduler.py`, чтобы учитывать не только внешние algorithm tasks, но и
декодирование и внутренние пулы OpenCV/ONNX Runtime.

- Видимая CPU-ёмкость — консервативный минимум известных ограничений процесса:
  affinity, cgroup CPU quota и физические CPU хоста. Если информацию получить
  нельзя, используется консервативный fallback.
- `resolve_resource_policy` распределяет ёмкость между algorithm workers,
  decode workers и native threads. В interactive-режиме оставляется запас для
  GUI event loop.
- `--workers N` — верхняя граница, не обещанное фактическое число потоков;
  итог также ограничен задачами, CPU-ёмкостью и памятью. `workers <= 0` ведёт
  к `ValueError`.
- `_effective_workers` дополнительно ограничивает конкурентные задачи при
  заданном `memory_budget`; адаптивное снижение может выдать `workers_reduced`
  при давлении ресурсов.

**Важно:** concurrency — настройка/предел планирования, а не доказательство
ускорения. Сравнивайте end-to-end время и throughput на одном наборе, проверяя
идентичность результатов и память. Зафиксированный warm-cache acceptance
workload дал одинаковые SHA-256 результатов, но automatic режим оказался
примерно в 3.16 раза медленнее последовательного; cold-storage physical I/O
не измерялся. Это ограниченное измерение, не утверждение об ускорении или
замедлении всех машин и алгоритмов.

Реестровые **экземпляры** (включая текущие `NN:*`) по-прежнему защищены
именованным `threading.Lock` и исполняются последовательно; зарегистрированные
**классы** получают отдельные экземпляры на поток. Координация нативных thread
pools не снимает это ограничение для instance-алгоритмов и не обещает, что
увеличение `--workers` распараллелит вызов одной ONNX-сессии.

### Перекрытие decode и compute

`iter_batches` готовит bounded batches, а `PrefetchLoader` держит максимум один
prefetched batch по умолчанию и перекрывает его decode с compute текущего.
Ограничения очереди и памяти действуют совместно (backpressure). Для
`memory_budget` 25% резерва оставляется под временные буферы алгоритмов;
decoded-input target не является жёстким лимитом RSS. Если одна пара превышает
бюджет, она обрабатывается отдельно, а не отбрасывается. Ошибка пары не должна
терять прочие допустимые пары.

`input_starvation` измеряет долю активных compute worker-slot-seconds, когда
готовый compute slot ожидал пустую input queue; старт/финиш и намеренное ожидание
сериализованного algorithm lane исключаются из этого числителя.

### Два независимых кэша

| Механизм | Что кеширует | Где лежит | Ключ |
|---|---|---|---|
| Сигнатура датасета | Готовые метрики по алгоритмам | `{data_root}/.cache/{sha256}.json` | Хэш от списка `путь:размер` файлов в `source/`, `masks/`, `cropped/`, `cropped_masks/` + `dataset.json` |
| `PredictionCache` | Маски предсказаний (`--cache`) | `{data_root}/.cache/preds/{algo}/{sha256}.png` | `algo_name` + `params` + байты изображения |

Ограничения, о которых стоит помнить:

- Сигнатура учитывает **только размеры** файлов: правка содержимого при том
  же размере кэш не инвалидирует.
- `--clear-cache` удаляет только файл с текущей сигнатурой, а не каталог
  `.cache/`; кэш предсказаний он не трогает.
- `PredictionCache.put*` не перезаписывает уже существующий файл, а `get*`
  возвращает `None`, если PNG не читается, — «битый» кэш не валит прогон.
- В ключ предсказания входит `is_cropped`, поэтому исходник и обрезок одного
  объекта не путаются.

### Телеметрия

`--telemetry` включает `TelemetryCollector`, который пишет два файла:

| Файл | Содержимое |
|---|---|
| `performance.json` (путь из `--performance-output`) | Итоговая сводка: стадии с перцентилями, счётчики, ошибки, число сериализованных вызовов |
| `performance.jsonl` | Поток событий: `batch_started`, `batch_loaded`, `workers_reduced`, `memory_budget_exceeded`, снапшоты очереди с интервалом `--telemetry-interval` |

Отслеживаемые стадии включают `file_read` (логическое чтение сжатых файлов),
`image_decode` (декодирование), `input_wait` (ожидание входа вычислителем),
`queue_wait` (ожидание отправленной задачи), `cache`, `detect`, `metrics` и
`report`. Telemetry также записывает logical I/O bytes, доступность CPU
(logical/physical count, affinity, quota), worker/native-thread limits, process
CPU time/cores consumed, текущую/пиковую RSS, input-starvation и capability
системных I/O-счётчиков. Если OS counters недоступны, они остаются неизвестными;
ноль physical read bytes может означать чтение из OS page cache, а не отсутствие
чтения или задержки.

### Измерения и границы выводов

Reference heartbeat-тесты проверяют, что Qt event loop продолжает обрабатывать
timer и пользовательские события во время длительной фоновой CPU-bound работы.
Латентность зависит от платформы и внешней нагрузки; за пределами зафиксированного
reference-профиля тест подтверждает отсутствие синхронной длительной работы в GUI,
а не универсальную численную гарантию.

Performance acceptance публикует последовательный и автоматический/ограниченно
параллельный прогоны с одним и тем же входом, digest результатов, end-to-end
wall/CPU time, throughput, RSS и input-starvation. Не следует выводить ускорение
из увеличенного worker count или меньшего starvation: для зафиксированного
warm-cache classic workload совпадение результатов было достигнуто, однако auto
режим был ~3.16x медленнее sequential. Холодное хранилище и физические дисковые
чтения этим benchmark не измерены.

---

## 7. Структура baseline отчёта (JSON)

```json
{
  "dataset_name": "22022540_imported",
  "image_count": {"source": 369, "cropped": 0},
  "timestamp": "2026-09-05T20:16:14.238053+00:00",
  "algorithms": [
    {
      "name": "ClassicDefault",
      "description": "Стандартные параметры: sensitivity=0.5, ...",
      "source": {
        "mean_iou": 0.1929, "std_iou": 0.1801,
        "mean_dice": 0.2866, "std_dice": 0.2452,
        "mean_f1": 0.2866, "std_f1": 0.2452,
        "mean_precision": 0.4983, "std_precision": 0.4284,
        "mean_recall": 0.2350, "std_recall": 0.1925,
        "mean_accuracy": 0.8969, "std_accuracy": 0.0945,
        "num_samples": 369
      }
    }
  ]
}
```

---

## 8. Baseline на датасете 22022540

Датасет 22022540: 369 изображений, 56 865 аннотированных колоний, 24 вида бактерий из коллекции НИИ антимикробной химиотерапии (Смоленск). Изображения импортированы через `CocoBboxImporter`, боксы растризованы во вписанные эллипсы бинарной маски.

| Алгоритм | IoU | Dice | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| ClassicDefault | 0.193 ± 0.180 | 0.287 ± 0.245 | 0.287 ± 0.245 | 0.498 ± 0.428 | 0.235 ± 0.193 | 0.897 ± 0.095 |
| ClassicHighSensitivity | **0.211** ± 0.194 | **0.307** ± 0.260 | **0.307** ± 0.260 | 0.405 ± 0.388 | **0.306** ± 0.218 | 0.862 ± 0.118 |
| ClassicSolidFill | 0.151 ± 0.156 | 0.232 ± 0.221 | 0.232 ± 0.221 | **0.532** ± 0.440 | 0.177 ± 0.171 | 0.904 ± 0.104 |
| ClassicLowSensitivity | 0.156 ± 0.164 | 0.238 ± 0.230 | 0.238 ± 0.230 | 0.515 ± 0.437 | 0.181 ± 0.176 | **0.909** ± 0.097 |

**Наблюдения:**
- ClassicHighSensitivity даёт лучший IoU (0.211) и Recall (0.306), но низкую Precision (0.405) — много ложных срабатываний
- ClassicSolidFill и ClassicLowSensitivity дают лучшую Precision (0.532 и 0.515) и Accuracy, но жертвуют Recall
- Нейросетевая модель (`models/colony_seg.onnx`) на валидации показывает IoU 0.72 — это существенно выше всех классических алгоритмов

---

## 9. Архитектура кода

Полный разбор слоёв и сигнатур — в [architecture.md § 6](architecture.md#6-пакет-testing).
Здесь только карта модулей, участвующих в оценке.

```
testing/
├── __init__.py            # импорт classic_algorithms: регистрация 4 классических
├── __main__.py            # CLI: test-algorithms (report/baseline/evaluate), baseline, evaluate
├── interface.py           # BaseDetectionAlgorithm (ABC)
├── registry.py            # @register_algorithm + register_algorithm_instance
├── classic_algorithms.py  # 4 OpenCV-алгоритма с фиксированными параметрами
├── onnx_algorithm.py      # OnnxModelAlgorithm, scan_bundled_models, register_bundled_models
├── tiled_onnx_algorithm.py# TiledOnnxModelAlgorithm, tile_positions — полное разрешение
├── metrics.py             # compute_segmentation_metrics (6 метрик + tp/fp/fn/tn)
├── statistics.py          # описательная статистика, Wilcoxon, доли побед, выбросы
├── scheduler.py           # execute_pipeline: CPU-aware, memory-bounded pipeline
├── pipeline_overlap.py    # bounded decode prefetch, backpressure, input-starvation
├── resource_policy.py     # algorithm/decode/native CPU budget coordination
├── pipeline_observer.py   # progress phases, task diagnostics and cancellation
├── profiling.py           # контекстное измерение стадий
├── cache.py               # сигнатура датасета, PredictionCache (маски в PNG)
├── telemetry.py           # CPU/RSS/I/O/stage telemetry: performance.json + .jsonl
├── dataset.py             # TestDataset: пары source/ + masks/ (+ cropped)
├── baseline.py            # BaselineDataset (manifest/legacy/importer) + run_baseline + export_json
├── runner.py              # run_all, агрегация, Wilcoxon-таблица, выбросы, доли побед
├── evaluator.py           # режим evaluate: load_config, create_run_dir, отчёты run-dir
└── dashboard.py           # generate_report: HTML с Chart.js
```

```
analysis/                    # слой предметной области, используемый классикой
├── colony_detector.py       # ColonyDetector: detect_petri_dish + detect_colonies
├── image_processor.py       # CLAHE, зелёный канал, median blur
├── params.py                # AnalysisParams (dataclass с валидацией диапазонов)
├── geometry.py              # PetriInfo (frozen dataclass)
└── results.py               # AnalysisResult (frozen dataclass)
```

Поток исполнения одинаков для всех режимов и для GUI:

```
TestDataset | BaselineDataset
  → scheduler._sample_refs → iter_batches (batch + decoded-input memory target)
  → PrefetchLoader (bounded decode producer + backpressure; decode/compute overlap)
  → ThreadPoolExecutor: задача = (sample × algorithm), shared CPU/native policy
      → SampleContext / PredictionCache.get_array → [_AlgorithmRuntime.detect] → put_array
      → compute_segmentation_metrics
  → PipelineObserver phases/progress/errors + structured telemetry
  → aggregation (compute_summary / wilcoxon / outliers / winners)
  → dashboard.generate_report | export_json + evaluator._generate_eval_html
```

Подробности: [scheduler](architecture.md#64-планировщик-testingschedulerpy),
[кэш](architecture.md#65-кэш-testingcachepy),
[telemetry](architecture.md#68-telemetry-testingtelemetrypy),
[отчёты](architecture.md#69-отчёты).
