# Тестирование и оценка качества алгоритмов распознавания колоний

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
5. Адаптивная бинаризация: `threshold = mean + k × std`, где `k = 3.0 − sensitivity × 2.5`
6. Морфология OPEN (3×3 эллипс) — удаление мелких шумов
7. Опционально SOLID FILL: морфология CLOSE + заливка контуров
8. Connected Components с фильтром по `min_colony_size`

**Детекция чашки Петри** (`detect_petri_dish`):
1. Поиск ярких объектов: GaussianBlur(9×9, σ=2), threshold 200, морфология CLOSE (30×30), `cv2.minEnclosingCircle` для самого большого контура
2. Fallback: `cv2.HoughCircles` (dp=1.2, minDist=min_dim/2, param1=100, param2=30)
3. Если чашка не найдена — искусственная окружность по центру (`r = 0.4 × min(w,h)`)

### 3.3. Нейросетевые алгоритмы (ONNX)

`OnnxModelAlgorithm` адаптирует ONNX-модели к единому интерфейсу:

1. BGR → RGB (цветовой формат)
2. Resize до 512×512 (INTER_LINEAR)
3. Нормализация ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
4. Инференс через `onnxruntime.InferenceSession`
5. Resize обратно к исходному размеру (INTER_LINEAR)
6. Бинаризация порогом 0.5 → маска uint8 (0/255)

Встроенные модели (из `models/*.onnx`) регистрируются автоматически при старте приложения как `NN:<имя_модели>`. Внешние модели подгружаются через GUI или флаг `--model`.

Готовая модель `models/colony_seg.onnx`: IoU 0.72, Dice 0.83 на валидации датасета 22022540.

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
uv run baseline --mode baseline \
    --data-root datasets/22022540_imported \
    --output baseline.json

# Без cropped-вариантов
uv run baseline --mode baseline \
    --data-root test_images \
    --output baseline.json \
    --no-cropped
```

### 5.3. GUI

В приложении: главное окно → «Тестирование алгоритмов» → выбор датасета, чекбоксы алгоритмов, загрузка моделей, запуск, экспорт HTML-отчёта.

---

## 6. Структура baseline отчёта (JSON)

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

## 7. Baseline на датасете 22022540

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

## 8. Архитектура кода

```
testing/
├── __init__.py
├── __main__.py        # CLI entry point (test-algorithms, baseline)
├── interface.py       # BaseDetectionAlgorithm (ABC)
├── registry.py        # @register_algorithm + register_algorithm_instance
├── classic_algorithms.py  # 4 OpenCV-алгоритма с фиксированными параметрами
├── onnx_algorithm.py      # ONNX adapter + scanner bundled models
├── dataset.py         # TestDataset (legacy structure loader)
├── baseline.py        # BaselineDataset + run_baseline + export_json
├── metrics.py         # compute_segmentation_metrics (6 метрик)
├── runner.py          # run_algorithm, run_all, compare_algorithms, _mean_metrics
└── dashboard.py       # generate_report (HTML with Chart.js)
```

```
analysis/
├── colony_detector.py  # ColonyDetector: detect_petri_dish + detect_colonies
├── image_processor.py   # CLAHE, green channel, median blur
├── params.py            # AnalysisParams (dataclass с валидацией)
└── geometry.py          # PetriInfo
```