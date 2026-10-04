# Архитектура кода

Подробный справочник по внутреннему устройству приложения: слои, модули, ключевые
классы и функции, потоки данных и точки расширения.

Документ описывает **код как он есть**. Если поведение расходится с этим описанием,
прав код и обновите документ.

Связанные документы:

| Файл | О чём |
|---|---|
| [user-guide.md](user-guide.md) | Возможности приложения для пользователя: GUI, CLI-флаги, форматы данных |
| [algorithms.md](algorithms.md) | Справочник алгоритмов детекции и контракт моделей |
| [testing-quality.md](testing-quality.md) | Метрики, реестр, кэширование, telemetry, отчёты |
| [dataset-training-pipeline.md](dataset-training-pipeline.md) | Путь COCO-датасета → обученная модель в приложении |

---

## Содержание

- [1. Слои и правило зависимостей](#1-слои-и-правило-зависимостей)
- [2. Карта репозитория](#2-карта-репозитория)
- [3. Точки входа](#3-точки-входа)
- [4. Пакет `analysis`](#4-пакет-analysis)
- [5. Пакет `utils`](#5-пакет-utils)
- [6. Пакет `testing`](#6-пакет-testing)
- [7. Пакет `ui`](#7-пакет-ui)
- [8. Пакет `labeling`](#8-пакет-labeling)
- [9. Пакет `train`](#9-пакет-train)
- [10. Сборка: пакет `scripts`](#10-сборка-пакет-scripts)
- [11. Тесты](#11-тесты)
- [12. Сквозные потоки данных](#12-сквозные-потоки-данных)
- [13. Точки расширения](#13-точки-расширения)
- [14. Подводные камни](#14-подводные-камни)

---

## 1. Слои и правило зависимостей

Проект состоит из пяти слоёв. Стрелки показывают направление зависимостей
(зависимость идёт **сверху вниз**, наружу):

```
┌──────────────────────────────────────────────────────────────┐
│  Слой 5. Точки входа          main.py, train/main.py,        │
│                              testing/__main__.py, ...        │
├──────────────────────────────────────────────────────────────┤
│  Слой 4. GUI                 ui/, labeling/                  │
│                              (PyQt6; бизнес-логика вынесена  │
│                               в ui/controllers/)             │
├──────────────────────────────────────────────────────────────┤
│  Слой 3. Оценка              testing/ (реестр, метрики,      │
│                              scheduler, отчёты)              │
├──────────────────────────────────────────────────────────────┤
│  Слой 2. Домен               analysis/ (детекция),           │
│                              utils/calculations.py           │
├──────────────────────────────────────────────────────────────┤
│  Слой 1. Данные/обучение     train/ (контракт данных,        │
│                              адаптеры, модели, пайплайны)    │
└──────────────────────────────────────────────────────────────┘
```

**Ключевое архитектурное решение.** Бизнес-логика GUI не живёт в виджетах.
Каждое окно делегирует работу контроллеру из `ui/controllers/`, который
**не импортирует Qt-виджеты** — только `numpy`, `cv2` и доменные классы:

| Окно | Контроллер | Что делает контроллер |
|---|---|---|
| `ui/analysis_window.py` | `ui/controllers/analysis_controller.py` | поиск чашки, прогон классики/алгоритма, расчёт площадей, обрезка маски по рабочей зоне |
| `labeling/labeling_window.py` | `ui/controllers/labeling_controller.py` | загрузка/сохранение масок, авто-поиск чашки, обрезка, ZIP-экспорт, листинг файлов |

Следствие: `analysis`, `utils`, `testing` и `train` тестируются без запуска Qt,
а GUI-слой — через `qtbot` (`pytest-qt`).

**Правило зависимости от ML-стека.** `onnxruntime` импортируется **лениво**
внутри `_get_session()`, а не на уровне модуля. `torch` вообще отсутствует в
runtime-зависимостях. Благодаря этому обычный запуск приложения и весь пакет
`testing` (кроме NN-алгоритмов) работают в лёгком окружении `.venv`.

---

## 2. Карта репозитория

Модули и перечень тестов далее описывают текущую структуру; точное количество
файлов не фиксируется, чтобы карта не устаревала при добавлении тестов.

```
bacterial-colony-area/
├── main.py                     Точка входа GUI + --smoke-test-model
│
├── analysis/                   Слой 2: детекция чашки и колоний
│   ├── colony_detector.py      ColonyDetector — основной пайплайн
│   ├── image_processor.py      ImageProcessor — CLAHE, зелёный канал, blur
│   ├── params.py               AnalysisParams + валидация диапазонов
│   ├── geometry.py             PetriInfo — геометрия чашки
│   ├── results.py              AnalysisResult — результат расчёта площадей
│   └── __init__.py             Публичный реэкспорт 5 символов
│
├── utils/                      Слой 2: утилиты
│   ├── calculations.py         AreaCalculator — площади, покрытие, px→mm²
│   ├── config.py               AppConfig, AnalysisDefaults
│   ├── image_loader.py         Загрузка изображений (Qt → OpenCV fallback)
│   └── logging.py              setup_logging()
│
├── testing/                    Слой 3: фреймворк оценки
│   ├── interface.py            BaseDetectionAlgorithm (ABC)
│   ├── registry.py             @register_algorithm, register_algorithm_instance
│   ├── classic_algorithms.py   4 классических алгоритма
│   ├── onnx_algorithm.py       OnnxModelAlgorithm + scan/register bundled
│   ├── tiled_onnx_algorithm.py TiledOnnxModelAlgorithm — полное разрешение
│   ├── metrics.py              compute_segmentation_metrics — 6 метрик
│   ├── statistics.py           Описательная статистика, Wilcoxon, выбросы
│   ├── dataset.py              TestDataset — загрузчик пар source/cropped
│   ├── baseline.py             BaselineDataset — 3 структуры + run_baseline
│   ├── scheduler.py            CPU-aware executor и bounded decode/compute pipeline
│   ├── pipeline_overlap.py     Ограниченный prefetch, backpressure, starvation
│   ├── resource_policy.py      Общий бюджет algorithm/decode/native-потоков
│   ├── pipeline_observer.py    Фазы прогресса/ошибок пайплайна
│   ├── profiling.py            Контекстное измерение стадий обработки
│   ├── cache.py                Сигнатура датасета + PredictionCache (PNG)
│   ├── telemetry.py            CPU/RSS/I/O/stage telemetry — json + jsonl
│   ├── runner.py               run_all, агрегация, Wilcoxon, выбросы, winners
│   ├── dashboard.py            HTML-отчёт Chart.js
│   ├── evaluator.py            Режим evaluate: конфиг, run-dir, отчёты
│   └── __main__.py             CLI: report / baseline / evaluate
│
├── ui/                         Слой 4: GUI
│   ├── background.py           Общий жизненный цикл worker, отмена и deferred close
│   ├── main_window.py          Стартовое окно
│   ├── analysis_window.py      Асинхронный анализ и presentation + ImageLabel
│   ├── testing_window.py       Окно тестирования с фоновыми операциями
│   ├── labeling_session_dialog.py  Диалог «Новая сессия разметки»
│   ├── styles.py               QSS-тема Catppuccin Mocha
│   └── controllers/            Qt-независимая бизнес-логика
│
├── labeling/                   Слой 4: разметка
│   ├── labeling_window.py      PaintLabel + LabelingWindow
│   └── session_manager.py      Сессии, sanitization имени, список «недавних»
│
├── train/                      Слой 1: данные и обучение
│   ├── main.py                 CLI: train / train-compare / list-models / ...
│   ├── config.py               TrainingConfig
│   ├── dataset_manifest.py     Контракт данных (DatasetManifest)
│   ├── dataset_adapters.py     Адаптеры + CocoBboxImporter + load_manifest
│   ├── dataset_split.py        Стратифицированный групповой сплит
│   ├── dataset.py              ColonyDataset, ColonyPatchDataset, сэмплер
│   ├── augment.py              Оффлайн-аугментация (albumentations)
│   ├── train.py                Цикл обучения + TensorBoard
│   ├── training_pipeline.py    Пайплайн `train-colony` целиком
│   ├── evaluation.py           Метрики, tiled-инференс, ONNX parity, отчёты
│   ├── compare.py              train-compare
│   ├── predict.py              ONNX-инференс (не подключён к CLI)
│   ├── reporter.py             Plotly-отчёт по эпохам
│   ├── export_single.py        best.pt → single-file ONNX
│   ├── importer_cli.py         CLI `import-22022540`
│   ├── colony_cli.py           CLI `train-colony`
│   ├── models/                 base.py, unet.py, unet_small.py, mobilenet...
│   └── dashboard/              FastAPI + WebSocket дашборд
│
├── scripts/                    Сборка
│   ├── build_nuitka.sh         Пользовательский скрипт сборки
│   ├── nuitka_build.py         Программный запуск Nuitka, compute_jobs()
│   └── nuitka_flags.py         Генерация --include-package флагов
│
├── models/                     Встроенные ONNX-модели (*.onnx)
├── test_images/                Датасет пар изображение/маска
├── tests/                      Pytest suite: pipeline, GUI heartbeat, cancellation, presentation и regressions
├── docs/                       Документация
└── pyproject.toml              Зависимости, extras, console scripts
```

---

## 3. Точки входа

| Команда | Модуль | Что запускает |
|---|---|---|
| `bacteria-analyzer` | `main:main` | GUI |
| `bacteria-analyzer --smoke-test-model <имя>` | `main:_smoke_test_model` | Проверка встроенной модели без GUI |
| `test-algorithms` | `testing.__main__:main` | Режимы `report` / `baseline` / `evaluate` |
| `baseline` | `testing.__main__:main_baseline` | `main(mode_override="baseline")` |
| `evaluate` | `testing.__main__:main_evaluate` | Отдельный парсер без лишних флагов |
| `augment` | `train.augment:main` | Оффлайн-аугментация |
| `import-22022540` | `train.importer_cli:main` | Импорт COCO-датасета |
| `train-colony` | `train.colony_cli:main` | Сквозной пайплайн обучения |
| `python -m train.main …` | `train.main:main` | Подкоманды `train`, `train-compare`, `list-models`, `dataset-info` |
| `python -m train.export_single` | `train.export_single:main` | Экспорт checkpoint в ONNX |

### Регистрация встроенных моделей

`register_bundled_models()` сканирует `models/*.onnx` и регистрирует каждый файл
как экземпляр алгоритма с именем `NN:<stem>`. Вызов происходит в трёх местах:

| Место | Вызов | Следствие |
|---|---|---|
| `main.py:27` | перед созданием QApplication | NN-алгоритмы доступны в окне анализа |
| `ui/analysis_window.py:452` (`_load_analysis_algorithms`) | при открытии окна | дублируется безвредно: реестр перезаписывает ключ |
| `testing/__main__.py` | **не вызывается** | в `test-algorithms`/`baseline`/`evaluate` модели `NN:*` недоступны |

Практический вывод: чтобы сравнить встроенную модель с классикой через CLI,
передайте файл явно — `--model models/colony_mobilenet_v3_small.onnx`.
Имя зарегистрируется как `NN:colony_mobilenet_v3_small` (тот же префикс `NN:`),
а адаптер выбирается по совпадению `stem` — см. [algorithms.md](algorithms.md).

### Поиск папки `models/`

`_default_models_dir()` в `testing/onnx_algorithm.py` разрешает путь в таком порядке:

1. Nuitka onefile: `Path(sys.executable).parent / "models"` (рядом с бинарником)
2. Nuitka standalone / PyInstaller: каталог данных сборки + `/models`
3. Из исходников: `Path(testing/onnx_algorithm.py).parent.parent / "models"`

---

## 4. Пакет `analysis`

Публичный интерфейс (`analysis/__init__.py`): `ColonyDetector`, `PetriInfo`,
`ImageProcessor`, `AnalysisParams`, `AnalysisResult`.

### `analysis/params.py` — `AnalysisParams`

Датакласс с параметрами классического алгоритма. Диапазоны проверяются
`assert` в `__post_init__` — при нарушении бросается `AssertionError`.

| Поле | Тип | По умолчанию | Допустимый диапазон |
|---|---|---|---|
| `sensitivity` | `float` | `0.5` | `[0.01, 1.0]` |
| `contrast` | `float` | `1.0` | `[0.5, 3.0]` |
| `margin_percent` | `float` | `8.0` | `[0, 30]` |
| `min_colony_size` | `int` | `50` | `[1, 1000]` |
| `solid_fill` | `bool` | `False` | — |
| `fill_strength` | `int` | `15` | `[1, 100]` |

### `analysis/geometry.py` — `PetriInfo`

`frozen=True` датакласс: `cx`, `cy`, `radius`, `image_shape: tuple[int, int]`.
`__post_init__` проверяет `radius > 0`, неотрицательность центра и что центр
лежит внутри кадра. Свойства: `center -> (cx, cy)`, `area_px -> int(pi * r²)`.

### `analysis/results.py` — `AnalysisResult`

`frozen=True` датакласс из семи полей: `colony_count`, `colony_area_px`,
`colony_area_mm2`, `petri_area_px`, `petri_area_mm2`, `coverage_percent`, `px_to_mm2`.

> **Важно:** окно анализа показывает только `coverage_percent` и `colony_area_px`.
> Поля в мм² вычисляются, но в интерфейсе не отображаются
> (см. [14. Подводные камни](#14-подводные-камни)).

### `analysis/image_processor.py` — `ImageProcessor`

Набор `@staticmethod`; состояния не хранит.

| Метод | Сигнатура | Поведение |
|---|---|---|
| `preprocess_image` | `(image) -> ndarray` | `cv2.medianBlur(image, 3)` |
| `apply_clahe` | `(image, clip_limit=2.0, grid_size=8) -> ndarray` | 3-канальный вход сводится к серому; `clipLimit=clip_limit`, `tileGridSize=(grid, grid)` |
| `to_grayscale` | `(image) -> ndarray` | `BGR2GRAY`; 2D passthrough |
| `extract_green_channel` | `(image) -> ndarray` | `image[:, :, 1]` (OpenCV BGR: индекс 1 = зелёный); 2D passthrough |
| `resize_image` | `(image, max_dimension=1024) -> (ndarray, float)` | уменьшает по большей стороне с `INTER_AREA`, возвращает масштаб; если уже меньше — `(image, 1.0)` |

### `analysis/colony_detector.py` — `ColonyDetector`

Единственный класс пакета. В конструкторе создаёт `ImageProcessor`; состояние
между вызовами не кешируется.

| Метод | Сигнатура | Назначение |
|---|---|---|
| `detect_petri_dish` | `(image) -> (mask \| None, PetriInfo \| None)` | Основной поиск чашки |
| `_detect_petri_with_hough` | `(image) -> (mask \| None, PetriInfo \| None)` | Запасной вариант |
| `create_inner_mask` | `(petri_mask, petri_info, margin_percent=5) -> ndarray` | Рабочая зона внутри чашки |
| `detect_colonies` | `(image, petri_mask, params, petri_info=None, blur_size=5) -> (mask, debug_dict)` | Основной пайплайн детекции |
| `_filter_components` | `(mask, min_size) -> ndarray` | Фильтрация по площади компонент |
| `count_colonies` | `(colony_mask) -> int` | Число компонент |
| `_find_contours` | `(mask) -> List` | Обёртка над `cv2.findContours` с совместимостью OpenCV 3/4 |

#### Поиск чашки (`detect_petri_dish`)

Идея: край чашки — самый яркий объект (блик), поэтому ищем большую окружность
среди ярких пикселей.

```
1. grayscale
2. GaussianBlur(9×9, σ=2)
3. threshold(200, THRESH_BINARY)
4. если ярких пикселей < 1% кадра
      → adaptiveThreshold(101, ADAPTIVE_THRESH_GAUSSIAN_C, offset −10)
5. morphologyEx CLOSE, ядро-эллипс 30×30, 2 итерации
6. findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
7. для каждого контура:
      area ≥ 10% площади кадра
      центр не дальше min(h,w)·0.3 от центра изображения
      minEnclosingCircle → (center, radius)
      выбрать контур с максимальной площадью
8. если найдено → cv2.circle(mask, center, radius, 255, −1) + PetriInfo
9. иначе → _detect_petri_with_hough
```

`_detect_petri_with_hough`: `HoughCircles(HOUGH_GRADIENT, dp=1.2,
minDist=min(h,w)/2, param1=100, param2=30, minRadius=0.3·min_dim,
maxRadius=0.48·min_dim)`; берётся первый круг. Если кругов нет — `(None, None)`.

#### Рабочая зона (`create_inner_mask`)

```
real_margin  = max(margin_percent, 1.0)      # «пол 1%» — даже при margin=0
inner_radius = int(radius · (100 − real_margin) / 100)
```

Минимум в 1% намеренный: без него классика на полнокадровых чашках
захватывала бы блики по самому краю.

#### Детекция колоний (`detect_colonies`)

```
1. ROI = create_inner_mask(...)  либо сам petri_mask, если petri_info не задан
2. channel  = extract_green_channel(image)               # зелёный канал агара
3. clip     = 2.0 · params.contrast                       # CLAHE clipLimit
   enhanced = apply_clahe(channel, clip_limit=clip, grid_size=8)
4. k_size   = blur_size, принудительно нечётный, минимум 3
   denoised = medianBlur(enhanced, k_size)
5. bg       = GaussianBlur(denoised, 51×51, σ=0)
   diff     = addWeighted(denoised, 1.5, bg, −0.5, 0)     # вычитание фона
   masked   = bitwise_and(diff, diff, mask=roi)
6. mean, std = статистика только по пикселям ROI
   k        = 3.0 − sensitivity · 2.5                       # 0.5 при 0.5; 1.25 при 0.7
   thresh   = clamp(mean + k·std, mean + 5, 254)
   binary   = threshold(masked, int(thresh), THRESH_BINARY)
7. морфология, ядро-эллипс 3×3:
     solid_fill = False → OPEN ×1, затем CLOSE ×2
     solid_fill = True  → OPEN ×1, CLOSE(fill_strength), затем
                         drawContours(..., FILLED) — сплошная заливка
8. _filter_components(min_size=params.min_colony_size)
9. debug = {"preprocessed": masked, "binary": clean_binary}
```

Пустой ROI (нет ни одного пикселя маски) → возвращается нулевая маска и пустой
словарь debug.

`_filter_components` использует `connectedComponentsWithStats(connectivity=8)`
и оставляет компоненты с `area >= min_size` (сравнение нестрогое: ровно
`min_size` пикселей проходит).

#### Соглашение о каналах

Весь код оперирует **BGR** `np.ndarray` (порядок OpenCV). Это относится и к
`utils/image_loader.py`, и к контроллерам, и к `detect()`. Модели-потребители
переключаются на RGB сами: `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` внутри
адаптеров ONNX.

---

## 5. Пакет `utils`

### `utils/calculations.py` — `AreaCalculator`

```python
AreaCalculator(petri_diameter_mm: float = 90.0)
    .calculate_areas(petri_mask, colony_mask, petri_info=None, margin_percent=0.0)
        -> AnalysisResult
```

Логика:

```
colony_area_px = count_nonzero(colony_mask)

если petri_info задан:
    radius_full = petri_info.radius
    radius_inner = radius_full · (100 − margin)/100   (margin > 0)
    petri_area_px   = pi · radius_inner²              (иначе — полный радиус)
    petri_area_mm2  = pi · (diameter_mm/2)²
    px_to_mm2       = petri_area_mm2 / (pi · radius_full²)
иначе:
    petri_area_px  = count_nonzero(petri_mask)        # фоновая оценка
    petri_area_mm2 = 0
    px_to_mm2      = 0

colony_area_mm2 = colony_area_px · px_to_mm2
coverage_percent = colony_area_px / petri_area_px · 100
colony_count      = компоненты(colony_mask, connectivity=8) − 1
```

Физический масштаб берётся из диаметра чашки, заданного конструктором, а не из
калибровки по эталону. `petri_diameter_mm` в GUI не настраивается —
используется значение по умолчанию 90 мм.

### `utils/config.py`

```python
AppConfig(petri_diameter_mm=90.0, supported_formats=(".png",".jpg",".jpeg",
           ".bmp",".tiff",".tif",".webp"))          # frozen
AnalysisDefaults(sensitivity=50, contrast=1.0, margin_pct=8.0,
                 min_colony_size_px=50, solid_fill=False, fill_strength=15)
```

`AppConfig` — источник истины для строки «ℹ️ Поддерживаемые форматы» в
главном окне и фильтра `*.png *.jpg …` в диалоге разметки. `AnalysisDefaults`
продублирован настройками GUI; при изменении одного нужно поменять и другое.

### `utils/image_loader.py`

```python
load_image(path) -> ndarray          # BGR uint8
load_image_grayscale(path) -> ndarray
```

Порядок попыток: сначала Qt (`QPixmap` → `QImage` → буфер), затем OpenCV.
Qt-путь нужен, чтобы корректно читать некоторые форматы и EXIF-ориентацию,
которые `cv2.imread` игнорирует. Обе реализации каждой функции возвращают
`None` при неудаче; при полном провале выбрасывается `RuntimeError` с текстом,
упоминающим **оба** загрузчика (проверяется тестом
`test_image_loader.py::TestLoadImageErrorMessage`). Отдельный тест
`TestLoadImageRowPadding` сверяет результат с `cv2.imread` на PNG с
непрямым числом байт в строке — Qt дополняет строки до 4-байтовой границы,
поэтому сравнение должно идти по визуальному результату, а не по байтам.

### `utils/logging.py`

`setup_logging(level=logging.WARNING)` — базовая конфигурация `logging`.
CLI-команды поднимают уровень до `INFO`; GUI остаётся на `WARNING`.

---

## 6. Пакет `testing`

Фреймворк оценки: единый интерфейс алгоритма, реестр, загрузчики датасетов,
планировщик с батчами и кэшем, метрики, статистика, telemetry и отчёты.

### 6.1 Интерфейс и реестр

```python
class BaseDetectionAlgorithm(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def description(self) -> str: ...
    @abstractmethod
    def detect(self, image: np.ndarray, is_cropped: bool = False) -> np.ndarray: ...
```

Реестр (`testing/registry.py`) хранит два словаря:

| Словарь | Заполняется | Поведение при `get_algorithm` |
|---|---|---|
| `_CLASSES: dict[str, Type[...]]` | `@register_algorithm` по имени класса | создаёт **новый** экземпляр при каждом запросе (`cls(**kwargs)`) |
| `_INSTANCES: dict[str, ...]` | `register_algorithm_instance(name, instance)` | возвращает **тот же** объект |

```python
register_algorithm(cls) -> cls                       # декоратор
register_algorithm_instance(name, instance) -> instance
get_algorithm(name, **kwargs) -> BaseDetectionAlgorithm   # иначе ValueError
list_algorithms() -> list[str]                       # отсортированное объединение
get_algorithm_descriptions() -> list[tuple[str, str]]
_all_names() -> list[str]                            # приватный хелпер
```

Особенности:

- Приоритет у `_CLASSES`: если имя есть в обоих словарях, победит класс.
- Имена не валидируются на уникальность — повторная регистрация молча
  перезаписывает ключ. Именно поэтому повторный вызов
  `register_bundled_models()` безвреден.
- `testing/__init__.py` импортирует `classic_algorithms`, поэтому четыре
  классических алгоритма зарегистрированы при первом `import testing`.

Различия в многопоточности (реализовано в `scheduler._AlgorithmRuntime`):

- **Классы** получают по экземпляру на поток (`threading.local`): классические
  алгоритмы могут исполняться параллельно при разрешённой ресурсной политике.
- **Экземпляры** защищены именованным `threading.Lock` и вызываются
  последовательно для одного зарегистрированного экземпляра. Текущие bundled
  `NN:*` алгоритмы — экземпляры, поэтому их вызовы сериализуются; увеличение
  внешнего worker count само по себе не распараллеливает один экземпляр.
- Общая политика CPU дополнительно ограничивает одновременно активные внешние
  algorithm workers, decoder и внутренние native threads OpenCV/ONNX Runtime.

### 6.2 Алгоритмы

Подробный разбор — в [algorithms.md](algorithms.md). Здесь только состав:

| Имя в реестре | Класс | Тип |
|---|---|---|
| `ClassicDefault` | `testing.classic_algorithms.ClassicDefault` | класс |
| `ClassicHighSensitivity` | `…ClassicHighSensitivity` | класс |
| `ClassicSolidFill` | `…ClassicSolidFill` | класс |
| `ClassicLowSensitivity` | `…ClassicLowSensitivity` | класс |
| `NN:colony_mobilenet_v3_small` | `TiledOnnxModelAlgorithm` | **экземпляр**, авто |
| `NN:colony_seg` | `OnnxModelAlgorithm` | **экземпляр**, авто |
| `NN:<stem>` | `OnnxModelAlgorithm` | экземпляр для любого `models/*.onnx` |

`testing/__init__.py` не вызывает `register_bundled_models()` — только
классические алгоритмы.

### 6.3 Загрузчики датасетов

Два независимых загрузчика с разными возможностями.

#### `TestDataset` (`testing/dataset.py`) — для режима `report`

```python
TestDataset(root="test_images", sample_limit=None, load_images=True)
    .samples -> list[TestSample]
    __len__ / __getitem__ / __iter__
```

`TestSample` — `NamedTuple` из 9 полей: `name`, `source_image`,
`source_mask`, `cropped_image`, `cropped_mask`, `source_path`,
`source_mask_path`, `cropped_path`, `cropped_mask_path`.

Правила подбора:

| Что | Как ищется |
|---|---|
| исходник | любой файл в `source/`, сортировка по имени |
| маска исходника | `masks/{stem}_mask{same_ext}` — **обязательна**, иначе снимок пропускается |
| обрезок | `cropped/{stem}_cropped{same_ext}` |
| маска обрезка | `cropped_masks/{stem}_cropped_mask{same_ext}` — обе пары должны существовать одновременно |

Особенности:

- Расширение маски **обязано совпадать** с расширением изображения: JPG-исходник
  требует `masks/{stem}_mask.jpg`. Это главное ограничение формата.
- `load_images=False` — не читать массивы, только пути. Используется в режиме
  `report`, где загрузкой занимается планировщик.
- `sample_limit` ограничивает число **исходников** (считается в порядке
  сортировки); обрезки сверх лимита не читаются.
- `dataset.json` **не читается**. Этот загрузчик понимает только четыре папки.

#### `BaselineDataset` (`testing/baseline.py`) — для GUI, `baseline` и `evaluate`

```python
BaselineDataset(root, sample_limit=None)
    .structure -> "manifest" | "legacy" | "importer"
    .samples -> list[BaselineSample]
    .samples_by_variant(variant) -> list[BaselineSample]
    .count_source() / .count_cropped() / __len__ / __iter__
```

`BaselineSample` хранит `name`, `image_path`, `mask_path`, `variant` и умеет
`load_image()` / `load_mask()` (ленивое чтение). Структура определяется
`_detect_structure(root)` в порядке приоритета:

| Приоритет | Структура | Признак | Загрузчик |
|---|---|---|---|
| 1 | `manifest` | есть `dataset.json` | `_load_manifest` |
| 2 | `legacy` | есть `source/` **и** `masks/` | `_load_legacy` |
| 3 | `importer` | есть `source/`, и рядом с картинкой `{stem}_mask{same_ext}` | `_load_importer` |

Если ни одна структура не подошла — `RuntimeError("No valid image-mask pairs
found in …")`. Разница в нюансе: для `importer` маски лежат **внутри**
`source/`, для `legacy` — в отдельной папке `masks/`.

Для CLI `baseline`/`evaluate` параметр `sample_limit` применяется после загрузки
записей и выбирает `sample_limit` уникальных имён из `source`-вариантов; в результат
попадают все варианты выбранных объектов. GUI создаёт такое же ограниченное
представление после загрузки, чтобы сохранить cropped-пары для выбранных source.

#### Совместимость планировщика

`scheduler._sample_refs(dataset)` поддерживает оба загрузчика: он смотрит на
наличие атрибута `samples` и `source_path` у элементов. Поэтому один и тот же
`execute_pipeline` обслуживает `report`, `baseline`, `evaluate` и GUI-окно
тестирования; GUI и `baseline`/`evaluate` используют `BaselineDataset`, а `report`
— `TestDataset`.

### 6.4 Планировщик (`testing/scheduler.py`)

```python
execute_pipeline(dataset, algorithm_names=None, *, workers=None,
                 batch_size=8, memory_budget=None, use_cache=False,
                 telemetry=None, use_cropped=True, diagnostics=None,
                 observer=None, cancel_event=None)
    -> dict[algo][sample][variant] -> dict[metric, float]
```

Единая точка исполнения для `report`, `baseline`, `evaluate` и GUI. Загрузчик
сэмплов декодирует image/mask один раз на variant и переиспользует массивы для
выбранных алгоритмов. Ошибки чтения/декодирования привязаны к паре и не мешают
обработке остальных валидных входов.

`_execute_batches` перекрывает подготовку следующего batch с compute текущего
через `PrefetchLoader` (`testing/pipeline_overlap.py`). Очередь по умолчанию
содержит не более одного заранее подготовленного batch; producer ограничен как
ёмкостью очереди, так и байтовым бюджетом активных + prefetched входов. При
заполнении ресурсов чтение/декодирование ожидает backpressure; ошибка или
отмена приводит к закрытию producer и очистке очереди.

`memory_budget` — целевой бюджет декодированных inputs, а не жёсткая граница
RSS процесса: резервируется 25% на временные буферы алгоритмов/runtime. Если
одна пара превышает input budget, она обрабатывается отдельно и не отбрасывается;
диагностика отмечает превышение. Память алгоритмических временных буферов и ОС
не входит в точное ограничение.

### CPU resource policy

`testing/resource_policy.py::resolve_resource_policy` распределяет видимую CPU
ёмкость между algorithm workers, decode и нативными потоками OpenCV/ONNX. Видимая
ёмкость берётся консервативно по минимуму известных process affinity, cgroup
CPU quota и физического CPU count хоста; если сигналы недоступны, применяется
консервативный fallback. Интерактивный режим резервирует CPU ёмкость под GUI.

`--workers N` задаёт верхнюю границу, но окончательный worker count ограничивается
ёмкостью процесса, готовым числом задач и памятью конкурентных inputs. При
давлении очереди/ресурсов scheduler может уменьшить число workers. Нативные пулы
ограничиваются совместно с внешним concurrency, чтобы не умножать без контроля
threads × workers.

```text
resolve_resource_policy → apply_native_limits
  → iter_batches → PrefetchLoader(decode producer, bounded queue)
  → ThreadPoolExecutor(algorithm tasks)
  → metrics / observer phases / telemetry
```

Число workers — настройка параллелизма, а не гарантия ускорения. Бенчмарк
принимается по end-to-end wall time/throughput, корректности результата и
соблюдению memory target. Зафиксированный warm-cache acceptance workload дал
одинаковые хэши результатов, но auto mode был примерно в 3.16 раза медленнее
последовательного; cold-storage physical I/O не измерялся. Это измерение
конкретного workload, а не универсальная характеристика.

`_run_task` измеряет время ожидания от момента отправки конкретной задачи,
переиспользование `SampleContext`/prediction cache, detect и metrics. Ошибки,
progress и фазы передаются через `PipelineObserver`; отмена останавливает
постановку новых работ на безопасной границе, уже начатые native-вызовы
завершаются кооперативно.

Ключ prediction cache включает алгоритм, параметры, variant (`is_cropped`) и
содержимое входа, поэтому исходник и обрезок одного объекта не путаются.

### 6.5 Кэш (`testing/cache.py`)

Два независимых механизма.

**Сигнатура датасета** — `evaluate/<run>/` и `run_evaluate`:

```python
compute_dataset_signature(data_root) -> sha256 hex
load_cache(data_root)  -> dict | None
save_cache(data_root, cache_data)
clear_cache(data_root)
```

Подпись считается по списку `"{rel_path}:{size_in_bytes}"` для файлов в
`source/`, `masks/`, `cropped/`, `cropped_masks/` плюс сам `dataset.json`.
Файл кэша: `{data_root}/.cache/{signature}.json`. Смена любого файла (по
размеру) даёт новую подпись → новое имя файла, старый кэш просто не найдётся.

**Кэш предсказаний** — `PredictionCache`, работает при `--cache`:

```python
PredictionCache(data_root)
    .key / .get / .put                 # по пути к файлу
    .get_array / .put_array            # по уже загруженному массиву
    .clear(algo_name=None)
```

- Хранилище: `{data_root}/.cache/preds/{algo_name}/{sha256}.png`.
- Хэш: `algo_name` + `params` (`json.dumps(sort_keys=True)`) + байты изображения
  (для `_hash` — содержимое файла, для `_array_hash` — `dtype`, `shape` и
  `tobytes()`).
- `put*` не перезаписывает существующий файл.
- `get*` возвращает `None`, если файл не читается; «битый» PNG не валит прогон.

`--clear-cache` удаляет **только** файл сигнатуры, а не весь каталог
`.cache/`. Полная очистка предсказаний — `PredictionCache.clear()`.

### 6.6 Метрики (`testing/metrics.py`)

```python
compute_segmentation_metrics(pred_mask, gt_mask, smooth=1e-6) -> dict
```

Считает `tp/fp/fn/tn` попиксельно после бинаризации обоих массивов
(`> 0`), затем 6 метрик:

```
iou       = tp / (tp + fp + fn + ε)
dice      = 2·tp / (2·tp + fp + fn + ε)
precision = tp / (tp + fp + ε)
recall    = tp / (tp + fn + ε)
f1        = 2·P·R / (P + R + ε)
accuracy  = (tp + tn) / (tp + fp + fn + tn + ε)
```

Возвращаются также сырые `tp/fp/fn/tn` — они исключаются из агрегации
`_mean_metrics`, но доступны в per-snapshot таблицах.

### 6.7 Агрегация и статистика

`testing/runner.py`:

| Функция | Результат |
|---|---|
| `_mean_metrics(list) -> dict` | `mean_*` + `std_*` (агрегация с делением на `n`, не `n−1`) |
| `compute_detailed_metrics(list) -> dict` | `mean/std` + `median/q1/q3/p5/p95/min/max` |
| `run_algorithm(algo, dataset)` | последовательный прогон одного алгоритма (совместимость) |
| `run_all(dataset, algorithms=None, workers=None, use_cache=False, batch_size=8, memory_budget=None, telemetry=None)` | обёртка над `execute_pipeline` |
| `compare_algorithms(dataset, name_a, name_b)` | `{metric: {sample: {variant: "a"/"b"/"tie"}}}`, всегда `workers=1` |
| `compute_summary(all_results)` | список записей с `name`, `description`, `num_images` + все агрегаты |
| `compute_wilcoxon_table(all_results)` | `{metric: {"A vs B": {p_value, interpretation}}}` |
| `compute_outlier_table(all_results)` | список `{algorithm, metric, sample, variant, value, z_score}` при \|z\| > 3 |
| `compute_all_winners(all_results)` | `{metric: {"A vs B": {a_wins, b_wins, ties}}}` |

`testing/statistics.py`:

| Функция | Что делает |
|---|---|
| `compute_descriptive(values)` | mean, std, median, q1, q3, p5, p95, min, max |
| `detect_outliers(values, z_thresh=3.0)` | точки с \|z\| > порога; при `std == 0` — пустой список |
| `wilcoxon_signed_rank(a, b)` | ранговый критерий Уилкоксона, нормальная аппроксимация с поправкой на ties |
| `compute_winner_fractions(metrics_a, metrics_b, metric_key)` | доли побед; ничья при \|Δ\| < `1e-9` |

`wilcoxon_signed_rank` возвращает `{"p_value": None, "message": "identical"}`,
когда все разности нулевые; при несовпадении длин — `ValueError`.

### 6.8 Telemetry (`testing/telemetry.py`)

```python
TelemetryCollector(enabled=False, output_path=None, interval=5.0,
                   debug_tasks=False)
    .start() / .event(type, **payload) / .maybe_snapshot(**payload)
    .record_stage(stage, duration)      # file_read, image_decode, input_wait,
                                        # queue_wait, cache, detect, metrics, report
    .record_logical_io(stage, bytes, decoded_bytes=...)
    .record_cpu_availability(logical_count, physical_count, affinity, quota)
    .record_worker_limits(outer_workers, native_threads, source)
    .record_input_starvation(starvation_seconds, active_seconds, ratio)
    .record_task(...) / .record_work(...) / .record_cache(...) / .record_error(...)
    .summary() / .finish(**payload)
```

Сводка включает доступность CPU/affinity/quota, внешние и нативные worker limits,
logical read/decode bytes, process CPU time и cores consumed, RSS/peak RSS,
доступность системных I/O counters, latency стадий и input-starvation. Ноль
физических read bytes не означает, что логического чтения не было: данные могли
поступать из OS page cache. Неизвестные системные счётчики остаются `null`, а не
подменяются нулём.

Артефакты (только при `enabled=True`):

| Файл | Содержимое |
|---|---|
| `{output_path}` | Итоговый JSON-свод: стадии с перцентилями, счётчики, ошибки |
| `{output_path с суффиксом .jsonl}` | Поток событий (append), включая периодические снапшоты |

Путь по умолчанию: `Path(output).with_suffix(".performance.json")` в режимах
`report`/`baseline` и `evaluations/<run_id>/performance.json` в `evaluate`.

### 6.9 Отчёты

#### `testing/dashboard.py` — HTML для `report` и GUI

```python
generate_report(all_results, output_path="test_report.html",
                include_per_snapshot=True, comparison=None,
                include_stats=True, include_boxplots=True, include_scatter=True,
                include_significance=True, include_outliers=True,
                stats_output=None)
```

Собирает страницу из блоков, каждый за которые отвечает своя функция
`_build_*`: сводная таблица, секция на каждый алгоритм, данные для графиков
(bar/box/scatter), секция статзначимости, таблица долей побед, таблица
выбросов, секции парного сравнения. Графики — Chart.js 4.4.7 с CDN
(`cdn.jsdelivr.net`), то есть для просмотра нужен интернет. Тёмная тема
приложения повторяется в inline-CSS.

`stats_output` дополнительно выгружает полную статистику в JSON
(`_export_stats_json`).

#### `testing/evaluator.py` — режим `evaluate`

```python
DEFAULT_CONFIG = {data_root, use_cropped, algorithms, models, compare,
                  per_snapshot, use_cache, workers, batch_size, memory_budget,
                  sample_limit, telemetry, telemetry_interval, performance_output}

load_config(config_path, cli_args) -> dict
create_run_dir(base_dir="evaluations") -> str
get_git_info() -> {commit_hash, commit_message}
run_evaluate(config) -> str          # возвращает run_id
_generate_eval_html(result, run_id) -> str
```

**Приоритет конфигурации** (`load_config`, `evaluator.py:46`):

```
DEFAULT_CONFIG
  ← значения из YAML/JSON-файла (--config), неизвестные ключи ИГНОРИРУЮТСЯ
  ← значения из CLI, но только если не None
```

Тонкость: argparse отдаёт дефолты (например `data_root="test_images"`,
`batch_size=8`), и они **перезаписывают** файл конфигурации, потому что
проверяется только `value is None`. Чтобы файл конфигурации действительно
влиял на эти параметры, флаг должен быть явно `default=None`.

**Структура каталога прогона:**

```
evaluations/2026-09-15_19-44-10/
├── report.json      полная выгрузка метрик (export_json)
├── report.html      самодостаточный отчёт с Chart.js (evaluate-версия)
├── run_info.json    run_id, timestamp, путь и размер датасета, алгоритмы, git
├── config.yaml      снимок конфигурации (config.json, если нет PyYAML)
└── performance.json (+ .jsonl)   телеметрия, если --telemetry
```

`create_run_dir` использует `mkdir(exist_ok=False)` — два прогона в одну секунду
упадут с `FileExistsError`, а не перезапишут каталог.

`--import-root` в `evaluate` вызывает `CocoBboxImporter().build()` перед
оценкой: сырой COCO-датасет можно импортировать и тут же измерить.

### 6.10 CLI (`testing/__main__.py`)

Один парсер на три режима; полный список флагов — в
[user-guide.md](user-guide.md#cli-html-отчёт). Ключевые правила:

| Правило | Последствие |
|---|---|
| `--algorithms` = `None` → все зарегистрированные | Без `register_bundled_models()` это только 4 классических |
| `--model` регистрирует `OnnxModelAlgorithm` | Всегда обычный (resize) адаптер, даже для MobileNet-файла |
| `--sample-limit` отключает загрузку агрегатного кэша в `evaluate` | Прогон на подмножестве не портит кэш полного датасета |
| `--no-cropped` отключает вариант `cropped` | Ускоряет прогон вдвое на датасетах с обрезками |
| `--output` в `baseline`/`report` — **путь к JSON/HTML**, а не каталог | В `evaluate` `--output` не используется, отчёт всегда в run-dir |

---

## 7. Пакет `ui`

### `ui/styles.py`

Три функции возвращают строки QSS: `get_application_style()` — вся тема
Catppuccin Mocha, `get_image_frame_style()` — рамка области просмотра,
`get_result_panel_style()` — панель результатов. Стиль применяется один раз
в `main.py` на уровне `QApplication`. Конкретные виджеты дополнительно
переопределяют стили инлайн (например, кнопки панели инструментов разметки).

### `ui/main_window.py` — `MainWindow`

Три кнопки-входа: «🔍 Анализировать», «✏️ Разметка тестовых изображений»,
«🧪 Тестирование алгоритмов». Путь к файлу вводится вручную или через
«📂 Открыть»; на изменение текста вызывается `_validate_file`, которая
включает/выключает кнопку анализа и формирует подсказку о форматах из
`AppConfig.supported_formats`. Ошибки показываются через `_show_error`
(`QMessageBox`).

### `ui/analysis_window.py` — `AnalysisWindow`

Состав элементов:

| Группа | Элементы и значения по умолчанию |
|---|---|
| Изображение | `ImageLabel` с масштабированием под размер виджета, подпись-режим, `QProgressBar`, `status_label` |
| 👁️ Режим просмотра | `QComboBox`: «🔍 Результат (С наложением)», «📷 Оригинал», «🌗 Предобработка (Контраст)», «🏁 Бинарная маска (Ч/Б)»; чекбоксы «Показать контур чашки» и «Закрасить колонии» (оба включены); `algorithm_combo` |
| 📏 Геометрия чашки | `spin_x`/`spin_y` (`0..5000`, суффикс ` px`), `spin_radius` (`10..3000`), кнопка «Сбросить к авто-поиску» |
| ⚙️ Параметры алгоритма | `slider_sens` `1..100` (50), `slider_contrast` `5..30` (10 → 1.0×), `spin_margin` `0..30` (8.0), `spin_min_size` `1..1000` (50), `chk_solid_fill`, `spin_fill_strength` `1..100` (15), кнопка «🔄 Пересчитать» |
| 📊 Результаты | Текстовое поле (макс. 150 px высоты), «💾 Сохранить», «Закрыть» |

`algorithm_combo` наполняется в `_load_analysis_algorithms`: первым пунктом
идёт псевдоалгоритм `"Классический (параметры ниже)"` с данными `"__classic__"`,
затем все зарегистрированные имена. При выборе NN-алгоритма
`_on_algorithm_changed` **отключает** чувствительность, контраст, мин. размер,
solid fill и силу заливки — они на него не влияют.

Окно создаёт `BackgroundOperation` (`ui/background.py`) для загрузки/декодирования,
начального поиска чашки, классического анализа и выбранных алгоритмов. Worker
получает snapshot массивов, сообщает статус/прогресс queued-сигналами и проверяет
кооперативную отмену между безопасными этапами (например, между NN-тайлами).
Результаты помечены generation; результат предыдущего изображения не применяется
после смены выбора. Qt widgets и `QPixmap` остаются в GUI-потоке.

Переключение представления и resize окна планируют актуальную задачу композитинга
видимой области; устаревшая generation отбрасывается. Полноразмерные маски и
алгоритмические данные сохраняются, а не заменяются уменьшенным pixmap.

Сохранение запускает тяжёлую сериализацию/запись в фоне по snapshot выбранного
режима и overlay. По совместимому контракту сохраняется содержимое и размер
отображаемого pixmap (поэтому сильный zoom может дать файл меньше исходника),
формат кодирования выбирается расширением файла.

При закрытии окна активная операция получает запрос отмены; GUI не делает
синхронный `wait()`, продолжает обрабатывать события в состоянии «Завершение…»
и закрывается после безопасного возврата worker.

### `ui/controllers/analysis_controller.py` — `AnalysisController`

```python
AnalysisController(detector=None, calculator=None)
    .find_petri_dish(image) -> (mask | None, PetriInfo | None)
    .analyze(image, petri_mask, params, petri_info=None, blur_size=5) -> AnalysisResult
    .calculate_algorithm_result(image, petri_mask, algorithm, petri_info=None,
                                margin_percent=0.0, is_cropped=False,
                                progress_callback=None) -> (AnalysisResult, mask)
    .analyze_with_algorithm(...) -> AnalysisResult       # + публикует маску
    .set_algorithm_mask(mask)
    .colony_mask -> ndarray | None
    .debug_images -> dict
```

`calculate_algorithm_result` — чистый метод, пригодный для фонового потока:

1. Если у алгоритма есть `detect_with_progress` **и** передан колбэк — вызывается
   он (тайловый прогресс), иначе обычный `detect`.
2. Проверяется, что маска алгоритма и маска чашки совпадают по форме с кадром.
3. Маска бинаризуется и **обрезается по рабочей зоне**: чашка, при наличии
   `petri_info`, дополнительно сужается внутренним кругом
   `max(margin_percent, 1.0)` — тот же минимум 1%, что и у классики.
4. Считаются площади и покрытие.

Маска публикуется в состоянии контроллера только явно — через
`set_algorithm_mask` или `analyze_with_algorithm`. Это гарантирует, что
частичный результат фонового потока не появится в GUI (покрыто тестом
`test_calculating_algorithm_result_does_not_publish_partial_mask`).

### `ui/testing_window.py` — `TestingWindow`

| Блок | Содержимое |
|---|---|
| 📁 Датасет | Кнопка «📂 Выбрать папку…», поле пути, подсказка про `source/` + `masks/`, предупреждение об отсутствии папок |
| 🧠 Алгоритмы | `QCheckBox` на каждый зарегистрированный, кнопка «⬇️ Загрузить модель (.onnx)» |
| ⚙️ Параметры | «Детализация по снимкам», `batch_size` (`1..512`, 8), `sample_limit` (`0..1000000`, 0 = без лимита), «Telemetry» |
| Парное сравнение | Два `QComboBox` со списком алгоритмов |
| 🚀 Запуск | «▶️ Запустить тест», «💾 Экспорт отчёта» |
| 📊 Результаты | Таблица метрик + таблица парного сравнения |

Прогон алгоритмов, report export и подготовка больших таблиц выполняются через
`BackgroundOperation` вне GUI-потока. Окно показывает фазу (scan/read-decode,
algorithm, comparison, result preparation), состояние и прогресс; большие
таблицы добавляются порциями. Отмена прекращает постановку новой работы на
безопасной границе; закрытие во время неотменяемого native-вызова не блокирует
GUI и завершается после безопасного возврата worker. Поле «Объектов» = 0
означает «без ограничения» и передаётся как `sample_limit=None`.

Важно: окно тестирования **не** вызывает `register_bundled_models()`, поэтому
в списке по умолчанию только классические алгоритмы; NN-модель добавляется
кнопкой «Загрузить модель».

### `ui/labeling_session_dialog.py` — `LabelingSessionDialog`

Диалог создания/открытия сессии: поле названия, текущий корень хранилища с
кнопкой «✏️ Изменить», список «Недавние сессии» (двойной клик или кнопка
открытия), «📂 Открыть существующую папку…». Результат — путь к папке сессии,
которую создаёт `SessionManager.session_path()`.

---

## 8. Пакет `labeling`

### `labeling/session_manager.py`

```python
sanitize_name(name) -> str | None
ensure_session_structure(session_dir) -> Path
default_config_path() -> Path
SessionManager(config_path=None)
    .current_root -> Path
    .recent_sessions -> list[dict]
    .set_root(path) / .session_path(name) / .add(session_dir, name) / .remove(path)
```

| Элемент | Поведение |
|---|---|
| Корень по умолчанию | `Path.home() / "BacteriaLabeling"` |
| Конфиг | JSON с `root` и списком недавних сессий; битый JSON игнорируется (тест `test_corrupt_json_ignored`) |
| `sanitize_name` | Вырезает недопустимые символы Windows (`<>:"/\|?*`), обрезает до предельной длины; возвращает `None` для пустого результата |
| `session_path` | Бросает исключение на пустом имени |
| Список недавних | Дедупликация с переносом в начало, ограничение по длине |
| Структура сессии | `source/`, `masks/`, `cropped/`, `cropped_masks/` — создаётся идемпотентно |

### `labeling/labeling_window.py`

`PaintLabel(QLabel)` — виджет полноразмерной маски с масштабированием и локальным рендерингом:

| Метод | Поведение |
|---|---|
| `set_image(image, mask=None)` | Установка кадра и маски |
| `set_petri_info(info)` | Обновление геометрии чашки |
| `mask` | Текущая маска (свойство) |
| `clear_mask()` | Полная очистка |
| `zoom_in/out/reset`, `zoom_percent` | Диапазон 0.1× – 20× |
| `wheelEvent` | Зум колесом мыши |
| `mousePress/Move/Release` | Рисование с поправкой координат на масштаб (`_widget_to_image`) |
| `_paint_at(pos)` | Меняет локальную область исходной маски, патчит кеш уровней масштаба и инвалидирует dirty rect |
| `_composite_region()` | Создаёт композит только запрошенной областью full-resolution кадра |
| `_build_level()` / `_get_level()` | Кэширует уменьшенные уровни для текущего режима просмотра |
| `paintEvent()` | Отображает только видимые области/участки, а не пересобирает полный overlay при каждом движении кисти |

`LabelingWindow(QMainWindow)` — левая панель (режимы source/cropped, текущий
путь, список файлов, «🔄 Обновить», «📂 Добавить изображения», геометрия чашки,
авто-поиск и обрезка), центральная область просмотра и кистевые инструменты.
Чтение списка, decode исходника/маски, поиск чашки, обрезка, массовое копирование,
save mask и ZIP export выполняются в `BackgroundOperation`; списки и результаты
применяются только для актуальных generation. Qt widgets изменяются только в GUI
потоке. Устаревшая работа отменяется кооперативно; закрытие ждёт возврата активных
worker через deferred close, не блокируя обработку событий.

Правила именования масок при сохранении:

| Режим | Куда | Имя |
|---|---|---|
| Исходники | `masks/` | `{stem}_mask.png` |
| Обрезки | `cropped_masks/` | `{stem}_cropped_mask.png` |

Маски всегда сохраняются в **PNG** независимо от формата исходника. Из-за
этого пара «JPG-исходник + PNG-маска» не будет найдена загрузчиком
`TestDataset` (он ищет одинаковое расширение) — см.
[14. Подводные камни](#14-подводные-камни).

Размер кисти меняется шагом 2 px в диапазоне 2–100 px.

### `ui/controllers/labeling_controller.py` — `LabelingController`

```python
.load_image_rgb(path) -> ndarray                     # RGB для Qt-отображения
.load_mask(path, expected_shape) -> ndarray | None
.detect_petri(image_bgr) -> PetriInfo | None
.crop_by_petri(image_bgr, petri_info) -> ndarray     # чёрный фон вне круга
.save_mask(mask, path)
.export_session_to_zip(session_dir, output_path)
.get_current_dir(session_dir, mode) -> Path          # создаёт папку при отсутствии
.get_mask_dir(session_dir, mode) -> Path
.list_image_files(directory) -> list[Path]           # фильтрует по SUPPORTED_EXTS
```

### Хранение настроек

`QSettings` в проекте **не используется**. Единственное постоянное хранилище —
JSON-конфиг `SessionManager` (`root` + недавние сессии). Выбор датасета,
алгоритмов и параметров в окнах тестирования и анализа между запусками
не сохраняется.

---

## 9. Пакет `train`

### 9.1 Контракт данных

`train/dataset_manifest.py` — единственный источник правды о том, что такое
обучающая выборка:

```python
class MaskMode(StrEnum):    BINARY = "binary"; MULTICLASS = "multiclass"
class StorageMode(StrEnum): REFERENCE = "reference"; COPY = "copy"
class SampleKind(StrEnum):  SOURCE = "source"; CROPPED = "cropped"; AUGMENTED = "augmented"
class Origin(StrEnum):      LABELING; LEGACY; EXTERNAL

@dataclass DishGeometry:    cx: int; cy: int; r: int
@dataclass SampleRecord:    id; kind; image: Path; mask: Path;
                            subset: str | None; dish: DishGeometry | None
@dataclass DatasetManifest: name; origin; mask_mode; storage; samples: list[SampleRecord]
    .validate(data_root, allow_multiclass=False)
validate_mask_mode(mask_mode, allow_multiclass=False)
```

Пути в `SampleRecord` **относительные** от корня датасета. `validate()`
проверяет допустимость значений `origin`/`storage`/`kind`/`subset`, режим маски
и существование обоих файлов каждой пары. `mask_mode="multiclass"` — задел:
без `allow_multiclass=True` бросается понятная ошибка.

### 9.2 Адаптеры (`train/dataset_adapters.py`)

```python
class BaseAdapter(ABC):
    @abstractmethod
    def build(self, data_root: Path) -> DatasetManifest: ...

class ManifestAdapter(BaseAdapter)    # читает готовый dataset.json
class LabelingAdapter(BaseAdapter)    # source/ + masks/ + cropped/ + cropped_masks/
class PairsAdapter(BaseAdapter)       # легаси images/ + masks/
class CocoBboxImporter(BaseAdapter)   # растризует COCO-боксы в маски

load_manifest(data_root, allow_multiclass=False) -> DatasetManifest
```

`load_manifest` выбирает адаптер по структуре: `dataset.json` → `source/`
(Labeling) → `images/` (Pairs) и вызывает `manifest.validate(data_root)`.

`LabelingAdapter` и `PairsAdapter` строят манифест **в памяти**
(`storage="reference"`) — файлы не копируются. `CocoBboxImporter`, наоборот,
материализует самодостаточную папку (`storage="copy"`).

`CocoBboxImporter.build(data_root, output_dir=None, crop=False, split=False,
seed=42, split_ratios=DEFAULT_SPLIT_RATIOS)`:

1. Читает `annot_COCO.json`, группирует боксы по `image_id`.
2. Растризует каждый бокс в **вписанный эллипс** (`_rasterize_ellipse`).
3. Пропускает снимки без разметки.
4. При `crop=True` ищет чашку (`_find_dish`) и пишет обрезки с чёрным фоном.
5. Копирует исходники и маски (`_write_source` / `_write_cropped`).
6. Пишет `dataset.json` (`_write_dataset_json`).
7. При `split=True` выполняет стратифицированный групповой сплит и сохраняет
   `split.json` (`_write_split_json`); пары `source` и `cropped` одного объекта
   всегда попадают в один подмножество.

Подробный разбор — в
[dataset-training-pipeline.md](dataset-training-pipeline.md).

### 9.3 Сплит (`train/dataset_split.py`)

```python
stratified_group_split(group_labels, *, seed=42, ratios=DEFAULT_SPLIT_RATIOS)
    -> dict[group_id, "train" | "val" | "test"]
```

Распределение стратифицировано по размеру группы (сколько снимков), чтобы
крупные и мелкие объекты попали в пропорции во все подмножества. Сплит
детерминирован по `seed` и **не зависит от порядка входных данных**
(проверяется тестом). `DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)`.
Некорректные доли (сумма ≠ 1, отрицательные) и неизвестные/дублирующиеся
идентификаторы приводят к `ValueError`.

### 9.4 Датасеты PyTorch (`train/dataset.py`)

```python
ColonyDataset(records, data_root, img_size=512, augment=False)
    __getitem__ -> (image[3,H,W], mask[1,H,W])

ColonyPatchDataset(records, data_root, patch_size=512, patches_per_image=4,
                   seed=42, augment=False, training=True)
    .set_epoch(epoch) / .epoch
    __getitem__ -> (image[3,H,W], mask[1,H,W])

GroupedPatchBatchSampler(dataset, batch_size, seed)
    .set_epoch(epoch); __iter__ -> list[list[int]]; __len__

_subsets_from_manifest(manifest, val_split, seed) -> (train_records, val_records)
make_datasets(data_root, img_size=512, val_split=0.2, seed=42, augment=True)
    -> (train_ds, val_ds)
make_test_dataset(data_root, img_size=512, *, records=None) -> ColonyDataset
```

| Что | Поведение |
|---|---|
| Разрешение | `img_size` — сторона входа; маска ресайзится тем же интерполятором |
| Нормализация | ImageNet: `(x/255 − mean) / std` |
| Аугментация train | flip / поворот (`albumentations`) |
| Подмножества | Берутся из `record.subset`, если задан; иначе seed-сплит `val_split` |
| Явный `subset="test"` | Записи **никогда** не попадают в train/val (тест `test_explicit_test_records_are_not_folded_into_train_or_validation`) |
| Датасет без train и val | Бросается исключение |
| `ColonyPatchDataset` | `patches_per_image` тайлов на снимок; координаты детерминированы `seed`+`epoch` |
| Первый тайл валидации | Всегда привязан к переднему плану |
| Валидационные тайлы | Не зависят от эпохи — координаты стабильны |
| `GroupedPatchBatchSampler` | Гарантирует локальность по снимку: тайлы одного изображения в одном батче |

Одновременное обучение `source` и `cropped` записей — нештатный режим:
используйте `train-colony` с `--split`, который сам разносит варианты.

### 9.5 Модели (`train/models/`)

```python
register_model(name)                 # декоратор
get_model(name, **kwargs)            # создаёт экземпляр
list_models() -> list[str]
```

```python
class BaseSegmenter(ABC, nn.Module):
    forward(x) -> Tensor
    predict(x) -> Tensor                       # sigmoid(logits)
    get_loss(pred, target) -> Tensor            # BCE(with_logits) + Dice
    compute_metrics(pred, target, threshold=0.5) -> dict
    to_onnx(path, input_shape=(1, 3, 512, 512))
    load_onnx(path)                            # staticmethod
    preprocess(image, img_size=512) -> ndarray
```

Ключевые детали `to_onnx`: `input_names=["input"]`, `output_names=["output"]`,
динамическая ось только по батчу, `opset_version=18`, `external_data=False`
(один файл без внешних `.data`).

`get_loss` = `BCEWithLogits` + `1 − Dice` (smooth `1e-6`), усреднение Dice по
батчу. `compute_metrics` возвращает `loss`, `iou`, `dice`, `precision`, `recall`.

| Модель | Энкодер | Примечание |
|---|---|---|
| `unet` | 4 уровня, `features=(64,128,256,512)` | Классический U-Net (`DoubleConv`/`Down`/`Up`) |
| `unet_small` | 4 уровня, `features=(32,64,128,256)` | Облегчённый U-Net |
| `mobilenet_v3_small_unet` | MobileNetV3-Small ( torchvision, `pretrained=True`) + `_FuseBlock`-декодер, `decoder_channels=(32,24,16,12)` | Компактная модель для CPU-инференса тайлами |

Выход — **логиты** размера `(N, 1, H, W)`. Сигмоида применяется на стороне
потребителя (`predict()`, либо `output_is_logits=True` в адаптере).

`list-models` печатает доступные имена; `get_model(name, pretrained=...)`
передаёт `pretrained` в конструктор (в тестах и CI — `pretrained=False`, чтобы
не скачивать веса).

### 9.6 Цикл обучения (`train/train.py`)

```python
run_training(cfg, progress_callback=None) -> (summary, train_hist, val_hist, run_dir)
train_epoch(model, loader, optimizer, device, scaler=None) -> dict
val_epoch(model, loader, device) -> dict            # @torch.no_grad()
```

```python
@dataclass TrainingConfig:
    data_root=Path("train/data"); img_size=512; batch_size=8; epochs=200
    lr=1e-3; weight_decay=1e-5; val_split=0.2; num_workers=4; seed=42
    patience=30; augment=True; model_name="unet"; architectures=("unet",)
    run_dir=Path("train/runs"); device="cuda"; dashboard=False
    dashboard_port=8765; patch_training=False; patches_per_image=4
    pretrained=True; resume_checkpoint=None
```

| Аспект | Поведение |
|---|---|
| Устройство | CUDA, если доступна, иначе CPU |
| Оптимизатор | AdamW (`lr`, `weight_decay`) |
| Early stopping | `patience` эпох без улучшения IoU на валидации |
| Лучший чекпойнт | По IoU валидации |
| AMP | `GradScaler`, когда CUDA доступна |
| TensorBoard | `SummaryWriter` в `{run_dir}/tensorboard/` |
| Артефакты | `{run_dir}/checkpoints/{best,last}.{pt,onnx}`, `summary.json`, `report.html` |
| Provenance | `_initialization_provenance(...)` пишет, был ли warm-start, хэш чекпойнта и источник инициализации |

Структура каталога прогона:

```
train/runs/{model}_{timestamp}/
├── checkpoints/
│   ├── best.pt, best.onnx
│   └── last.pt, last.onnx
├── summary.json
├── report.html               # Plotly, train/reporter.py
└── tensorboard/
```

### 9.7 Оценка и отчёты (`train/evaluation.py`)

| Функция | Назначение |
|---|---|
| `compute_binary_metrics(prediction, target)` | IoU/Dice/Precision/Recall/Accuracy из массивов |
| `predict_full_image_tiled(model, image_bgr, *, tile_size=512, stride=384, progress_callback=None)` | Полноразмерная тайловая сегментация |
| `evaluate_model_on_records(model, records, data_root, *, tile_size, stride, limit)` | Метрики по записям |
| `evaluate_test_subset(model, data_root, ...)` | То же по подмножеству `test` |
| `evaluate_patch_records(model, records, data_root, *, patch_size, patches_per_image, batch_size, seed)` | Метрики по фиксированным тайлам |
| `verify_onnx_cpu_parity(model, onnx_path, records, data_root, *, patch_size, patches_per_image, max_patches, seed, tolerance=1e-4, logit_tolerance=2e-4)` | Совпадение PyTorch и ONNX |
| `compute_binary_metrics_from_counts(counts)` | Метрики из агрегированных счётчиков |
| `class_counts(data_root)` | Распределение классов по датасету |
| `git_revision_info()` | Хэш и версия git |
| `build_training_report(...) -> dict` | Итоговый отчёт (пишется в `report.json`) |

Parity-критерий: вероятности должны совпасть с точностью `tolerance`, а логиты
допускают расхождение до `logit_tolerance` — так проверка не ломается из-за
разного порядка вычислений в CPU-операциях.

`train/reporter.py::generate_report(summary, train_hist, val_hist, output_path)`
собирает Plotly-графики по эпохам (отдельный отчёт от `evaluation.build_training_report`,
который возвращает словарь для машинного чтения).

### 9.8 Пайплайн `train-colony` (`train/training_pipeline.py`)

```python
run_colony_training(*, data_root, output_root, seed=42,
                    ratios=(0.7, 0.15, 0.15), epochs=100, batch_size=8,
                    patches_per_image=4, img_size=512, pretrained=True,
                    model_promotion_path=None, allow_overwrite=False,
                    cpu_benchmark=False, resume_run=None) -> dict
```

Шаги:

```
1. Проверка output_root: непустой каталог → отказ, если не resume_run
2. dataset_root = output_root/"dataset"
3. Импорт: CocoBboxImporter.build(..., split=True) с проверкой подписи
   исходного дерева (_tree_signature) ДО и ПОСЛЕ импорта
4. Тренировка в output_root/"runs" с patch-обучением
5. Повторная проверка подписи источника: обучение не должно менять данные
6. Оценка на test-подмножестве (evaluate_test_subset)
7. ONNX parity (verify_onnx_cpu_parity)
8. Опциональный CPU-бенчмарк полноразмерного тайлового инференса
9. build_training_report → report.json
10. Промоция best.onnx в model_promotion_path
    (требует allow_overwrite, если файл уже существует)
```

Ключевые гарантии:

- **Источник только для чтения.** Три проверки `_tree_signature` гарантируют,
  что ни импорт, ни обучение не модифицируют исходный датасет. Это покрыто
  тестами `test_training_pipeline.py`.
- **Отказ вместо перезаписи.** Непустой `output_root` приводит к исключению,
  а не к слиянию с прежними артефактами.
- **Явное согласие на перезапись модели.** `model_promotion_path` не
  перетирается без `allow_overwrite`.
- **Resume.** `resume_run` указывает на предыдущий каталог прогона: подготовленный
  датасет и чекпойнт переиспользуются, импорт повторяется не будет.

### 9.9 Прочее в `train/`

| Модуль | Назначение |
|---|---|
| `main.py` | CLI-диспетчер: `train`, `train-compare`, `list-models`, `dataset-info`, `train-colony` |
| `augment.py` | Оффлайн-генерация 15 вариантов каждой пары (`albumentations`) |
| `compare.py` | `run_train_compare`: обучает каждую архитектуру, оценивает `best.onnx` на тестовых парах, пишет `compare.json` + `compare.html` |
| `predict.py` | `predict_single` / `batch_predict` — ONNX-инференс. **Не подключён** ни к одной консольной команде; используется как библиотека |
| `export_single.py` | `best.pt` → single-file ONNX (без внешних данных) |
| `importer_cli.py` | CLI `import-22022540`: импорт + `--verify` |
| `dashboard/server.py` | FastAPI: `GET /`, `GET /history`, `WS /ws`; `run_server(port=8765)` |

Дашборд открывает `train/dashboard/templates/index.html` (Chart.js) и шлёт
состояние обучения через WebSocket. Живёт только пока идёт обучение.

---

## 10. Сборка: пакет `scripts`

| Файл | Роль |
|---|---|
| `build_nuitka.sh` | Пользовательский скрипт: создаёт изолированное `.venv-build`, ставит `nuitka` + `zstandard`, запускает `nuitka_build.py` |
| `nuitka_build.py` | Программный запуск Nuitka; `compute_jobs()` — число параллельных задач из числа ядер с потолком |
| `nuitka_flags.py` | `build_flags(root) -> list[str]` — генерация `--include-package` для `analysis`, `ui`, `utils`, `labeling`, `testing`; исключения `.venv*`, `train`, `test_images`; отдельные флаги для `onnxruntime` и его `.dll`/`.so`; `--include-data-files=…/models/*.onnx=models/`; `--enable-plugin=pyqt6` |

`nuitka_flags.py` — единый источник флагов: и локальный скрипт, и CI читают
его, поэтому расхождение между локальной и CI-сборкой невозможно по построению.

Сборка всегда идёт из `.venv-build`, который не пересекается с `.venv` (runtime),
`.venv-dev` и `.venv-full`. Флаг `--no-sync` у `uv run` не даёт `uv` удалить
вручную установленные `nuitka`/`zstandard`.

Наследие PyInstaller (`build_pyinstaller.sh`, `pyinstaller_build.py`,
`bacteria_analyzer.spec`) в репозитории сохранено, но текущий путь сборки —
Nuitka; собирается он в `.github/workflows/build.yaml`.

Триггеры CI в `build.yaml`:

| Событие | Ветки | Типы |
|---|---|---|
| `pull_request` | `develop` | `opened`, `synchronize`, `reopened` |
| `push` | `main`, `master` | — |
| `workflow_dispatch` | любая, вручную | — |

Feature-ветки проверяются через PR в `develop` (или ручной dispatch), отдельного
push-триггера для feature/develop веток нет. `detect-changes` пропускает build,
если затронуты только файлы `openspec/`. Иначе запускаются lint/format и тесты
в матрице `ubuntu-latest` + `windows-latest`, Python 3.13 через `uv`, далее
Nuitka-сборка и CPU inference smoke checks. Windows также запускает GUI-worker
тесты offscreen; Linux публикует JUnit pytest artifact. Сборочные артефакты
публикуются в GitHub Actions.

---

## 11. Тесты

Файлы тестов и `conftest.py` покрывают домен, pipeline/resources, Qt GUI,
фоновые операции и сборку. Точный состав меняется; актуальный перечень находится
непосредственно в `tests/`. Фикстуры включают `blank_image_bgr`,
`binary_mask_circle`, `synthetic_colony_image`, `petri_info` и др.
Запуск: см. [user-guide.md](user-guide.md#запуск-тестов).

| Область | Файлы |
|---|---|
| Детекция | `test_colony_detector.py`, `test_image_processor.py`, `test_analysis_params.py` |
| Расчёты | `test_calculations.py`, `test_metrics.py` |
| Загрузка изображений | `test_image_loader.py` |
| Реестр и алгоритмы | `test_registry.py`, `test_classic_algorithms.py`, `test_onnx_algorithm.py`, `test_tiled_onnx_algorithm.py` |
| Прогон и планировщик | `test_runner.py`, `test_pipeline_support.py`, `test_pipeline_overlap.py`, `test_resource_policy.py`, `test_telemetry.py`, `test_evaluation_cache.py` |
| GUI и responsiveness | `test_analysis_controller.py`, `test_analysis_model_selection.py`, `test_analysis_window_responsiveness.py`, `test_testing_window.py`, `test_testing_window_responsiveness.py`, `test_labeling_controller.py`, `test_labeling_window_responsiveness.py`, `test_background_operation.py` |
| Разметка | `test_session_manager.py`, labeling GUI tests |
| Данные и сплит | `test_coco_importer.py`, `test_dataset_split.py`, `test_patch_dataset.py`, `test_training_dataset.py` |
| Обучение | `test_training_loop.py`, `test_training_evaluation.py`, `test_training_pipeline.py`, `test_training_cli.py`, `test_training_cli_runtime.py`, `test_mobilenet_segmenter.py` |
| Сборка и зависимости | `test_nuitka_build.py`, `test_runtime_dependencies.py` |

Покрытые грабли, на которые стоит смотреть при изменениях: инстанцирование
алгоритма дважды, детерминизм сплита, покрытие всех пикселей тайлами,
`providers == ["CPUExecutionProvider"]` в smoke-тесте, соответствие
PyTorch/ONNX, отказ перезаписи при обучении.

---

## 12. Сквозные потоки данных

### Анализ одного снимка в GUI

```
Файл → utils.image_loader.load_image (BGR)
     → ColonyDetector.detect_petri_dish → (petri_mask, PetriInfo)
     → ColonyDetector.detect_colonies(AnalysisParams)
     → AreaCalculator.calculate_areas → AnalysisResult
     → AnalysisWindow._update_display (композит + текст результата)
```

### Фоновая операция в окне анализа

```
load/decode или (image, mask, params, generation) snapshot
   → BackgroundOperation / QThread
       → dish search / classic analysis / algorithm inference
       → status/progress/error/result queued signals
   → GUI thread: применить только актуальное generation
       → viewport presentation или результат/ошибка
```

Отмена прекращает будущие этапы и проверяется на безопасных границах; native
вызов, который нельзя прервать, возвращается естественно. Закрытие окна
откладывается без блокирующего GUI wait. Saving и presentation также используют
фоновые задачи со snapshot и проверкой актуальности.

Маска публикуется в состоянии контроллера **только** после завершения —
частичный результат в GUI не попадает.

### Оценка датасета (все режимы)

```
TestDataset / BaselineDataset
   → scheduler._sample_refs → iter_batches (лимит batch и decoded-input memory)
   → PrefetchLoader (bounded decode producer + backpressure; overlap decode/compute)
   → ThreadPoolExecutor: (sample × algorithm), с общей CPU/native-thread policy
       → SampleContext / PredictionCache.get_array → [runtime.detect] → put_array
       → compute_segmentation_metrics
   → PipelineObserver (progress/phases/errors) + structured Telemetry
   → агрегация (compute_summary / wilcoxon / outliers / winners)
   → dashboard.generate_report | export_json + evaluator HTML
```

### COCO → модель в приложении

```
datasets/22022540/annot_COCO.json
   → CocoBboxImporter.build(split=True) → {output}/dataset/{source,cropped}/ + dataset.json + split.json
   → load_manifest → make_datasets (ColonyPatchDataset + GroupedPatchBatchSampler)
   → run_training → checkpoints/best.{pt,onnx}
   → export_single / промоция → models/*.onnx
   → register_bundled_models() → "NN:<stem>" в выпадающем списке окна анализа
```

---

## 13. Точки расширения

| Задача | Что менять | Что нужно |
|---|---|---|
| Новый классический алгоритм | `testing/classic_algorithms.py` | Подкласс `_ClassicBase` + `@register_algorithm` + `params = AnalysisParams(...)` |
| Нейросетевой алгоритм со своей логикой препроцессинга | `testing/` | Подкласс `BaseDetectionAlgorithm` + `register_algorithm_instance` |
| Своя ONNX-модель | `models/*.onnx` | Положить файл; имя станет `NN:<stem>` |
| Свой источник данных для обучения | `train/dataset_adapters.py` | Подкласс `BaseAdapter`, вернуть `DatasetManifest`; при необходимости добавить в `load_manifest` |
| Новая метрика | `testing/metrics.py` | Дописать в `compute_segmentation_metrics`; агрегация подхватит ключ автоматически |
| Новый отчёт | `testing/dashboard.py` | Своя функция `_build_*` + вызов из `generate_report` |
| Новый экран GUI | `ui/` | Окно + контроллер в `ui/controllers/` без Qt-логики в контроллере |
| Новая архитектура сети | `train/models/` | Подкласс `BaseSegmenter` + `@register_model` |

Подробные инструкции по добавлению алгоритма и модели —
в [algorithms.md](algorithms.md).

---

## 14. Подводные камни

Это места, где код ведёт себя неочевидно. Проверяйте их при изменениях.

1. **Маски всегда PNG, а `TestDataset` требует одинакового расширения.**
   Разметчик сохраняет `{stem}_mask.png` независимо от формата исходника.
   `TestDataset` ищет `masks/{stem}_mask{same_ext}`. Для JPG-исходника пара не
   найдётся, и снимок будет молча пропущен. Рабочие варианты: PNG-исходники,
   одинаковые расширения, манифест `dataset.json` для `baseline`/`evaluate`.

2. **`register_bundled_models()` вызывается не везде.**
   В окне анализа — да; в `test-algorithms`, `baseline`, `evaluate` — нет.
   `NN:*`-алгоритмы там появляются только через `--model`.

3. **`--model` всегда создаёт обычный (resize) адаптер.**
   `testing/__main__.py::_register_models` не различает модели. Для
   MobileNet-файла через CLI вы получите `OnnxModelAlgorithm`, который сожмёт
   кадр до 512×512 и растянет маску обратно, вместо тайлового inference.
   Автоматический выбор тайлового адаптера работает только в
   `scan_bundled_models` (то есть для файлов из папки `models/`).

4. **GUI не показывает площадь в мм².**
   `AnalysisResult.colony_area_mm2` и `px_to_mm2` вычисляются, но в окно
   выводятся только `coverage_percent` и `colony_area_px`. `AreaCalculator`
   используется вне GUI только в тестах.

5. **Минимум отступа 1%.** `create_inner_mask` и
   `AnalysisController.calculate_algorithm_result` применяют
   `max(margin, 1.0)`. Указание «0%» не означает «ровно по краю чашки».

6. **Ключи конфигурации `evaluate` перетираются дефолтами argparse.**
   `load_config` пропускает только `None`. Чтобы `--config` влиял на
   `data_root`/`batch_size` и т. п., не полагайтесь на дефолты парсера.

7. **`--clear-cache` удаляет один файл, а не каталог.**
   Кэш предсказаний в `.cache/preds/` он не трогает.

8. **Сигнатура датасета учитывает только размеры файлов.**
   Изменение содержимого при том же размере не инвалидирует агрегатный кэш.

9. **`create_run_dir` падает при совпадении секунд.**
   `mkdir(exist_ok=False)` — два параллельных `evaluate` в одну секунду дадут
   `FileExistsError`. Это защита от молчаливой перезаписи.

10. **`AnalysisParams` валидируется через `assert`.**
    При запуске Python с `-O` проверки отключаются, и неверные значения
    пройдут дальше без предупреждения.

11. **Сохранение результата в окне анализа пишет pixmap области просмотра.**
    При зуме разрешение сохранённого PNG ниже исходного.

12. **Отмена native-вызова кооперативная.**
    Уже начатый вызов OpenCV/ONNX может завершиться до реакции на отмену;
    окна показывают «Завершение…», продолжают обрабатывать события и
    закрываются после безопасного возврата worker, не ожидая его синхронно.

13. **`TestDataset` не читает `dataset.json`.**
    GUI тестирования и режим `report` работают только с папками
    `source/` + `masks/`.

14. **Многопоточность экземпляров алгоритмов сериализована.**
    Зарегистрированные экземпляры вызываются под именованным `threading.Lock`;
    текущие `NN:*` представлены экземплярами, тогда как классовые алгоритмы
    получают экземпляры по потокам. Общая CPU/native resource policy ограничивает
    суммарную concurrency, но worker count не гарантирует ускорение: сверяйтесь
    с end-to-end benchmark для конкретного набора и машины.

15. **Нет сохранения состояния окон.**
    `QSettings` не используется: параметры анализа, выбранный датасет и
    набор алгоритмов не переживают перезапуск.
