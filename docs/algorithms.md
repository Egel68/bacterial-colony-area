# Алгоритмы детекции колоний

Справочник всех алгоритмов, которые приложение умеет применять к изображению
чашки Петри: классические (OpenCV) и нейросетевые (ONNX). Описаны контракт
алгоритма, параметры, поведение каждого адаптера и способы добавить свои.

Связанные документы: [architecture.md](architecture.md) (модули и API),
[testing-quality.md](testing-quality.md) (метрики и отчёты),
[user-guide.md](user-guide.md) (GUI и CLI).

---

## Содержание

- [1. Контракт `BaseDetectionAlgorithm`](#1-контракт-basedetectionalgorithm)
- [2. Реестр алгоритмов](#2-реестр-алгоритмов)
- [3. Обзор: что зарегистрировано](#3-обзор-что-зарегистрировано)
- [4. Классические алгоритмы](#4-классические-алгоритмы)
- [5. Нейросетевые алгоритмы](#5-нейросетевые-алгоритмы)
- [6. Контракт ONNX-модели](#6-контракт-onnx-модели)
- [7. Добавить свою модель](#7-добавить-свою-модель)
- [8. Написать свой алгоритм](#8-написать-свой-алгоритм)
- [9. Сравнение и оценка](#9-сравнение-и-оценка)
- [10. Ограничения и подводные камни](#10-ограничения-и-подводные-камни)

---

## 1. Контракт `BaseDetectionAlgorithm`

`testing/interface.py`:

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

| Требование | Детали |
|---|---|
| Вход | `np.ndarray` **BGR** `uint8`, форма `(H, W, 3)` — порядок OpenCV |
| Выход | `np.ndarray` `uint8` формы `(H, W)`, значения только `0` и `255` |
| Форма выхода | **Обязана** совпадать с `image.shape[:2]`. `AnalysisController` проверяет это и бросает `ValueError` при несовпадении |
| `is_cropped` | `True`, если кадр — обрезок чашки (чёрный фон вне круга). Классика использует это, чтобы не искать чашку заново; тайловый адаптер параметр игнорирует |
| `name` | Имя для реестра и UI. Для NN-моделей — `NN:<stem>` файла |
| `description` | Человекочитаемое описание для выпадающих списков и отчётов |

Опциональный (не входит в ABC, но используется приложением):

```python
def detect_with_progress(self, image, is_cropped=False, progress_callback=None)
```

Если метод есть **и** передан `progress_callback`, `AnalysisController`
вызовет именно его, и GUI покажет прогресс по тайлам.

---

## 2. Реестр алгоритмов

`testing/registry.py` — два словаря в памяти процесса:

| Способ | API | Ключ | Что возвращает `get_algorithm` |
|---|---|---|---|
| Класс | `@register_algorithm` над классом | `cls.__name__` | **Новый** экземпляр на каждый запрос |
| Экземпляр | `register_algorithm_instance(name, instance)` | `name` | **Тот же** объект |

```python
list_algorithms() -> list[str]                 # отсортированное объединение ключей
get_algorithm(name, **kwargs) -> алгоритм       # ValueError с перечнем доступных
get_algorithm_descriptions() -> [(name, description), ...]
```

Свойства, важные для поведения:

- **Имя — это ключ.** Регистрация молча перезаписывает существующий ключ.
  Повторные вызовы `register_bundled_models()` поэтому безопасны.
- **Приоритет у классов:** если имя есть в обоих словарях, `get_algorithm`
  вернёт свежий экземпляр класса.
- **Классы параллельны, экземпляры — нет.** `scheduler._AlgorithmRuntime`
  создаёт по экземпляру на поток для `_CLASSES` (классика работает
  параллельно), а для `_INSTANCES` берёт общий `threading.Lock` на имя и
  вызывает `detect()` строго последовательно — одна ONNX-сессия не должна
  работать из нескольких потоков одновременно.
- **Регистрация — только в памяти.** Ничего не персистится: чтобы NN-алгоритм
  появился в CLI, модель обязана быть передана через `--model` явно.

---

## 3. Обзор: что зарегистрировано

| Имя в реестре | Класс реализации | Тип регистрации | Как появляется |
|---|---|---|---|
| `ClassicDefault` | `testing.classic_algorithms.ClassicDefault` | класс | автоматически при `import testing` |
| `ClassicHighSensitivity` | `…ClassicHighSensitivity` | класс | автоматически |
| `ClassicSolidFill` | `…ClassicSolidFill` | класс | автоматически |
| `ClassicLowSensitivity` | `…ClassicLowSensitivity` | класс | автоматически |
| `NN:colony_mobilenet_v3_small` | `TiledOnnxModelAlgorithm` | **экземпляр** | `register_bundled_models()` из `models/*.onnx` |
| `NN:colony_seg` | `OnnxModelAlgorithm` | **экземпляр** | то же |
| `NN:<stem>` | `OnnxModelAlgorithm` | экземпляр | то же, для любого `models/*.onnx` |

Четыре классических алгоритма регистрируются как **классы** и всегда доступны.
Модели `NN:*` доступны только там, где вызывается `register_bundled_models()`:

| Точка | Вызов есть? | Следствие |
|---|---|---|
| Окно анализа (`main.py`, `AnalysisWindow._load_analysis_algorithms`) | да | модели в выпадающем списке |
| Окно тестирования алгоритмов | нет | модели нужно загрузить кнопкой «⬇️ Загрузить модель» |
| `test-algorithms` / `baseline` / `evaluate` | нет | модель нужно передать `--model` |

---

## 4. Классические алгоритмы

### Общая база

`testing/classic_algorithms.py::_ClassicBase` — единственное место, где
описан весь классический пайплайн. Различаются подклассы только объектом
`params: AnalysisParams`.

```python
class _ClassicBase(BaseDetectionAlgorithm):
    name = ""
    description = ""
    params: AnalysisParams | None = None

    def __init__(self):
        self.detector = ColonyDetector()

    def detect(self, image, is_cropped=False) -> np.ndarray
```

Логика `detect()`:

```
если is_cropped:
    центр = (w//2, h//2), радиус = min(w,h)//2      # чашка уже найдена
    petri_mask = круг(center, radius)
иначе:
    petri_mask, petri_info = detector.detect_petri_dish(image)
    если чашка не найдена → фолбэк: центр кадра, радиус 0.4·min(w,h)

colony_mask, _ = detector.detect_colonies(image, petri_mask,
                                          params=self.params,
                                          petri_info=petri_info)
return colony_mask
```

Обратите внимание на два фолбэка: для обрезка геометрия чашки выводится из
размеров кадра, а если автопоиск не сработал — рисуется круг в 40% от
меньшей стороны. Оба фолбэка гарантируют, что `detect()` никогда не падает на
изображении без чашки (но качество результата будет низким).

### Предустановленные наборы параметров

| Алгоритм | `sensitivity` | `min_colony_size` | `margin_percent` | `contrast` | `solid_fill` | `fill_strength` | Назначение |
|---|---|---|---|---|---|---|---|
| `ClassicDefault` | 0.5 | 50 | 10 | 1.0 | выкл | 15 | Базовая линия |
| `ClassicHighSensitivity` | 0.7 | 30 | 5 | 1.0 | выкл | 15 | Много мелких колоний, допустимы ложные срабатывания |
| `ClassicSolidFill` | 0.3 | 50 | 15 | 1.0 | **вкл** | 15 | Сливной газон, колонии слипаются в одну область |
| `ClassicLowSensitivity` | 0.3 | 100 | 10 | 1.0 | выкл | 15 | Крупные разреженные колонии, минимум шума |

### Смысл каждого параметра

| Параметр | Диапазон | На что влияет в пайплайне |
|---|---|---|
| `sensitivity` | `[0.01, 1.0]` | Через `k = 3.0 − sensitivity · 2.5` задаётся адаптивный порог `mean + k·std` (с клампом в `[mean + 5, 254]`). **Больше чувствительность → ниже порог → больше колоний**, в том числе ложных |
| `contrast` | `[0.5, 3.0]` | `clipLimit` для CLAHE = `2.0 · contrast`. Усиливает локальный контраст, помогает на блёклых снимках |
| `margin_percent` | `[0, 30]` | Радиус рабочей зоны: `radius · (100 − margin)/100`. Фактический минимум — **1%** (`max(margin, 1.0)`) |
| `min_colony_size` | `[1, 1000]` | Порог площади компоненты в пикселях (`area >= min_size`). Главный фильтр шума |
| `solid_fill` | флаг | Включает заливку внутренних областей: `drawContours(..., FILLED)` после морфологического закрытия |
| `fill_strength` | `[1, 100]` | Размер ядра эллиптического `MORPH_CLOSE` в режиме `solid_fill` (минимум 3) |

Полный разбор шагов пайплайна с формулами — в
[architecture.md § `detect_colonies`](architecture.md#4-пакет-analysis).

### Свой алгоритм на классике

Ничего нового изобретать не нужно: добавьте подкласс с собственным `params`.

```python
# testing/classic_algorithms.py
from analysis.params import AnalysisParams
from .registry import register_algorithm

@register_algorithm
class ClassicFineSmall(_ClassicBase):
    name = "Classic (fine small)"
    description = "Чувствительность 0.85, мин. размер 12"
    params = AnalysisParams(
        sensitivity=0.85, min_colony_size=12, margin_percent=5, contrast=1.2
    )
```

Класс появится в реестре при первом `import testing` и станет доступен везде:
в окне анализа, в окне тестирования, в `test-algorithms` (без `--algorithms`),
в `baseline`. Регистрация как класс даёт ещё и параллельное выполнение.

---

## 5. Нейросетевые алгоритмы

Оба адаптера — CPU-only: `onnxruntime.InferenceSession(..., providers=
["CPUExecutionProvider"])`. `onnxruntime` импортируется **лениво** в
`_get_session()`, поэтому приложение запускается и работает без него, пока
NN-алгоритм не выбран. Отсутствие пакета даёт `RuntimeError` с понятным
текстом (а не `ImportError` из импорта).

Нормализация в обоих адаптерах одинаковая и совпадает с `train`:

```
BGR → RGB → float32 / 255 → (x − mean) / std → transpose в CHW → [1, 3, H, W]
mean = [0.485, 0.456, 0.406]   std = [0.229, 0.224, 0.225]      # ImageNet
```

### `OnnxModelAlgorithm` — обычный (resize) адаптер

`testing/onnx_algorithm.py`

```python
OnnxModelAlgorithm(model_path, name="", description="",
                   img_size=512, threshold=0.5)
```

Поведение `detect()`:

```
session = _get_session(); input_name = session.get_inputs()[0].name
h, w = image.shape[:2]
resized = cv2.resize(BGR2RGB(image), (512, 512), INTER_LINEAR)
tensor   = нормализация → [1, 3, 512, 512]
raw      = session.run(None, {input_name: tensor})[0][0, 0]     # логиты [512,512]
raw      = cv2.resize(raw, (w, h), INTER_LINEAR)
binary   = (raw > threshold) * 255
```

| Свойство | Значение |
|---|---|
| Разрешение | Кадр сжимается до квадрата `img_size` и растягивается обратно |
| Порог | **строгое** `>` |
| Выход модели | Трактуется как вероятность, **сигмоида не применяется** |
| Прогресс | Не поддерживается (`detect_with_progress` отсутствует) |
| Применение | Модели, обученные на ресайзе целиком: `colony_seg.onnx` |

Плюс этого подхода — работает с любой геометрией, включая сильно вытянутые
кадры, и очень быстрый (один проход). Минус — мелкие колонии теряются при
сжатии, а маска растягивается и размывается.

### `TiledOnnxModelAlgorithm` — тайловый адаптер

`testing/tiled_onnx_algorithm.py`

```python
TiledOnnxModelAlgorithm(model_path, name="", description="",
                        img_size=512, stride=384, threshold=0.5,
                        output_is_logits=True)
```

Поведение `detect_with_progress()`:

```
ys = tile_positions(h, 512, 384);  xs = tile_positions(w, 512, 384)
total = len(ys) · len(xs)
probability_sum = zeros((h, w), float32);  coverage = zeros((h, w), uint16)

для y в ys, для x в xs:
    y2, x2 = min(h, y+512), min(w, x+512)
    tile = image[y:y2, x:x2]
    если тайл меньше 512 → copyMakeBorder до 512
        (BORDER_REFLECT_101, для вырожденных — BORDER_REPLICATE)
    tensor = нормализация(tile) → [1, 3, 512, 512]
    raw    = session.run(...)[0][0, 0]
    prob   = sigmoid(raw) если output_is_logits, иначе clip(raw, 0, 1)
    probability_sum[y:y2, x:x2] += prob[:patch_h, :patch_w]
    coverage[y:y2, x:x2]        += 1
    progress_callback(completed, total)

если остались не покрытые пиксели → RuntimeError
probability_sum /= coverage
return (probability_sum >= threshold) * 255
```

`tile_positions(length, tile_size=512, stride=384)`:

```
если length <= tile_size → [0]            # кадр меньше тайла: один тайл + padding
иначе range(0, length, stride)            # покрытие гарантировано условием stride <= tile_size
```

Проверки входа: `tile_size > 0`, `0 < stride <= tile_size`, `length > 0`,
`0 < threshold < 1` (иначе `ValueError`). Требование `stride <= tile_size` —
не формальность: именно оно гарантирует, что последний тайл дотянется до края
кадра и `coverage` нигде не останется нулевым.

| Свойство | Значение |
|---|---|
| Разрешение | Полное: маска всегда `(h, w)` исходного кадра |
| Перекрытие | `stride 384` при `tile 512` → перекрытие 128 px, вероятности **усредняются** |
| Порог | **нестрогий** `>=` (в отличие от `OnnxModelAlgorithm`) |
| Выход модели | `output_is_logits=True` → применяется сигмоида со стабилизацией `exp(-clip(x, ±80))` |
| `is_cropped` | Игнорируется (`del is_cropped`): тайлы покрывают весь кадр в обоих случаях |
| Прогресс | Есть: сигнал по числу обработанных тайлов |
| Цена | Число проходов = `ceil(размер / 384)²`; для 3000×2000 это 8×6 = 48 проходов |

Модель выбрана именно под этот режим: `colony_mobilenet_v3_small.onnx` —
4 МБ против 54 МБ у U-Net-модели, и при полном разрешении на тайлах она даёт
сопоставимое качество.

### Автовыбор адаптера

`scan_bundled_models()` в `testing/onnx_algorithm.py` решает по имени файла:

| `stem` файла в `models/` | Адаптер | Имя в реестре | Описание |
|---|---|---|---|
| `colony_mobilenet_v3_small` | `TiledOnnxModelAlgorithm` | `NN:colony_mobilenet_v3_small` | «MobileNetV3-Small: CPU-инференс на полном разрешении, с перекрывающимися тайлами» |
| `colony_seg` | `OnnxModelAlgorithm` | `NN:colony_seg` | «Legacy ONNX-модель: …» |
| любой другой | `OnnxModelAlgorithm` | `NN:<stem>` | «Встроенная нейросетевая модель …» |

**Важно:** этот выбор действует только для файлов из папки `models/`.
`--model` в CLI и кнопка «Загрузить модель» в GUI создают
`OnnxModelAlgorithm` **без разбора имени** — то есть мобильная модель,
подключённая через `--model`, пойдёт по пути resize, а не тайлов.

Специальный случай (`stem == "colony_mobilenet_v3_small"`) зафиксирован
тестом `test_tiled_onnx_algorithm.py::test_bundled_compact_model_uses_tiled_adapter_and_legacy_model_is_unchanged`.
Если вы добавите свою тайловую модель, либо расширьте список специальных имён
в `scan_bundled_models`, либо явно создайте `TiledOnnxModelAlgorithm` в коде.

### Как модель попадает в приложение

```
models/*.onnx
   → scan_bundled_models()            # выбирает адаптер по stem
   → register_algorithm_instance()    # имя NN:<stem>
   → выпадающий список окна анализа
```

Папка `models/` упаковывается в бинарник Nuitka флагом
`--include-data-files=models/*.onnx=models/` (генерируется
`scripts/nuitka_flags.py`). Путь к папке в собранном приложении разрешает
`_default_models_dir()` — сначала onefile (рядом с бинарником), затем
standalone и PyInstaller, затем каталог исходников.

---

## 6. Контракт ONNX-модели

Чтобы модель корректно работала как алгоритм детекции, она должна:

| Требование | Значение |
|---|---|
| Вход | Один вход, имя читается из `session.get_inputs()[0].name` (ожидается `input`, но берётся фактическое) |
| Форма входа | `[1, 3, H, W]`, `float32`, динамическая ось батча |
| Порядок каналов | **RGB** (адаптер сам конвертирует из BGR) |
| Нормализация | ImageNet: `(x/255 − mean)/std` |
| Выход | Один выход формы `[1, 1, H, W]`, `float32` |
| Семантика выхода | `output_is_logits=True` (по умолчанию) → логиты, адаптер применяет сигмоиду. Если модель уже выдаёт вероятности — передайте `output_is_logits=False` |
| Разрешение | Для `OnnxModelAlgorithm` — любое, адаптер ресайзит. Для `TiledOnnxModelAlgorithm` — 512×512 (или совпадающее с `img_size`) |
| Экспорт | `opset_version=18`, `external_data=False` (один файл) — см. `BaseSegmenter.to_onnx` |

Проверить готовую модель без запуска GUI:

```bash
uv run bacteria-analyzer --smoke-test-model NN:colony_mobilenet_v3_small
```

Команда создаёт нулевой кадр 97×83, прогоняет модель и печатает JSON с
`providers`, формами и набором значений выхода. Она же проверяет, что
`providers == ["CPUExecutionProvider"]` — то есть модель не уехала на GPU.

---

## 7. Добавить свою модель

### Способ 1 — положить файл в `models/` (рекомендуется)

```bash
cp my_model.onnx models/
```

Модель зарегистрируется сама как `NN:my_model`. Чтобы она стала NN:* в CLI,
передайте её явно: `--model models/my_model.onnx`.

Файл попадёт в бинарник при следующей сборке. Размер модели напрямую влияет
на размер бнарника — для 54 МБ U-Net это +50 МБ к сборке.

### Способ 2 — внешний файл без пересборки

```bash
# CLI
uv run test-algorithms --model /path/to/model.onnx --compare ClassicDefault,NN:model
```

```python
# GUI
# Окно «Тестирование алгоритмов» → «⬇️ Загрузить модель (.onnx)…»
# Окно анализа: модели вне models/ в списке не появятся
```

### Способ 3 — тайловая модель под своим именем

```python
# testing/onnx_algorithm.py → scan_bundled_models()
if model_file.stem == "my_tiled_model":
    algos.append(
        TiledOnnxModelAlgorithm(
            model_path=str(model_file),
            name=f"NN:{model_file.stem}",
            description="Моя модель: тайловый CPU-инференс",
        )
    )
    continue
```

### Чек-лист совместимости

- [ ] Модель выдаёт **логиты**, а не вероятности (иначе `output_is_logits=False`)
- [ ] Один вход, один выход; имена берутся из метаданных ONNX
- [ ] Экспорт с `opset 18`, `external_data=False` (один файл)
- [ ] Нет внешних `.data`-файлов рядом
- [ ] Модель в формате fp32, не fp16 (CPU-исполнение fp16 нестабильно)
- [ ] Вход 512×512 для тайловой модели, произвольный — для resize-модели
- [ ] Проверка проходит: `--smoke-test-model <имя>`
- [ ] Хеш файла меньше разумного предела (модель уезжает в бинарник целиком)

---

## 8. Написать свой алгоритм

### Вариант A — на базе классики (10 строк)

См. раздел [4. Классические алгоритмы](#свой-алгоритм-на-классике). Подходит,
если нужно лишь изменить набор параметров.

### Вариант B — своя обработка OpenCV

```python
# testing/my_algorithm.py
import cv2
import numpy as np

from .interface import BaseDetectionAlgorithm
from .registry import register_algorithm


@register_algorithm
class MyAdaptiveThreshold(BaseDetectionAlgorithm):
    """Порог по центральной области с коррекцией освещённости края."""

    name = "My (adaptive local)"
    description = "Адаптивный порог с локальным выравниванием"

    def __init__(self, block: int = 151, c: float = 8.0):
        self.block = block
        self.c = c

    def detect(self, image: np.ndarray, is_cropped: bool = False) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")

        channel = image[:, :, 1]                              # зелёный канал
        normed = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        normed = cv2.medianBlur(normed, 5)

        mask = cv2.adaptiveThreshold(
            normed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.block, self.c,
        )
        return cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
```

Подключение: добавьте `from . import my_algorithm` в `testing/__init__.py`
(рядом с `classic_algorithms`) — этого достаточно, чтобы класс зарегистрировался
при любом `import testing`.

### Вариант C — обёртка над своей ONNX-моделью

```python
from .onnx_algorithm import OnnxModelAlgorithm
from .registry import register_algorithm_instance

algo = OnnxModelAlgorithm(
    model_path="models/my_model.onnx",
    name="NN:my_model",
    description="Моя модель, resize-режим",
    img_size=384,        # под своё обучение
    threshold=0.45,      # подобрать на валидации
)
register_algorithm_instance("NN:my_model", algo)
```

Для собственной препроцессинки наследуйте `BaseDetectionAlgorithm` и
переопределите `detect()` — копируйте `OnnxModelAlgorithm.detect` как образец
и правьте только блок препроцессинга.

### Вариант D — с прогрессом для GUI

```python
class MySlowAlgorithm(BaseDetectionAlgorithm):
    name = "My (slow, tiled)"
    description = "Мой тайловый алгоритм с отчётом о прогрессе"

    def __init__(self):
        self.steps = 10

    def detect(self, image, is_cropped=False):
        return self.detect_with_progress(image, is_cropped=is_cropped)

    def detect_with_progress(self, image, is_cropped=False, progress_callback=None):
        h, w = image.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        for i in range(self.steps):
            out[i * h // self.steps:(i + 1) * h // self.steps] = 255
            if progress_callback:
                progress_callback(i + 1, self.steps)
        return out
```

Наличие `detect_with_progress` — единственное, что нужно, чтобы GUI
показывал полосу прогресса вместо бесконечного индикатора.

### Правила и проверка

| Правило | Почему |
|---|---|
| Возвращать ровно `(H, W)` `uint8` с `0`/`255` | `AnalysisController` проверяет форму; тесты ждут бинарных значений |
| Принимать BGR | Весь код приложения оперирует BGR |
| Не менять входной массив | `original_image.copy()` передаётся в поток; мутация испортит пересчёты |
| Не блокировать GUI надолго без `detect_with_progress` | Иначе окно «замерзнет» на время инференса |
| Регистрировать как **класс**, если алгоритм stateless | Тогда он получит собственный экземпляр на поток и будет работать параллельно |
| Обрабатывать `is_cropped` осмысленно | Иначе классика будет искать чашку внутри обрезка и промахиваться |
| Описать `description` | Показывается пользователю в выпадающих списках и в HTML-отчёте |

Проверка нового алгоритма тестами:

```bash
# быстрый цикл
UV_PROJECT_ENVIRONMENT=.venv-dev uv run pytest tests/test_registry.py \
  tests/test_classic_algorithms.py -q

# оценка качества на датасете
uv run test-algorithms --algorithms MyAlgorithm,ClassicDefault \
  --output my_report.html
```

---

## 9. Сравнение и оценка

### Команды

```bash
# Полный прогон всех зарегистрированных алгоритмов → HTML
uv run test-algorithms --data-root ./test_images --output ./report.html

# Выборочно + внешняя модель + парное сравнение
uv run test-algorithms --algorithms ClassicDefault,NN:colony_seg \
  --model models/colony_seg.onnx \
  --compare ClassicDefault,NN:colony_seg --no-per-snapshot

# JSON для автоматической обработки (только 4 классических)
uv run baseline --data-root ./test_images --output ./baseline.json

# Каталог прогона с конфигом, git-инфо и отчётами
uv run evaluate --data-root ./test_images
```

### Как читать результат

| Метрика | На что указывает высокое значение |
|---|---|
| **IoU** | Общая точность сегментации. Главная метрика для сравнения моделей |
| **Dice** | То же, но чувствительнее к балансу классов; ближе к F1 для масок |
| **Precision** | Мало ложных срабатываний (маска не «размазана» по фону) |
| **Recall** | Мало пропусков (все колонии найдены) |
| **F1** | Компромисс precision/recall |
| **Accuracy** | Для колоний почти всегда завышена: фон занимает большую часть кадра, поэтому accuracy высока даже у плохой модели. **Ориентироваться на неё не стоит** |

Типичный профиль ошибок:

| Симптом | Вероятная причина |
|---|---|
| Высокий recall, низкий precision | Занижен порог (классика: высокая чувствительность) или слабая модель |
| Низкий recall, высокий precision | Завышен порог, слишком большой `min_colony_size` |
| Мелкие колонии теряются | Resize-модель вместо тайловой либо слишком большой `min_colony_size` |
| Колонии «размазаны», растёт фон | Классика с высокой чувствительностью либо тайлы с малым `stride` |
| Метрики скачут между снимками | Смотрите таблицу выбросов (|z| > 3) в отчёте |

Отчёты содержат не только средние, но и разброс: `std`, медиану, квартили,
`p5`/`p95`, min/max, таблицу выбросов и попарный Wilcoxon-критерий. Если у двух
алгоритмов `p < 0.05`, разница между ними статистически значима, а не шумом.

Подробности про метрики, агрегацию, кэш и telemetry — в
[testing-quality.md](testing-quality.md).

---

## 10. Ограничения и подводные камни

1. **NN-алгоритмы недоступны в CLI без `--model`.** `register_bundled_models()`
   не вызывается в `testing/__main__.py`.

2. **`--model` всегда создаёт resize-адаптер.** Мобильная модель, подключённая
   через `--model`, пойдёт по пути «сжать до 512 и растянуть» вместо тайлов.
   Для честного сравнения положите модель в `models/` и перезапустите
   приложение — или создайте `TiledOnnxModelAlgorithm` в коде.

3. **Два разных оператора сравнения с порогом.** `OnnxModelAlgorithm`
   использует `raw > threshold`, `TiledOnnxModelAlgorithm` — `>=`.
   На вероятности ровно 0.5 результат различается.

4. **Разная интерпретация выхода модели.** По умолчанию
   `output_is_logits=True`: адаптер применяет сигмоиду. Модель, уже выдающая
   вероятности, будет интерпретирована неверно (и даст маску «везде»).
   Лечится `output_is_logits=False`.

5. **Тайловая модель чувствительна к `stride`.** `stride == tile_size` убирает
   перекрытие и артефакты на стыках; меньший `stride` глаже, но заметно
   медленнее. Значение `stride > tile_size` запрещено и привело бы к
   непокрытым пикселям.

6. **Перекрывающиеся тайлы усредняются по вероятностям, а не по маскам.**
   Это лучше, чем усреднение после бинаризации, но требует, чтобы модель
   выдавала осмысленные вероятности, а не «0 или 10».

7. **Сигмоид защищён от переполнения.** `1/(1+exp(-clip(x, ±80)))` — при
   экстремальных логитах `np.exp` не даст inf/nan.

8. **Кэш предсказаний не различает тайловый и resize-режим для одного файла.**
   Ключ включает `type(algo).__qualname__`, `model_path`, `img_size`,
   `threshold` и `is_cropped` — этого достаточно, чтобы не перепутать
   Classic и NN, но замена файла модели по тому же пути кэш не сбросит.
   Сбрасывайте `.cache/preds/` вручную при подмене весов.

9. **Классика на обрезках полагается на «чашка = весь круг».** Для `is_cropped=True`
   радиус берётся как `min(w, h) // 2`, то есть предполагается, что обрезок
   плотно обрезан по чашке. Если в кадре остались поля, часть колоний уйдёт
   за рабочую зону.

10. **`solid_fill` заливает всё, что нашёл, включая артефакты.** Режим предназначен
    для сливного газона; на разреженных посевах он склеит отдельные колонии в
    один компонент и занизит счётчик.

11. **Замеренное качество NN-модели (`IoU ≈ 0.72` на валидации 22022540) не
    переносится на другие чашки.** Метрики зависят от датасета, распределения
    колоний и параметров съёмки. Всегда прогоняйте `test-algorithms` на своих
    данных.
