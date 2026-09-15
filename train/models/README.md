# Как подключить собственную модель (архитектуру) к фреймворку обучения

В проекте есть минимальный ML-фреймворк для обучения сверточных сетей сегментации
колоний бактерий. Он состоит из:

| Модуль | Назначение |
|---|---|
| `train/models/base.py` | Базовый класс сети `BaseSegmenter` (loss, метрики, ONNX-экспорт) |
| `train/models/__init__.py` | Реестр архитектур: `@register_model`, `get_model`, `list_models` |
| `train/models/unet.py`, `unet_small.py` | Готовые примеры архитектур |
| `train/train.py` | Цикл обучения `run_training` |
| `train/compare.py` | Сравнение нескольких архитектур (`train-compare`) |
| `train/dataset.py` | Загрузка пар «изображение + маска» (контракт данных) |
| `testing/onnx_algorithm.py` | Адаптер ONNX-модели к фреймворку тестирования (`OnnxModelAlgorithm`) |

Эта инструкция описывает, как написать свою архитектуру, зарегистрировать её,
обучить и подключить к фреймворку тестирования алгоритмов.

---

## 1. Окружение

Обучение и инференс требуют ML-зависимостей (`torch`, `onnxruntime`, ...):

```bash
# Создание полного окружения (~5 ГБ, включает torch + CUDA)
UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full

# Активация
source .venv-full/bin/activate
```

Список установленных архитектур:
```bash
uv run python -m train.main list-models
# → unet
# → unet_small
```

---

## 2. База: `BaseSegmenter`

Любая будущая модель наследует `BaseSegmenter` (`train/models/base.py`),
который уже предоставляет:

| Метод | Что делает |
|---|---|
| `forward(x) -> Tensor` (абстрактный) | Прямой проход, вернёт логиты (без sigmoid) |
| `predict(x)` | `sigmoid(forward(x))` для инференса |
| `get_loss(pred, target)` | Loss BCE + Dice (`smooth=1e-6`) |
| `compute_metrics(pred, target, threshold=0.5)` | loss, iou, dice, precision, recall |
| `to_onnx(path)` | Экспорт в ONNX (opset 18, dynamic batch) |
| `load_onnx(path)` | Загрузка ONNX-сессии |
| `preprocess(image, img_size=512)` | BGR→RGB, resize, нормализация ImageNet |

**Минимум от вашей архитектуры — реализовать `forward`**, возвращающий логиты
размерности `[B, C=1, H, W]` (для бинарной сегментации один выходной канал).

---

## 3. Шаг за шагом: добавить новую архитектуру

### 3.1 Создать файл модуля

Пример: `train/models/my_model.py`:

```python
import torch
import torch.nn as nn

from . import register_model          # декоратор реестра
from .base import BaseSegmenter


@register_model("my_model")
class MyModel(BaseSegmenter):
    """Моя архитектура сегментации колоний."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1,
                 features: tuple = (32, 64, 128, 256)):
        super().__init__()
        # --- постройте сеть здесь ---
        self.features = features
        # Пример для иллюстрации (реальную сеть подставьте на своё усмотрение):
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, features[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[-1], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(features[-1], features[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Возвращает ЛОГИТЫ (без sigmoid) размерности [B, n_classes, H, W]
        return self.decoder(self.encoder(x))
```

Требования к архитектуре:
- **Параметризуемость**: конструктор принимает гиперпараметры (`features`,
  `n_channels`, `n_classes`, ...), чтобы их можно было менять через
  `get_model(name, **params)`.
- Выход — **логиты**: sigmoid применяет сам `BaseSegmenter` (`predict`,
  `get_loss`, `compute_metrics`, `to_onnx`).
- Наследование от `BaseSegmenter` обязательно (иначе фреймворк не сможет
  посчитать loss/метрики и экспортировать в ONNX).

### 3.2 Зарегистрировать модель в реестре

`@register_model("my_model")` регистрирует архитектуру. Чтобы импорт модуля
произошёл при запуске, добавьте его в **`train/models/__init__.py`**:

```python
from .unet import UNet as UNet              # существующее
from .unet_small import UNetSmall as UNetSmall  # существующее
from .my_model import MyModel as MyModel        # ← ваша модель (new)
```

Именно строка импорта «срабатывает» декоратор — иначе модель не появится в
реестре, даже если файл существует.

### 3.3 Проверить регистрацию

```bash
uv run python -m train.main list-models
# → my_model
# → unet
# → unet_small
```

Программно:

```python
from train.models import get_model, list_models

print(list_models())                      # ["my_model", "unet", "unet_small"]
model = get_model("my_model", features=(16, 32, 64, 128))  # с параметрами
model = get_model("nope")                 # ValueError: Unknown model
```

---

## 4. Обучить модель

### Единственная архитектура

Данные для `train` берутся из `TrainingConfig.data_root` (по умолчанию
`train/data` — подготовьте пары в этой папке или правьте конфиг):

```bash
uv run python -m train.main train \
  --model my_model \
  --epochs 50 \
  --img-size 256 \
  --batch-size 4 \
  --dashboard        # вкл. метрики в реальном времени (порты 8765 + TensorBoard 6006)
```

### Сравнение архитектур (`train-compare`)

`train-compare` принимает собственные `--data-root` и `--eval-root`:

```bash
uv run python -m train.main train-compare \
  --architectures my_model,unet,unet_small \
  --data-root train/data \
  --eval-root test_images \
  --epochs 5
```

> Примечание: CLI обучает с **дефолтными** параметрами конструктора архитектуры
> (передача `features=...` и прочих возможна программно — см. §3.3).

Результаты:
- per-архитектура: `train/runs/{model}_{timestamp}/` с `best.pt`, `best.onnx`,
  `last.pt`, `last.onnx`, `summary.json`, `report.html`, `tensorboard/`;
- сводный отчёт: `train/runs/compare_{timestamp}/compare.json` + `compare.html`.

### Где смотреть метрики в реальном времени

| Канал | Как включить | Что показывает |
|---|---|---|
| rich-прогресс (терминал) | по умолчанию | loss/IoU/Dice по эпохам |
| Web Dashboard | `--dashboard` | Chart.js, порт 8765 |
| TensorBoard | `--dashboard` (авто) | полные графики `train/*`, `val/*`, порт 6006 |

---

## 5. Подключить модель к фреймворку тестирования алгоритмов

Обученная модель подключается к `testing/` в **формате ONNX** — через
`OnnxModelAlgorithm` (`testing/onnx_algorithm.py`), который реализует
`BaseDetectionAlgorithm` и может детектировать маски как классический алгоритм.

### 5.1 Обученные веса → ONNX

`run_training` автоматически экспортирует `best.onnx` и `last.onnx` в
`train/runs/{model}_{timestamp}/checkpoints/`. Экспорт вручную:

```python
import torch
from train.models import get_model

model = get_model("my_model")
model.load_state_dict(torch.load("weights.pt", map_location="cpu"))
model.to_onnx("my_model.onnx")
```

### 5.2 Инференс через testing-фреймворк

**Вариант A — недостроенный `.onnx` разметить в `models/` (встроенная модель):**

```bash
cp train/runs/my_model_.../checkpoints/best.onnx models/
```

При старте приложения модель появится в списке алгоритмов как `NN:best`
(сканируется `models/*.onnx`).

**Вариант B — внешний файл без пересборки (CLI):**

```bash
uv run test-algorithms \
  --data-root test_images \
  --model train/runs/my_model_.../checkpoints/best.onnx \
  --compare NN:best,ClassicDefault
```

**Вариант C — в GUI:**
- «🧪 Тестирование алгоритмов» → кнопка «Загрузить модель (.onnx)» → выбрать файл.

### 5.3 Программно

```python
import numpy as np
from testing.onnx_algorithm import OnnxModelAlgorithm

algo = OnnxModelAlgorithm(model_path="best.onnx", img_size=512)
image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
mask = algo.detect(image, is_cropped=False)   # uint8 [H, W], значения 0/255
```

Оценка на тестовых парах:
```python
from testing.dataset import TestDataset
from testing.runner import run_algorithm

dataset = TestDataset(root="test_images")
results = run_algorithm(algo, dataset)   # метрики по каждому снимку (source/cropped)
```

---

## 6. Чек-лист подключения

- [ ] Сеть наследует `BaseSegmenter` (реализован `forward`, логиты).
- [ ] Конструктор параметризуем (`features`, `n_channels`, `n_classes`, ...).
- [ ] Файл модуля создан в `train/models/`.
- [ ] `from .my_model import MyModel` добавлен в `train/models/__init__.py`
      (иначе декоратор не сработает).
- [ ] `list-models` показывает имя новой модели.
- [ ] Обучается: `train --model my_model`.
- [ ] В `checkpoints/` появились `best.onnx` — модель готова к подключению через
      `OnnxModelAlgorithm` / `test-algorithms --model` / GUI.

---

## 7. Реестр моделей — справочник

См. `train/models/__init__.py`:

```python
def register_model(name: str):        # декоратор: регистрация класса
def get_model(name: str, **kwargs):   # создание инстанса (параметры в kwargs)
def list_models() -> list[str]:       # сортированный список имён
```

Неизвестное имя → `ValueError` с перечнем доступных моделей.

`BaseSegmenter` (полная сигнатура) — `train/models/base.py`:

```python
class BaseSegmenter(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def predict(self, x): ...                 # sigmoid(forward)
    def get_loss(self, pred, target): ...     # BCE + Dice
    def compute_metrics(self, pred, target, threshold=0.5): ...
    def to_onnx(self, path, input_shape=(1, 3, 512, 512)): ...  # opset 18, dynamic batch
    @staticmethod
    def load_onnx(path): ...                  # onnxruntime.InferenceSession
    def preprocess(self, image, img_size=512): ...  # для ручного инференса
```