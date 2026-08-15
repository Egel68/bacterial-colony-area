# Bacterial Colony Area Analyzer

Python 3.13+ / PyQt6 / OpenCV / NumPy desktop app for analysing bacterial colonies on Petri dish images.

---

## 1. Быстрый старт

### Из исходников (разработка)

```bash
# Установка runtime-окружения (~600 МБ)
uv sync

# Запуск приложения
uv run bacteria-analyzer
```

### Из бинарника (пользователи)

Скачайте собранный бинарник со страницы [Releases](https://github.com/Egel68/bacterial-colony-area/releases).  
Файл называется `BacteriaAnalyzer` (Linux) или `BacteriaAnalyzer.exe` (Windows). Просто запустите его — дополнительные зависимости не нужны.

---

## 2. Функционал приложения

### 2.1. Анализ колоний (главный режим)

**Назначение:** автоматический подсчёт колоний бактерий на изображении чашки Петри.

**Как использовать:**

1. Нажмите «📂 Открыть» и выберите изображение чашки Петри (поддерживаются PNG, JPG, BMP, TIFF, WebP).
2. Нажмите «🔍 Анализировать».
3. Откроется окно анализа с изображением и панелью параметров.

**Параметры анализа:**

| Параметр | Диапазон | Описание |
|---|---|---|
| Чувствительность | 1–100% | Чем выше, тем меньше колоний детектируется (порог адаптивной бинаризации) |
| Контраст | 0.5–3.0x | Усиление контраста перед детекцией (CLAHE) |
| Отступ от края | 0–30% | Сужение рабочей зоны от края чашки, % от радиуса |
| Мин. размер колонии | 1–1000 px | Фильтрация мелких шумов |
| Solid fill | вкл/выкл | Режим для сливного газона — заливка областей целиком |
| Сила fill | 1–100 | Радиус ядра морф. закрытия для solid fill |

**Режимы просмотра** (выпадающий список над изображением):

| Режим | Что показывает |
|---|---|
| Результат | Оригинал + контуры найденных колоний |
| Оригинал | Исходное изображение |
| Предобработка | Зелёный канал + CLAHE (как видит алгоритм) |
| Бинарная маска | Ч/б маска найденных колоний |

**Кнопки:**

- «💾 Сохранить результат» — сохраняет текущее изображение (с наложениями) в PNG.
- «Сбросить авто-детекцию» — перезапускает поиск чашки Петри.
- «Пересчитать» — повторяет детекцию с новыми параметрами.

**Процесс анализа (внутренности):**

```
Загрузка изображения → поиск чашки Петри (отражения → HoughCircles)
→ извлечение зелёного канала → CLAHE → медианный blur
→ вычитание фона (Gaussian blur 51×51)
→ адаптивная бинаризация (mean + k·std)
→ морфология (open/close) → connected components
→ фильтрация по размеру → подсчёт площади и %
```

### 2.2. Разметка изображений (Labeling)

**Назначение:** ручное создание ground-truth масок для обучения и тестирования алгоритмов.

#### 2.2.1. Начало сессии разметки

На главном экране нажмите «✏️ Разметка тестовых изображений».  
Откроется диалог **«Новая сессия разметки»**:

1. Введите **название сессии** — папка сессии создастся автоматически в стандартном хранилище `~/BacteriaLabeling/`.
2. Хранилище сессий можно сменить кнопкой **«✏️ Изменить»** — выбор сохраняется между запусками.
3. Повторный вход: сессию можно открыть одним кликом из списка **«Недавние сессии»** или через **«📂 Открыть существующую папку…»** для произвольной папки.

Структура папок создаётся автоматически; в окне разметки всегда показан полный путь к текущей рабочей папке.

**Структура папки сессии:**

```
session_dir/
├── photo1.png, my_image.jpg, ...   ← исходные изображения (можно добавить через интерфейс)
├── masks/                           ← маски исходников (*_mask.png, 255=колония)
├── cropped/                         ← обрезки чашек Петри (*_cropped.png)
└── cropped_masks/                   ← маски обрезков (*_cropped_mask.png)
```

#### 2.2.2. Интерфейс разметки

Окно разделено на две части:

**Левая панель (файлы):**

| Элемент | Назначение |
|---|---|
| «📁 Исходные изображения» / «✂️ Обрезки чашек» | Переключение режимов: исходники или обрезки |
| Текущий путь (под заголовком) | Полный путь к рабочей папке сессии |
| Список файлов | Клик для загрузки изображения |
| «🔄 Обновить» | Перечитать список файлов |
| «📂 Добавить изображения» | Выбрать файлы на компьютере — они скопируются в `source/` |
| «🔍 Чашка Петри» | Координаты центра и радиус чашки (спиннеры + авто-поиск + обрезка) |

**Центральная область (изображение):**

- Изображение + маска (зелёная полупрозрачная накладка)
- Можно переключить на режим «Только маска»
- Под изображением — **панель инструментов**:

**Панель инструментов (слева направо):**

| Элемент | Назначение |
|---|---|
| Размер кисти − / `20 px` / + | Уменьшить/увеличить кисть (шаг 2px, диапазон 2–100px) |
| `100%` | Текущий зум |
| Зум + / − / 1:1 | Приблизить/отдалить/сбросить (0.1× – 20×) |
| «🖼 Изображение + маска» / «🏁 Только маска» | Переключение режима отображения |
| «✏️ Рисовать» / «🧹 Ластик» | Переключение режима кисти (рисует/стирает) |
| «🗑 Очистить» | Сбросить всю маску |
| «💾 Сохранить маску» | Сохранить текущую маску в `masks/` или `cropped_masks/` |
| «📦 Экспорт в ZIP» | Упаковать всю папку сессии в ZIP-архив |

#### 2.2.3. Типовой сценарий разметки

1. На главном экране → «✏️ Разметка» → выбрать папку (или создать новую).
2. Нажать «📂 Добавить изображения» → выбрать исходные фото чашек Петри.
3. Выбрать файл из списка — чашка определяется автоматически.
4. Если авто-поиск не сработал: настроить центр и радиус вручную (спиннеры в левой панели) или нажать «🔍 Авто-поиск».
5. (Опционально) Нажать «✂️ Обрезать по чашке» — программа вырежет чашку, закрасит фон чёрным, переключится в режим «Обрезки чашек». Дальше можно размечать маску на обрезке.
6. Рисовать маску кистью (зелёная накладка поверх изображения).
   - «✏️ Рисовать» — закрашивает колонии.
   - «🧹 Ластик» — стирает ошибочные закрашивания.
   - Колёсико мыши — зум.
   - Кнопки «+» / «−» — размер кисти.
7. Нажать «💾 Сохранить маску».
8. Повторить для всех изображений.
9. Нажать «📦 Экспорт в ZIP» — выбрать, куда сохранить архив.
10. Отправить ZIP автору программы.

#### 2.2.4. Импорт размеченных данных в репозиторий

```bash
unzip session.zip
cp -r session/source/* test_images/source/
cp -r session/masks/* test_images/masks/
cp -r session/cropped/* test_images/cropped/
cp -r session/cropped_masks/* test_images/cropped_masks/
```

### 2.3. Тестирование алгоритмов

**Назначение:** сравнение алгоритмов детекции на эталонных изображениях.

Запуск доступен двумя способами: **GUI-окно в приложении** («🧪 Тестирование алгоритмов» на главном экране) и **CLI**.

```bash
# Из runtime-окружения
uv run test-algorithms
# Или указать путь к данным, алгоритмы, модели и отчёт:
uv run test-algorithms --data-root ./test_images --output ./my_report.html \
  --algorithms ClassicDefault,ClassicSolidFill \
  --model path/to/model.onnx --compare ClassicDefault,ClassicSolidFill \
  --no-per-snapshot
```

**Что делает:**
1. Загружает парные изображения из `test_images/source/` + `masks/` и `cropped/` + `cropped_masks/`.
2. Прогоняет выбранные алгоритмы (по умолчанию — все зарегистрированные) на всех изображениях.
3. Считает метрики: IoU, Dice, F1, Precision, Recall, Accuracy.
4. Генерирует HTML-отчёт `test_report.html` с таблицами и графиками Chart.js.

**Параметры CLI:**

| Флаг | Описание |
|---|---|
| `--data-root` | Корень датасета (по умолчанию `test_images`) |
| `--output` | Путь к HTML-отчёту |
| `--algorithms A,B` | Подмножество алгоритмов для прогона |
| `--model <path>` | Подключить внешнюю ONNX-модель как алгоритм (повторяемый) |
| `--compare A,B` | Парный сравнение: «победитель по снимку» |
| `--per-snapshot` / `--no-per-snapshot` | Детальные метрики по снимкам (по умолчанию включены) |

**Алгоритмы и модели:**

- Классические алгоритмы регистрируются через `@register_algorithm` (классы).
- Нейросетевые/ONNX-модели — через `register_algorithm_instance` (готовый экземпляр).
  Единый интерфейс: `BaseDetectionAlgorithm` (`name`, `description`, `detect(image, is_cropped)`).
- Внешние веса: файл `.onnx` подгружается через GUI-кнопку «Загрузить модель» или `--model` — без пересборки приложения.
- Встроенные веса: файлы `models/*.onnx` упаковываются в бинарник Nuitka (`--include-data-files=models/*.onnx=models/`) и регистрируются автоматически при запуске.
- `onnxruntime` импортируется лениво (в `detect()`); без него приложение работает, NN-алгоритмы недоступны с понятной ошибкой.

**Доступные алгоритмы:**

| Алгоритм | Описание |
|---|---|
| ClassicDefault | Чувствительность 50%, отступ 10%, мин. размер 50px |
| ClassicHighSensitivity | Чувствительность 70%, отступ 5%, мин. размер 30px |
| ClassicSolidFill | Solid fill для сливных чашек (чувствительность 30%, fill 15) |
| ClassicLowSensitivity | Чувствительность 30%, отступ 10%, мин. размер 100px |

### 2.4. Аугментация данных (CLI)

**Назначение:** генерация обучающей выборки для U-Net.

```bash
uv run augment
```

Генерирует 15 вариантов каждой пары изображение+маска из `test_images/cropped/`:
- **Геометрические:** Rotate (произвольный), HorizontalFlip, VerticalFlip, Affine (scale ±10%, translate ±5%)
- **Пиксельные:** BrightnessContrast, GaussNoise, Blur, HueSaturationValue

Результат: `train/data/images/` + `train/data/masks/` (по ~15 файлов на каждый исходник).

### 2.5. Обучение U-Net (CLI, требуется dev-окружение)

**Назначение:** обучение нейросетевой сегментации колоний.

```bash
# Установка dev-окружения (~5 ГБ, включает torch + CUDA)
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev

# Активация dev-окружения
source .venv-dev/bin/activate

# Запуск обучения
UV_PROJECT_ENVIRONMENT=.venv-dev uv run python -m train.main train --model unet --epochs 200 --dashboard
```

**Параметры `train.main train`:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--model` | unet | Архитектура (список: `unet`, `unet_small`; полный — `list-models`) |
| `--epochs` | 200 | Максимум эпох |
| `--batch-size` | 8 | Размер батча |
| `--lr` | 1e-3 | Learning rate |
| `--img-size` | 512 | Размер изображений при обучении |
| `--data-root` | train/data | Корень обучающих данных (манифест/папка пар) |
| `--no-augment` | — | Отключить аугментацию |
| `--dashboard` | — | Включить веб-дашборд (порт 8765) |

**Сравнение архитектур (`train.main train-compare`):**

```bash
uv run python -m train.main train-compare --architectures unet,unet_small \
  --data-root train/data --eval-root test_images --epochs 2
```

Обучает каждую архитектуру из списка и оценивает `best.onnx` каждой на тестовых парах
(`TestDataset` из `testing/`), формируя сводный отчёт в `train/runs/compare_{timestamp}/`
(`compare.json` + `compare.html`).

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--architectures` | unet,unet_small | Список архитектур через запятую |
| `--data-root` | train/data | Корень обучающих данных |
| `--eval-root` | test_images | Корень тестовых пар для оценки |

**Мониторинг в реальном времени:**

- **TensorBoard** — автоматически запускается на порту 6006, живёт после завершения обучения.
- **Web Dashboard** (`--dashboard`) — Chart.js на порту 8765 (живёт только пока идёт обучение).

**Результаты обучения:**

```
train/runs/{model}_{timestamp}/
├── best.pt                    # Лучшая модель (по IoU на валидации)
├── best.onnx                  # best.pt в формате ONNX
├── last.pt                    # Последняя модель
├── last.onnx                  # last.pt в формате ONNX
├── report.html                # Интерактивный отчёт с графиками (Plotly)
└── tensorboard/               # Логи TensorBoard
```

**Просмотр отчёта после обучения:**
```bash
# Откройть в браузере
xdg-open train/runs/unet_20260204_120000/report.html

# Или через TensorBoard
tensorboard --logdir train/runs/unet_20260204_120000/tensorboard
```

### 2.6. Контракт данных и задел импортёров

**Контракт данных.** Обучение не знает о конкретных структурах папок: оно
потребляет **манифест** `DatasetManifest` — список пар «изображение + маска» с
метаданными (`mask_mode`, `subset`, `kind`, `storage`). Источники данных
подключаются через адаптеры (`train/dataset_adapters.py`):

| Адаптер | Что читает |
|---|---|
| `ManifestAdapter` | Готовый `dataset.json` в корне датасета |
| `LabelingAdapter` | Структуру сессии разметки: `source/` + `masks/{stem}_mask`, `cropped/` + `cropped_masks/{stem}_cropped_mask` (без копирования, `storage="reference"`) |
| `PairsAdapter` | Легаси `images/` + `masks/` по совпадающим именам |

Фабрика `load_manifest(data_root)` выбирает адаптер в порядке: `dataset.json` →
`source/` → `images/`. `make_datasets` разбивает записи на train/val по `subset`
из манифеста или, при его отсутствии, seed-сплитом от `val_split`.

**Задел на будущее (НЕ реализовано):**

- **Импортёры внешних датасетов.** Планируется подключать внешние данные
  (COCO / VOC / произвольные пары) через декоратор `@register_importer`: импортёр
  растризует анотации в `{id}_mask.png`, материализует файлы (`storage="copy"` —
  самодостаточная папка, либо `reference` — ссылки на местонахождение) и пишет
  `dataset.json` той же схемы, что и `ManifestAdapter`. Ядро обучения при этом не
  меняется — контракт уже готов.
- **Multiclass-маски.** `mask_mode` в манифесте резервирует значение
  `multiclass`. Сейчас конвейер (loss BCE+Dice, бинаризация порогом 0.5)
  поддерживает только `binary` и бросает понятную ошибку при попытке обучать
  multiclass. Точки расширения помечены комментариями в `train/dataset_manifest.py`.

### 2.7. Импорт внешнего датасета 22022540 (COCO-bbox) и готовая модель

**Первый материализованный импортёр** — `CocoBboxImporter` (`train/dataset_adapters.py`),
приводит детекционный датасет `datasets/22022540` (369 изображений, 56 865 колоний,
24 вида, аннотации COCO/TSV/YOLO/VOC) к контракту обучения: растризует боксы во
**вписанные эллипсы** бинарной маской (255 = колония) и пишет самодостаточную папку
с `dataset.json` (читается `ManifestAdapter`). Модель определяет **наличие колонии**,
а не класс бактерии.

```bash
# Полный импорт (source-снимки, ~607 МБ, 2–3 мин; без GUI)
uv run import-22022540 --data-root datasets/22022540 --output <dir> [--crop] [--verify]

# Создание пары "изображение + маска" и проверка читаемости манифестом:
#   --output/<dir>/source/*.jpg + source/*_mask.png + dataset.json
#   load_manifest(<dir>) SHALL резолвиться через ManifestAdapter
```

- `--crop` — дополнительно формирует обрезки по чашке (`kind=cropped`, чёрный фон
  вне круга); при этом `dish={cx,cy,r}` вычисляется авто-поиском (`analysis/colony_detector.py`).
  Для source-only-импорта авто-поиск чашки пропускается ради скорости.
- `--verify` — сверка числа масок на снимок с числом боксов из COCO и YOLO/VOC-дублей;
  предупреждает при расхождении > 5 (в этом датасете форматы согласованы, 0 расхождений).
- Источник читается только; выход пишется в `--output` (`storage="copy"`).

**Готовая модель** — `models/colony_seg.onnx` (U-Net, обучен на импортированном
датасете 22022540, бинарная сегментация). Автоматически регистрируется при старте
как алгоритм `NN:colony_seg` через `register_bundled_models`; сигнатура
`detect(image)->uint8 mask` (порог 0.5). Базовые метрики на валидации: IoU 0.72,
Dice 0.83.

**Воспроизведение обучения:**
```bash
# 1) Импорт датасета
uv run import-22022540 --data-root datasets/22022540 --output /tmp/ds_22022540
# 2) Обучение (в окружении .venv-full)
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.main train \
  --model unet --data-root /tmp/ds_22022540 --img-size 512 --epochs 200
# 3) best.onnx лежит в train/runs/unet_<ts>/checkpoints/best.onnx
# 4) (опционально) компактный single-file ONNX без внешних данных:
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.export_single \
  --checkpoint train/runs/unet_<ts>/checkpoints/best.pt \
  --output models/colony_seg.onnx --img-size 512
# 5) Инференс как алгоритм: тест покрыт tests/test_coco_importer.py + OnnxModelAlgorithm
```

---

## 3. Сборка (Nuitka)

### Локальная сборка (Linux)

```bash
bash scripts/build_nuitka.sh
# Результат: ./BacteriaAnalyzer (~85–125 МБ, standalone)
```

### Структура бинарника

- Включает пакеты: `analysis`, `ui`, `utils`, `labeling`, `testing`
- Исключены: `train`, `.venv`, `.venv-dev`, `.venv-full`, `.venv-build`, `test_images/`
- Включён плагин PyQt6
- Размер бинарника зависит от чистоты окружения сборки: сборка выполняется из изолированного `.venv-build` без dev/ML-пакетов

### Минимальное окружение сборки

Сборка всегда идёт из изолированного build-окружения **`.venv-build`** — как локально, так и в CI. Оно создаётся ad-hoc и содержит **только**:
- runtime-зависимости: `PyQt6`, `opencv-python-headless`, `numpy`;
- инструменты сборки: `nuitka`, `zstandard`.

`.venv` / `.venv-dev` / `.venv-full` при сборке не изменяются и не загрязняются. `.venv-build` исключён из `.gitignore` и из флагов Nuitka (`scripts/nuitka_flags.py`). Запуск Nuitka использует `--no-sync`, чтобы `uv run` не удалил вручную установленные `nuitka`/`zstandard`.

### CI (GitHub Actions)

Файл `.github/workflows/build.yaml`:

- **Триггеры:** push в `develop`/`main`/`feature/*`, PR в `develop`
- **Матрица:** ubuntu-latest + windows-latest
- **Python:** 3.13 (через `uv`)
- **Windows:** MSVC, отключена консоль, иконка `icon.ico`
- **Артефакты:** загружаются в GitHub Actions

---

## 4. Окружения

| Окружение | Команда | Размер | Назначение |
|---|---|---|---|
| `.venv` (runtime) | `uv sync` | ~475 МБ | Запуск приложения. Содержит только PyQt6, opencv-python-headless, numpy |
| `.venv-dev` (dev) | `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev` | ~530 МБ | Разработка и тестирование без ML: pytest + nuitka + zstandard |
| `.venv-full` (full) | `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full` | ~5 ГБ | Полная разработка, обучение U-Net, аугментация (включает torch, onnxruntime) |
| `.venv-build` (build, ad-hoc) | `uv sync` без extras + `uv pip install --python .venv-build nuitka zstandard` | ~0,5 ГБ | Сборка бинарника Nuitka. Создаётся автоматически `scripts/build_nuitka.sh` и CI; вручную не требуется |

**Важно:** после обновления ветки с новой структурой окружений старые `.venv`/`.venv-dev`/`.venv-full` нужно удалить и пересоздать заново. Если в `.venv` вручную были доустановлены пакеты (nuitka, pytest и т.д.), они будут удалены при `uv sync`. Сборка бинарника изолирована в `.venv-build` и не загрязняет `.venv`.

---

## 5. Архитектура проекта

| Путь | Назначение |
|---|---|
| `main.py` | Точка входа: QApplication, тёмная тема (Catppuccin Mocha), MainWindow |
| `ui/main_window.py` | Главное окно: выбор файла, кнопки «Анализировать» и «Разметка» |
| `ui/analysis_window.py` | Окно анализа: 4 режима просмотра, слайдеры, кнопка «Пересчитать» |
| `ui/testing_window.py` | Окно тестирования алгоритмов: выбор датасета, чекбоксы алгоритмов, загрузка моделей, QThread-прогон, парное сравнение, экспорт |
| `ui/styles.py` | Catppuccin Mocha QSS-стили |
| `analysis/image_processor.py` | Предобработка: CLAHE, зелёный канал, медианный blur |
| `analysis/colony_detector.py` | Детекция чашки и колоний |
| `utils/calculations.py` | Расчёт площади, покрытия, конвертация px→mm² |
| `labeling/labeling_window.py` | Разметка: кисть/ластик, зум, авто-поиск, обрезка, ZIP-экспорт |
| `testing/` | Фреймворк тестирования алгоритмов |
| `testing/__main__.py` | CLI `test-algorithms` |
| `testing/interface.py` | ABC для алгоритмов детекции |
| `testing/registry.py` | Декоратор `@register_algorithm` + `register_algorithm_instance` |
| `testing/classic_algorithms.py` | 4 варианта классического алгоритма |
| `testing/onnx_algorithm.py` | ONNX-адаптер (`OnnxModelAlgorithm`), сканирование встроенных моделей `models/*.onnx` |
| `testing/dataset.py` | Загрузчик тестовых пар (изображение + GT-маска) |
| `testing/metrics.py` | IoU, Dice, F1, Precision, Recall |
| `testing/runner.py` | Прогон алгоритмов и сбор метрик |
| `testing/dashboard.py` | HTML-отчёт с Chart.js |
| `models/` | Встроенные ONNX-модели (`*.onnx`), регистрируются при запуске |
| `train/` | Модуль обучения U-Net |
| `train/augment.py` | Аугментация (15× на оригинал, albumentations) |
| `train/config.py` | Гиперпараметры |
| `train/dataset.py` | PyTorch Dataset с train/val split |
| `train/train.py` | Цикл обучения с early stopping |
| `train/predict.py` | Инференс через ONNX Runtime |
| `train/reporter.py` | Plotly HTML-отчёт |
| `train/models/base.py` | BaseSegmenter (BCE+Dice loss, ONNX-экспорт) |
| `train/models/unet.py` | U-Net архитектура |
| `train/dashboard/server.py` | FastAPI + WebSocket дашборд |
| `train/dashboard/templates/index.html` | Chart.js фронтенд |
| `scripts/nuitka_build.py` | Программный запуск Nuitka |
| `scripts/nuitka_flags.py` | Генерация `--include-package` флагов |
| `scripts/build_nuitka.sh` | Пользовательский скрипт сборки |
| `.github/workflows/build.yaml` | CI: Nuitka на push/PR |

---

## 6. Конвенции

- Комментарии в коде — на русском.
- UI — на русском языке.
- Модели подключаются через `@register_model` (декоратор).
- Алгоритмы подключаются через `@register_algorithm` (декоратор).
- `.onnx` файлы отслеживаются в git, `.pt` — игнорируются.
- Три виртуальных окружения: `.venv` (runtime), `.venv-dev` (dev без torch), `.venv-full` (полная разработка с NN).

---

## 7. Что НЕ настроено

- Нет тестов (unit/integration)
- Нет линтера, typechecker, форматтера
- Нет pre-commit хуков
- CI не проверяет качество кода
