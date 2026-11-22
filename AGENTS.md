# Bacterial Colony Area Analyzer

Python 3.13+ / PyQt6 / OpenCV / NumPy desktop app for analysing bacterial colonies on Petri dish images.

---

## 1. Быстрый старт

### Из исходников (разработка)

```bash
# Runtime-окружение для приложения и ONNX Runtime CPU (без PyTorch)
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

Пользовательский анализ автоматически выполняется только встроенной ONNX-моделью
`NN:colony_seg`; выбор алгоритма и параметры детекции недоступны. Начальную
геометрию авто-поиска можно скорректировать, перемещая круг и меняя радиус.
Загрузка, авто-поиск чашки и анализ выполняются в фоне; окно остаётся
отзывчивым и показывает состояние/прогресс. Если модель или авто-поиск недоступны,
показывается понятная ошибка без переключения на иной алгоритм. Переключение
вида/сохранение больших кадров также не должно блокировать GUI.
Обучение MobileNet выполняется на CUDA, если доступна GPU; ONNX-инференс приложения
остаётся CPU-only через ONNX Runtime `CPUExecutionProvider`.

**Как использовать:**

1. Нажмите «📂 Открыть» и выберите изображение чашки Петри (поддерживаются PNG, JPG, BMP, TIFF, WebP).
2. Нажмите «🔍 Анализировать».
3. Откроется окно анализа; приложение автоматически найдёт чашку и запустит `NN:colony_seg`.

При отмене или закрытии окна выполняющаяся операция прекращает следующие этапы на ближайшей безопасной точке; неотменяемый OpenCV/ONNX native-вызов может завершиться естественно. Окно показывает состояние завершения и не блокирует обработку GUI-событий.

**Настройки распознавания:** пользовательские параметры алгоритма отсутствуют; анализ
выполняется с фиксированными настройками `NN:colony_seg`.

**Режимы просмотра** (выпадающий список над изображением):

| Режим | Что показывает |
|---|---|
| Результат | Оригинал + контуры найденных колоний |
| Оригинал | Исходное изображение |
| Предобработка | Для фиксированной NN-модели показывает оригинал, отдельная classic-предобработка не создаётся |
| Бинарная маска | Бинарная маска `NN:colony_seg` |

**Действия:**

- «💾 Сохранить» — сохраняет отображаемый результат.
- «🔄 Пересчитать» — повторно запускает фиксированную модель после успешного
  авто-поиска чашки. При неудаче поиска пользователю предлагается проверить снимок.

**Процесс анализа:** загрузка изображения → авто-поиск чашки Петри → инференс
`NN:colony_seg` → расчёт маски/метрик → отображение и сохранение результата.

Алгоритмы распознавания, параметры и ручная геометрия в основном пользовательском
GUI недоступны. Classic-алгоритмы остаются только во внутреннем testing/CLI инструментарии.

### 2.2. Разметка изображений (Labeling)

**Назначение:** ручное создание ground-truth масок для обучения и тестирования алгоритмов.

Список файлов, чтение/декодирование, авто-поиск чашки, обрезка, копирование,
сохранение маски и экспорт ZIP выполняются в фоне с состоянием операции. При
смене выделения устаревший результат не применяется; рисование по большому
изображению обновляет только изменённую область. Отмена и закрытие активной
операции кооперативны: уже выполняемый native-вызов может завершиться до
безопасной точки, при этом GUI остаётся отзывчивым.

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

### 2.3. Тестирование алгоритмов (внутренний инструмент)

**Назначение:** внутренняя разработка и сравнение алгоритмов детекции на эталонных
изображениях. Это не пользовательская функция, и главное окно приложения не
предоставляет входа в testing GUI. Разработчик может запускать внутренний модуль
GUI отдельно или использовать CLI.

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
- Внешние веса: файл `.onnx` подключается для внутренней разработки через GUI testing module (не доступен из главного окна) или `--model` в CLI — без пересборки приложения.
- Встроенные веса: файлы `models/*.onnx` упаковываются в бинарник Nuitka (`--include-data-files=models/*.onnx=models/`) и регистрируются автоматически при запуске.
- `onnxruntime` импортируется лениво (в `detect()`); без него пользовательский анализ показывает понятную ошибку и не выбирает замену.

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

### 2.5. Обучение U-Net (CLI, требуется full-окружение)

**Назначение:** обучение нейросетевой сегментации колоний.

```bash
# Установка full-окружения с PyTorch/обучающими зависимостями
UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full

# Запуск обучения
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.main train --model unet --epochs 200 --dashboard
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

**Статус импортёров и расширений:**

- **COCO 22022540.** `CocoBboxImporter` реализован для текущего источника; отдельная
  команда `prepare-colony-training` выполняет preflight, материализует маски, split и
  black-background варианты. Воспроизводимый LFS/training workflow описан в разделе 2.7.
  Универсальный registry для произвольных COCO/VOC/пользовательских источников не реализован.
- **Multiclass-маски.** `mask_mode` в манифесте резервирует значение
  `multiclass`. Сейчас конвейер (loss BCE+Dice, бинаризация порогом 0.5)
  поддерживает только `binary` и бросает понятную ошибку при попытке обучать
  multiclass. Точки расширения помечены комментариями в `train/dataset_manifest.py`.

### 2.7. COCO 22022540: подготовленный snapshot и Git LFS

`CocoBboxImporter` (`train/dataset_adapters.py`) растеризует COCO bbox-в разметку
во вписанные эллипсы (255 = колония); модель сегментирует наличие колонии, а не
класс бактерии. Для обучения/передачи используйте отдельный материализующий
этап `prepare-colony-training`, а не запускайте importer внутри `train-colony`.
Локальная версия источника: 369 размеченных изображений, 56 865 боксов, 24 вида.
Проверенная подготовка на seed 42: 369 source + 369 black-background вариантов,
738 парных записей; source split train/val/test = 253/62/54.

**Подготовка и упаковка (полное окружение, никаких Git-изменений):**

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync prepare-colony-training \
  --data-root datasets/22022540 --output datasets/work/22022540-training-webp \
  --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --black-background
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync pack-colony-training \
  --data-root datasets/work/22022540-training-webp \
  --archive datasets/archives/22022540-training-webp.tar.zst
```

`prepare-colony-training` проверяет наличие/декодируемость/размер всех размеченных
файлов, копирует исходные изображения байт-в-байт и записывает маски с
полноразмерными black-background вариантами в lossless WebP (с проверкой точных пикселей),
а также `dataset.json` и `split.json`.
`pack-colony-training` создаёт детерминированный lossless `.tar.zst` на уровне 19,
полностью проверяет round-trip SHA-256 и пишет метрики размера/сжатия. Рядом с
архивом появляются обычные Git sidecar-файлы `.dataset.json`, `.split.json`,
`.annot_COCO.json`, `.archive.json`. Каталог `datasets/work/` игнорируется.

**Получение на новой машине (после явной публикации архива владельцем):**

Для Fedora: `sudo dnf install git-lfs`; для других ОС — [официальная инструкция по установке Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Затем выполните `git lfs install` для своего пользователя.

```bash
git lfs install                 # один раз на машине
git clone https://github.com/Egel68/bacterial-colony-area.git training-checkout
cd training-checkout
git lfs pull
git lfs ls-files
test -s datasets/archives/22022540-training-webp.tar.zst
UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync unpack-colony-training \
  --archive datasets/archives/22022540-training-webp.tar.zst \
  --output datasets/work/22022540-training-webp
```

`unpack-colony-training` распаковывает в новый каталог и проверяет архив, sidecars,
манифест, split и пары. Без `git lfs pull` в checkout может оставаться только LFS
pointer; см. [установку Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
и [получение больших файлов](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage).

**Обучение только из полного локального snapshot (CUDA обязательна для `unet`):**

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync train-colony \
  --data-root datasets/work/22022540-training-webp \
  --output train/runs/22022540_black_background_seed42_your-new-run \
  --model unet --black-background --no-pretrained --seed 42 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --epochs 100 --batch-size 2 --patches-per-image 4 --img-size 512
```

`train-colony` требует `dataset.json`, `split.json` и все пары до baseline/optimizer;
он не импортирует COCO и не записывает images/masks. `--output` должен быть отдельным
новым/пустым каталогом. Патчи/случайные аугментации остаются in-memory. Для данных
без black-background при подготовке укажите `--no-black-background`, а при обучении
не передавайте `--black-background`. Отчёты/checkpoints пишутся в
`<output>/runs/unet_<ts>/`; `models/colony_seg.onnx` не меняется автоматически.
Полное руководство и все флаги CLI: [docs/dataset-training-pipeline.md](docs/dataset-training-pipeline.md).

**Лицензия/квоты/ручная публикация.** Figshare для версии 3 указывает CC BY 4.0
([DOI 10.6084/m9.figshare.22022540.v3](https://doi.org/10.6084/m9.figshare.22022540.v3));
сохраняйте атрибуцию Makrai et al. и ссылку на лицензию, локальная сверка хешей с
Figshare не выполнялась. По [GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
(проверено 2026-11-08) квоты Free/Pro — 10 GiB storage и download bandwidth;
Team/Enterprise Cloud — по 250 GiB. Максимум одного LFS-файла — 2/2/4/5 GB для
Free/Pro/Team/Enterprise Cloud ([лимиты](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage));
перед отправкой сравните фактический размер из `.archive.json` с тарифом и текущим
budget. В LFS новые версии архива учитывают полный размер, скачивания расходуют
bandwidth.

Публикует только владелец после ручной проверки лицензии, размера и видимости репо:

```bash
git check-attr filter -- datasets/archives/22022540-training-webp.tar.zst
git add .gitattributes datasets/archives/22022540-training-webp.tar.zst \
  datasets/archives/22022540-training-webp.*.json
git lfs status
git diff --cached --stat
git commit -m "Add 22022540 training archive"
git push origin HEAD
```

`.tar.zst` входит в LFS, sidecars `.json` остаются обычными Git-файлами. Эти
команды только документированы; подготовка/упаковка не выполняет add/commit/push.

**Текущая production-модель** — `models/colony_seg.onnx`: scratch U-Net, обученная
на парах 22022540 (seed 42, source split 70/15/15) на CUDA/NVIDIA GeForce RTX 2060
with Max-Q Design. `NN:colony_seg` использует CPU, тайлы 512×512, stride 384, sigmoid
для logits, усреднение вероятностей перекрытий, порог 0.5. На paired holdout из 54
source ID: исходный фон — IoU 0.4812, Dice 0.6497, Precision 0.8340, Recall 0.5322;
чёрный — IoU 0.4721, Dice 0.6414, Precision 0.8375, Recall 0.5197. Пересечение
test-ID с историческим baseline train неизвестно (`unknown`). Report:
`train/runs/22022540_black_background_seed42/runs/unet_20261107_171657_193829/comparison.json`.

---

## 3. Сборка (Nuitka)

### Локальная сборка (Linux)

```bash
bash scripts/build_nuitka.sh
# Результат: ./BacteriaAnalyzer (размер зависит от runtime-зависимостей и ONNX-моделей)
```

### Структура бинарника

- Включает пакеты: `analysis`, `ui`, `utils`, `labeling`, `testing`
- Исключены: `train`, `.venv`, `.venv-dev`, `.venv-full`, `.venv-build`, `test_images/`
- Включён плагин PyQt6
- Размер бинарника зависит от чистоты окружения сборки: сборка выполняется из изолированного `.venv-build` без dev/ML-пакетов
- Проверенная Linux onefile-сборка 2026-09-27 с ONNX Runtime 1.27.0 и двумя bundled-моделями: 170,221,056 bytes (162.3 MiB; контрольный диапазон этой конфигурации 155–175 MiB). Standalone executable: 61,191,088 bytes; каталог standalone: около 499 MiB.

### Минимальное окружение сборки

Сборка всегда идёт из изолированного build-окружения **`.venv-build`** — как локально, так и в CI. Оно создаётся ad-hoc и содержит **только**:
- runtime-зависимости: `PyQt6`, `opencv-python-headless`, `numpy`, `onnxruntime` CPU;
- инструменты сборки: `nuitka`, `zstandard`.

`.venv` / `.venv-dev` / `.venv-full` при сборке не изменяются и не загрязняются. `.venv-build` исключён из `.gitignore` и из флагов Nuitka (`scripts/nuitka_flags.py`). Запуск Nuitka использует `--no-sync`, чтобы `uv run` не удалил вручную установленные `nuitka`/`zstandard`.

### CI (GitHub Actions)

Файл `.github/workflows/build.yaml`:

- **Триггеры:** PR (`opened`, `synchronize`, `reopened`) в `develop`, push в `main`/`master`, ручной `workflow_dispatch`. Push в `develop` и feature-ветки самостоятельно workflow не запускает.
- **Матрица:** ubuntu-latest + windows-latest; отдельный `required-ci` контролирует обязательность матрицы.
- **Python:** 3.13 через `uv`.
- **Проверки:** Ruff lint/format, тесты без slow/gui на обоих runners, GUI worker tests offscreen на Windows, Nuitka build и CPU inference smoke checks.
- **Артефакты:** бинарники для платформ; Linux JUnit pytest report. Изменения только под `openspec/` пропускают build matrix.

---

## 4. Окружения

| Окружение | Команда | Размер | Назначение |
|---|---|---|---|
| `.venv` (runtime) | `uv sync` | зависит от платформы | Запуск приложения и ONNX Runtime CPU; без PyTorch |
| `.venv-dev` (dev) | `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev` | зависит от платформы | Тесты и инструменты сборки; включает ONNX Runtime CPU, без PyTorch |
| `.venv-full` (full) | `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full` | зависит от платформы | Обучение MobileNet/U-Net; включает torch/torchvision и ONNX Runtime |
| `.venv-build` (build, ad-hoc) | `uv sync` без extras + `uv pip install --python .venv-build nuitka zstandard` | зависит от платформы | Nuitka-сборка с ONNX Runtime CPU, без PyTorch; создаётся скриптом/CI |

**Важно:** после обновления ветки с новой структурой окружений старые `.venv`/`.venv-dev`/`.venv-full` нужно удалить и пересоздать заново. Если в `.venv` вручную были доустановлены пакеты (nuitka, pytest и т.д.), они будут удалены при `uv sync`. Сборка бинарника изолирована в `.venv-build` и не загрязняет `.venv`.

---

## 5. Архитектура проекта

| Путь | Назначение |
|---|---|
| `main.py` | Точка входа: QApplication, тёмная тема (Catppuccin Mocha), MainWindow |
| `ui/main_window.py` | Главное окно: выбор файла, кнопки «Анализировать» и «Разметка» |
| `ui/analysis_window.py` | Окно анализа: фиксированная `NN:colony_seg`, режимы просмотра, пересчёт и сохранение |
| `ui/testing_window.py` | Внутреннее GUI тестирования: выбор датасета/алгоритмов, загрузка моделей, прогон и экспорт; не открывается из главного окна |
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
| `testing/onnx_algorithm.py` | ONNX-адаптеры и выбор tiled/resize для bundled-моделей `models/*.onnx` |
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

## 7. Проверки и ограничения разработки

- В репозитории есть unit, integration и GUI-тесты под pytest; часть тестов помечена `slow`/`gui` и не входит в обычный CI-набор.
- CI проверяет Ruff lint и форматирование, запускает тесты `not slow and not gui` на Linux/Windows и отдельные GUI worker tests offscreen на Windows, затем собирает бинарники и выполняет CPU inference smoke checks.
- Обязательного typechecker и pre-commit hooks сейчас нет.
