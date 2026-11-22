# 🔬 Bacteria Colony Analyzer

Десктопное приложение на Python для анализа фотографий чашек Петри: автоматический
подсчёт бактериальных колоний, ручная разметка эталонных масок, сравнение алгоритмов
детекции по метрикам и обучение нейросетевой сегментации.

Проект решает задачу, которая в лабораторной практике считается ручной: по одному фото
чашки получить число колоний, покрытие и площадь, а затем — воспроизводимую оценку
качества алгоритма на размеченных данных.

| | |
|---|---|
| **Интерфейс** | PyQt6, тёмная тема Catppuccin Mocha, русский язык |
| **Классические алгоритмы** | OpenCV: CLAHE, адаптивная бинаризация, морфология, connected components |
| **Нейросетевые алгоритмы** | ONNX Runtime (CPU-only), встроенная MobileNetV3-Small с тайловым inference |
| **Обучение** | PyTorch, U-Net / MobileNetV3-Small-UNet, TensorBoard + веб-дашборд |
| **Сборка** | Nuitka → один исполняемый файл для Linux и Windows (~162 MiB) |
| **Версии** | Python 3.13+, [`uv`](https://docs.astral.sh/uv/) |

---

## 📑 Содержание

- [Возможности](#-возможности)
- [Быстрый старт](#-быстрый-старт)
- [Виртуальные окружения](#-виртуальные-окружения)
- [Консольные команды (CLI)](#-консольные-команды-cli)
  - [Сводная таблица](#сводная-таблица)
  - [test-algorithms](#test-algorithms--режим-report)
  - [baseline](#baseline--json-метрики-классических-алгоритмов)
  - [evaluate](#evaluate--пакетная-оценка-с-каталогом-отчёта)
  - [import-22022540](#import-22022540--импорт-coco-датасета)
  - [augment](#augment--аугментация-обучающей-выборки)
  - [train.main](#trainmain--обучение-u-net-и-сравнение-архитектур)
  - [train-colony](#train-colony--пайплайн-coco--mobilenet)
  - [export_single](#export_single--экспорт-checkpoint-в-onnx)
  - [bacteria-analyzer](#bacteria-analyzer--запуск-gui-и-smoke-test)
- [Форматы данных](#-форматы-данных)
- [Алгоритмы детекции](#-алгоритмы-детекции)
- [Метрики качества](#-метрики-качества)
- [Сборка бинарника](#-сборка-бинарника)
- [Тесты](#-тесты)
- [Структура проекта](#-структура-проекта)
- [Документация](#-документация)
- [Ограничения](#-ограничения)

---

## ✨ Возможности

**Анализ колоний.** Автопоиск чашки Петри (отражения → окружность), автоматический
расчёт маски и числа колоний, покрытия рабочей зоны и площади в пикселях. Пользовательский
анализ всегда использует встроенную `NN:colony_seg`; параметры модели недоступны,
но найденную круглую область можно перемещать и менять её радиус. После коррекции
нужен явный пересчёт. Загрузка и инференс выполняются в фоне с сохранением отзывчивости.

**Внутренняя валидация.** Классические алгоритмы, дополнительные NN-модели, GUI/CLI
сравнение и загрузка внешних весов остаются для разработки и проверки качества, но не
являются пользовательскими действиями обычного приложения.

**Разметка (labeling).** Полноценный инструмент рисования масок: кисть, ластик, зум,
обрезка по чашке, сохранение в структурированную сессию и экспорт в ZIP. Длительные
операции выполняются в фоне; рисование обновляет локальную область изображения.

**Внутреннее сравнение алгоритмов.** Разработчикам доступны GUI-модуль и CLI-прогон
по датасету с расчётом IoU, Dice, F1, Precision, Recall, Accuracy; попарное сравнение,
кэш предсказаний, telemetry и HTML/JSON-отчёты. Testing GUI не открывается из главного
пользовательского окна.

**Обучение.** Импорт внешнего COCO-bbox датасета, аугментация, обучение U-Net и
MobileNetV3-Small с воспроизводимым split, проверкой ONNX-паритета, CPU-бенчмарком и
held-out оценкой.

---

## 🚀 Быстрый старт

### Для пользователя: готовый бинарник

Скачайте последний релиз со страницы [Releases](https://github.com/Egel68/bacterial-colony-area/releases):

| Платформа | Файл |
|---|---|
| Linux | `BacteriaAnalyzer` |
| Windows | `BacteriaAnalyzer.exe` |

Запустите файл — Python и библиотеки устанавливать не нужно. Встроенные ONNX-модели
уже внутри.

### Для разработчика: из исходников

Нужен Python 3.13+ и [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Egel68/bacterial-colony-area.git
cd bacterial-colony-area

# Runtime-окружение: пользовательский GUI с фиксированной NN-моделью + CPU-инференс (без PyTorch)
uv sync
uv run bacteria-analyzer
```

### Демо-датасет

В репозитории лежит небольшой размеченный набор для внутреннего тестирования
алгоритмов (не пользовательская функция приложения):

```bash
# все зарегистрированные классические алгоритмы на test_images/
uv run test-algorithms

# выборочно, с внешней моделью
uv run test-algorithms \
  --algorithms ClassicDefault,ClassicSolidFill \
  --model ./models/colony_mobilenet_v3_small.onnx
```

> Встроенные модели из `models/` регистрируются только при запуске GUI-приложения.
> Чтобы прогнать их из CLI, передайте файл через `--model`.

---

## 🧪 Виртуальные окружения

Зависимости разделены на несколько окружений, чтобы обычный запуск приложения не тянул
PyTorch и тяжёлые ML-пакеты.

| Окружение | Назначение | Создание |
|---|---|---|
| `.venv` | Пользовательский GUI с фиксированной `NN:colony_seg`, внутренние CLI-инструменты | `uv sync` |
| `.venv-dev` | Тесты и инструменты сборки, **без** PyTorch | `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev` |
| `.venv-full` | Обучение моделей и аугментация (**с** PyTorch) | `UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full` |
| `.venv-build` | Изолированная сборка бинарника | создаётся `scripts/build_nuitka.sh` или CI |

Пример для `.venv-full` в PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-full"
uv sync --extra full
```

> **Важно.** Не доустанавливайте вручную ML-пакеты в `.venv` — после `uv sync` они
> будут удалены. Выберите нужное окружение. `--extra full` тянет ~5 ГБ (torch + CUDA).
>
> Примеры ниже используют `uv run <команда>` для runtime-команд и
> `UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync <команда>` для обучения.
> Флаг `--no-sync` нужен, чтобы `uv` не пересобрал окружение перед запуском.

---

## 🖥 Консольные команды (CLI)

### Сводная таблица

| Команда | Что делает | Окружение |
|---|---|---|
| `uv run bacteria-analyzer` | Запуск GUI-приложения | `.venv` |
| `uv run bacteria-analyzer --smoke-test-model <имя>` | Диагностика встроенной модели без GUI | `.venv` |
| `uv run test-algorithms` | HTML-отчёт по всем зарегистрированным алгоритмам | `.venv` |
| `uv run baseline` | JSON-метрики четырёх классических алгоритмов | `.venv` |
| `uv run evaluate` | Пакетная оценка с каталогом `evaluations/<дата-время>/` | `.venv` |
| `uv run import-22022540` | Импорт COCO-bbox датасета в формат проекта | `.venv` |
| `uv run augment` | Аугментация: 15 вариантов каждой пары | `.venv-full` |
| `uv run train-colony` | Полный pipeline COCO → MobileNet → ONNX → оценка | `.venv-full` |
| `python -m train.main train` | Обучение U-Net / MobileNet | `.venv-full` |
| `python -m train.main train-compare` | Обучение нескольких архитектур и сравнение | `.venv-full` |
| `python -m train.main list-models` | Список доступных архитектур | `.venv-full` |
| `python -m train.main dataset-info` | Информация о датасете из `--data-root` | `.venv-full` |
| `python -m train.export_single` | Повторный экспорт `best.pt` в один `.onnx` | `.venv-full` |

Актуальный список флагов любой команды — через `--help`.

---

### `test-algorithms` — режим `report`

HTML-отчёт с таблицами и графиками Chart.js. Режим `report` используется по умолчанию.

```bash
uv run test-algorithms \
  --data-root ./test_images \
  --output ./test_report.html \
  --algorithms ClassicDefault,ClassicSolidFill \
  --model ./weights/another-model.onnx \
  --compare ClassicDefault,ClassicSolidFill \
  --stats-output ./test_report.stats.json \
  --no-per-snapshot
```

| Флаг | Назначение |
|---|---|
| `--data-root DIR` | Корень датасета; по умолчанию `test_images` |
| `--output FILE` | Путь к HTML-отчёту |
| `--algorithms A,B` | Список алгоритмов через запятую; по умолчанию все зарегистрированные |
| `--model FILE.onnx` | Зарегистрировать внешнюю ONNX-модель; флаг повторяемый |
| `--compare A,B` | Добавить секцию попарного сравнения двух алгоритмов |
| `--per-snapshot` / `--no-per-snapshot` | Таблицы подробных метрик по снимкам (по умолчанию включены) |
| `--stats` / `--no-stats` | Расширенная статистика (по умолчанию включена) |
| `--stats-output FILE.json` | Дополнительно выгрузить полную статистику в JSON |
| `--workers N` | Число параллельных worker-ов; по умолчанию выбирается автоматически |
| `--batch-size N` | Размер пакета пар; по умолчанию 8 |
| `--memory-budget BYTES` | Ориентир по памяти для пакетной загрузки |
| `--sample-limit N` | Ограничить число исходных снимков (cropped-пара тянется вместе) |
| `--cache` / `--no-cache` | Кэш масок в `<data-root>/.cache/preds/` (по умолчанию выключен) |
| `--telemetry` / `--no-telemetry` | Структурированная telemetry (по умолчанию выключена) |
| `--telemetry-interval SEC` | Интервал telemetry-снимков; по умолчанию 5 с |
| `--performance-output FILE` | JSON-сводка telemetry; события — в соседний `.jsonl` |

> CLI-команды регистрируют классические алгоритмы и модели, переданные через
> `--model`, но **не** сканируют папку `models/`. Автоматическая регистрация
> встроенных моделей (`NN:<stem>`) выполняется только при запуске GUI-приложения —
> в `main.py` вызывается `register_bundled_models()`. Чтобы прогнать встроенную
> модель из CLI, укажите её файл через `--model`.
>
> Графики Chart.js в HTML подгружаются с CDN — для полностью офлайн-отчёта их нужно
> положить рядом с файлом.

---

### `baseline` — JSON-метрики классических алгоритмов

Всегда прогоняет четыре встроенных классических алгоритма. Флаги `--algorithms` и
`--model` к этому режиму не применяются.

```bash
uv run baseline --data-root ./test_images \
  --output ./classic-baseline.json \
  --no-cropped
```

Указывайте расширение `.json`: общий парсер по умолчанию называет файл
`test_report.html`, даже если содержимое будет JSON. Также доступны `--workers`,
`--batch-size`, `--memory-budget`, `--sample-limit`, `--cache`, `--clear-cache`,
`--telemetry`, `--telemetry-interval`, `--performance-output`.

---

### `evaluate` — пакетная оценка с каталогом отчёта

Фиксированный набор четырёх классических алгоритмов, результаты складываются в
`evaluations/<дата-время>/`:

```
evaluations/2026-09-27_18-30-00/
├── report.json      # метрики по алгоритмам и снимкам
├── report.html      # визуальный отчёт
├── config.json      # фактическая конфигурация прогона
└── run_info.json    # время, окружение, параметры запуска
```

```bash
uv run evaluate \
  --data-root ./test_images \
  --workers 4 --batch-size 8 \
  --telemetry --telemetry-interval 2
```

| Флаг | Назначение |
|---|---|
| `--data-root DIR` | Корень датасета (по умолчанию `test_images`) |
| `--no-cropped` | Отключить cropped-варианты |
| `--sample-limit N` | Ограничить число снимков |
| `--workers`, `--batch-size`, `--memory-budget` | Пакетная обработка |
| `--cache` | Кэш масок предсказаний |
| `--clear-cache` | Очистить агрегированный baseline-кэш для `--data-root` перед прогоном |
| `--config FILE` | Конфигурация в JSON (PyYAML для YAML) |
| `--import-root DIR` | Импортировать COCO-источник в `--data-root` перед оценкой |
| `--telemetry` / `--no-telemetry`, `--telemetry-interval`, `--performance-output` | Telemetry |

> ⚠️ **Осторожно с `--sample-limit` и кэшем.** После прогона с ограниченной выборкой
> агрегированный кэш перезаписывается метриками только этой выборки. Перед полным
> прогоном на том же датасете запустите команду с `--clear-cache`.
>
> В конфиге используются ключи с подчёркиванием (`workers`, `memory_budget`,
> `sample_limit`, `import_root`). Значения из конфига для `data_root`, `use_cropped`,
> `use_cache`, `batch_size`, `telemetry`, `telemetry_interval` перезаписываются
> значениями CLI, включая их значения по умолчанию — задавайте их флагами.
>
> Эквивалент: `uv run test-algorithms --mode evaluate ...`

---

### `import-22022540` — импорт COCO-датасета

Растрирует COCO bounding boxes во вписанные эллипсы бинарных масок и приводит внешний
датасет к контракту обучения (самодостаточная папка + `dataset.json`).

```bash
uv run import-22022540 \
  --data-root ./datasets/22022540 \
  --output ./datasets/22022540_imported \
  --crop --split --seed 42 --verify
```

| Флаг | Назначение |
|---|---|
| `--data-root DIR` | **Обязательно.** Папка COCO-источника (только чтение) |
| `--output DIR` | Каталог результата; по умолчанию `<data-root>_imported` |
| `--crop` | Дополнительно создать обрезки по найденной чашке (чёрный фон вне круга) |
| `--split` | Записать воспроизводимый стратифицированный train/val/test split |
| `--seed N` | Seed разбиения; по умолчанию 42 |
| `--verify` | Сверить число масок с боксами COCO и дублями YOLO/VOC |

> Исходный датасет не изменяется. Всегда выбирайте отдельный каталог `--output`:
> при `--split` совпадение путей проверяется автоматически, без `--split` — нет, и
> существующие файлы могут быть перезаписаны. Изображения декодируются и записываются
> заново, поэтому выходные файлы не являются побайтовыми копиями.
>
> Эллипсы внутри bbox — **weak labels**, а не ручная pixel-perfect разметка.

---

### `augment` — аугментация обучающей выборки

Команда без аргументов: читает `test_images/cropped/*.png` и
`test_images/cropped_masks/*_mask.png`, создаёт 15 вариантов каждой валидной пары.

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync augment
```

```text
train/data/images/   # геометрия + пиксельные аугментации
train/data/masks/    # маски, синхронно преобразованные с изображениями
```

Применяются повороты, отражения, аффинные преобразования, яркость/контраст, шум, blur,
hue/saturation. Изображения без парной маски пропускаются. Повторный запуск использует
те же базовые имена и может заменить ранее созданные файлы.

---

### `train.main` — обучение U-Net и сравнение архитектур

```bash
# Какие архитектуры доступны
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync python -m train.main list-models
# → mobilenet_v3_small_unet, unet, unet_small

# Обучение
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync python -m train.main train \
  --model unet --data-root ./train/data \
  --epochs 200 --batch-size 8 --img-size 512 --lr 1e-3 \
  --dashboard

# Обучение нескольких архитектур и их сравнение
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync python -m train.main train-compare \
  --architectures unet,unet_small \
  --data-root ./train/data --eval-root ./test_images \
  --epochs 50 --batch-size 8

# Информация о датасете
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync python -m train.main dataset-info
```

Флаги `train`:

| Флаг | По умолчанию | Назначение |
|---|---|---|
| `--model` | `unet` | Архитектура из `list-models` |
| `--epochs` | 200 | Максимум эпох, используется ранняя остановка |
| `--batch-size` | 8 | Размер batch |
| `--img-size` | 512 | Размер входа |
| `--lr` | 1e-3 | Learning rate |
| `--data-root` | `train/data` | Корень данных или манифест |
| `--dashboard` | выключен | Веб-дашборд (порт 8765) + TensorBoard (6006) |
| `--dashboard-port` | 8765 | Порт веб-дашборда |

`train-compare` дополнительно принимает `--architectures` и `--eval-root`. Результаты:

```text
train/runs/<model>_<timestamp>/
├── checkpoints/
│   ├── best.pt                # лучший checkpoint по IoU на валидации
│   ├── best.onnx
│   ├── last.pt
│   └── last.onnx
├── summary.json
├── report.html                # интерактивный отчёт (Plotly)
└── tensorboard/               # логи TensorBoard

train/runs/compare_<timestamp>/
├── compare.json
└── compare.html
```

> `--eval-root` в `train-compare` должен содержать пары, читаемые `TestDataset`, и не
> должен пересекаться с обучающими снимками. Сам режим held-out выборку не создаёт —
> для воспроизводимого split используйте `train-colony`.

---

### `train-colony` — обучение из подготовленного COCO snapshot

`prepare-colony-training`, упаковка, получение архива и обучение — отдельные шаги.
Обучающий pipeline принимает только полный распакованный каталог с `dataset.json`,
`split.json` и всеми файлами из манифеста; он проверяет данные до baseline/optimizer и
не создаёт/не перезаписывает изображения, маски или black-background варианты.

#### 1. Подготовить и упаковать (на машине с локальным COCO-источником)

```bash
# Полное окружение (однократно для этой копии репозитория)
UV_PROJECT_ENVIRONMENT=.venv-full uv sync --extra full

UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync prepare-colony-training \
  --data-root datasets/22022540 \
  --output datasets/work/22022540-training-webp \
  --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --black-background

UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync pack-colony-training \
  --data-root datasets/work/22022540-training-webp \
  --archive datasets/archives/22022540-training-webp.tar.zst
```

Подготовка копирует исходные изображения байт-в-байт, а полноразмерные
black-background варианты записывает в lossless WebP и проверяет точное совпадение
пикселей. Упаковщик выполняет детерминированный
Zstandard level 19, полный SHA-256 round-trip и печатает фактический размер/коэффициент
сжатия. Рядом с архивом он создаёт обычные Git
sidecar-файлы `22022540-training-webp.dataset.json`, `.split.json`, `.annot_COCO.json` и
`.archive.json` (метрики и SHA-256 архива). Эти команды не выполняют `git add`, commit
или push. `datasets/work/` игнорируется Git.

#### 2. Получить и распаковать на новой машине

Сначала установите Git LFS (Fedora: `sudo dnf install git-lfs`; для других ОС см. [официальную инструкцию](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)) и инициализируйте его. В новом checkout (после того, как архив явно опубликован владельцем):

```bash
git lfs install
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

`unpack-colony-training` требует новый/несуществующий `--output`, безопасно распаковывает
архив, проверяет его SHA-256, sidecar-метаданные и полноту manifest/split. Без Git LFS
в checkout может находиться только маленький pointer-файл, а не `.tar.zst`; `git lfs pull`
получает LFS-объект. См. [Git LFS collaboration](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage).

#### 3. Обучить из распакованного каталога

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync train-colony \
  --data-root datasets/work/22022540-training-webp \
  --output train/runs/22022540_black_background_seed42_your-new-run \
  --model unet --black-background --no-pretrained --seed 42 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --epochs 100 --batch-size 2 --patches-per-image 4 --img-size 512 \
  --cpu-benchmark
```

`--output` должен быть новым/пустым каталогом и находиться отдельно от snapshot. Для
варианта без чёрного фона подготовьте данные с `--no-black-background`, укажите модель
(по умолчанию `mobilenet_v3_small_unet`) и не передавайте `--black-background`. Для
парного black-background опыта требуется `--model unet`; обучение и обновление весов
выполняются на CUDA без CPU fallback. Checkpoint выбирается по validation, test остаётся
held-out; отчёты и `best.onnx` пишутся только в `<output>/runs/...`.

| Флаг | По умолчанию | Назначение |
|---|---|---|
| `--data-root` | обязателен | Полный распакованный snapshot (`dataset.json`, `split.json`, images/masks) |
| `--output` | обязателен | Отдельный корень результатов запуска |
| `--model` | `mobilenet_v3_small_unet` | `mobilenet_v3_small_unet` или `unet` |
| `--black-background` | выключен | Использовать подготовленные пары и парную оценку; требует `unet` |
| `--seed` | 42 | Seed split и обучения |
| `--train-ratio` / `--val-ratio` / `--test-ratio` | 0.70 / 0.15 / 0.15 | Доли, сумма должна быть 1 и совпадать с `split.json` |
| `--epochs` | 100 | Максимум эпох |
| `--batch-size` | 8 | Размер batch |
| `--patches-per-image` | 4 | Число patch-сэмплов на запись |
| `--img-size` | 512 | Поддерживается только 512 |
| `--no-pretrained` | выключен | Не использовать ImageNet weights |
| `--resume-run DIR` | — | Продолжить незавершённый run из `<output>/runs/` (несовместим с `--black-background`) |
| `--cpu-benchmark` | выключен | Полноразмерный tiled ONNX CPU benchmark |
| `--promote-model FILE.onnx` | — | Явно скопировать ONNX по отдельному пути |
| `--allow-overwrite` | выключен | Разрешить перезапись цели promotion |

Патчи 512×512 и случайные преобразования формируются в памяти. Исходный snapshot
проверяется до baseline/optimizer и не изменяется. `models/colony_seg.onnx` не
перезаписывается автоматически; сначала изучите `report.json`/`comparison.json`, затем
при необходимости вручную выберите отдельный promotion path.

#### Лицензия, квоты и явная публикация

Для `22022540` запись Figshare версии 3 указывает [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([версия набора и DOI](https://doi.org/10.6084/m9.figshare.22022540.v3)); сохраняйте атрибуцию и
ссылку на лицензию и перепроверьте условия/идентичность локальной копии перед
публикацией. По [актуальной справке GitHub LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
(проверено 2026-11-08) бесплатные квоты составляют 10 GiB storage и 10 GiB download
bandwidth для Free/Pro и 250 GiB/250 GiB для Team/Enterprise Cloud; тариф и budgets
аккаунта могут меняться. Отдельный лимит размера файла — 2/2/4/5 GB соответственно
([Git LFS file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)).
Проверьте размер из `.archive.json`, тариф, остаток квоты и видимость репозитория до
отправки; новые версии LFS-архива целиком занимают storage, скачивание расходует
bandwidth.

Первоначальную публикацию выполняет только владелец после этих проверок; инструменты
подготовки и распаковки Git не меняют. Явные команды для проверки/отправки:

```bash
git lfs install
git check-attr filter -- datasets/archives/22022540-training-webp.tar.zst
git add .gitattributes datasets/archives/22022540-training-webp.tar.zst \
  datasets/archives/22022540-training-webp.*.json
git lfs status
git diff --cached --stat
git commit -m "Add 22022540 training archive"
git push origin HEAD
```

`*.tar.zst` отправляется в LFS; `.json` sidecar-файлы остаются обычными Git-файлами.
Команды `git add`/commit/push приведены как ручная инструкция — здесь они не запускались.
Для текущих условий и возможных изменений квот используйте [GitHub billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

---

### `export_single` — экспорт checkpoint в ONNX

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run --no-sync python -m train.export_single \
  --checkpoint ./train/runs/unet_<timestamp>/checkpoints/best.pt \
  --output ./models/my_unet.onnx \
  --model unet --img-size 512
```

Архитектура и размер входа должны соответствовать checkpoint. Положите полученный
`.onnx` в `models/` — он попадёт в бинарник при сборке и зарегистрируется автоматически
при запуске приложения.

---

### `bacteria-analyzer` — запуск GUI и smoke-test

```bash
uv run bacteria-analyzer
```

Диагностика встроенной модели без открытия окон (проверяет бинарность маски и то, что
ONNX Runtime использует именно `CPUExecutionProvider`):

```bash
./BacteriaAnalyzer --smoke-test-model NN:colony_mobilenet_v3_small
# {"model": "...", "providers": ["CPUExecutionProvider"], "output_values": [0, 255], ...}
```

Доступные внутреннему model smoke test имена: `NN:colony_seg`,
`NN:colony_mobilenet_v3_small`. Пользовательское окно анализа использует только
`NN:colony_seg`.

---

## 📁 Форматы данных

### Датасет для тестирования и разметки

```text
<root>/
├── source/                             исходные снимки
├── masks/                              маски исходников  (<stem>_mask.png)
├── cropped/                            обрезки по чашке  (<stem>_cropped.png)
└── cropped_masks/                      маски обрезков     (<stem>_cropped_mask.png)
```

Разметчик (labeling) создаёт эту структуру сам в `~/BacteriaLabeling/<сессия>/`;
выбранный путь и список недавних сессий сохраняются между запусками. Сессию можно
экспортировать в ZIP одной кнопкой и импортировать в репозиторий:

```bash
unzip session.zip
cp -r session/source/*        test_images/source/
cp -r session/masks/*         test_images/masks/
cp -r session/cropped/*       test_images/cropped/
cp -r session/cropped_masks/* test_images/cropped_masks/
```

### Датасет для обучения (манифест)

Обучение не знает про конкретные папки — оно читает список пар «изображение + маска».
`load_manifest()` выбирает источник по приоритету:

| Приоритет | Источник | Что читает |
|---|---|---|
| 1 | `ManifestAdapter` | `dataset.json` в корне датасета |
| 2 | `LabelingAdapter` | `source/` + `masks/`, `cropped/` + `cropped_masks/` (по ссылкам) |
| 3 | `PairsAdapter` | легаси `images/` + `masks/` по совпадающим именам |

`dataset.json` описывает пары, режим масок (`binary`), subset (`train`/`val`/`test`) и
способ хранения (`copy` / `reference`). Такой файл создаёт, например,
`import-22022540 --split`.

---

## 🧠 Алгоритмы детекции

Единый интерфейс: `BaseDetectionAlgorithm` с полями `name`, `description` и методом
`detect(image, is_cropped) -> uint8 mask`.

| Имя | Тип | Профиль |
|---|---|---|
| `ClassicDefault` | Классический | sensitivity 0.5, отступ 10%, мин. размер 50 px |
| `ClassicHighSensitivity` | Классический | sensitivity 0.7, отступ 5%, мин. размер 30 px |
| `ClassicLowSensitivity` | Классический | sensitivity 0.3, отступ 10%, мин. размер 100 px |
| `ClassicSolidFill` | Классический | sensitivity 0.3, отступ 15%, solid fill |
| `NN:colony_seg` | ONNX | U-Net, тайлы 512×512 со шагом 384 px, усреднение вероятностей перекрытий |
| `NN:colony_mobilenet_v3_small` | ONNX | MobileNetV3-Small, тайлы 512×512 со шагом 384 px, усреднение перекрытий |

Внутреннее расширение и валидация моделей:

- **без пересборки** — флаг `--model` в CLI `test-algorithms`; имя в реестре берётся
  из имени файла. GUI testing module доступен разработчику отдельно, не из главного окна;
- **в составе приложения** — положите `.onnx` в папку `models/`: при сборке Nuitka каталог
  включается в бинарник (`--include-data-files=models/*.onnx=models/`), а runtime может
  зарегистрировать модели для внутренней диагностики. Обычный анализ по-прежнему
  использует исключительно `NN:colony_seg`.

Модель определяет **наличие колонии**, а не вид бактерии. Пайплайн обучения и импорта
описан в [docs/dataset-training-pipeline.md](docs/dataset-training-pipeline.md).

---

## 📊 Метрики качества

Предсказание каждого алгоритма попиксельно сравнивается с эталонной бинарной маской:

| Метрика | Формула |
|---|---|
| IoU (Jaccard) | `TP / (TP + FP + FN)` |
| Dice (F1-score) | `2·TP / (2·TP + FP + FN)` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | гармоническое среднее Precision и Recall |
| Accuracy | доля верно классифицированных пикселей |

Агрегированные значения, попарное сравнение, таблицы по снимкам, кэш предсказаний и
telemetry описаны в [docs/testing-quality.md](docs/testing-quality.md).

---

## 📦 Сборка бинарника

```bash
bash scripts/build_nuitka.sh
# → ./BacteriaAnalyzer
```

Скрипт создаёт изолированное окружение `.venv-build` (runtime-зависимости + Nuitka и
zstandard, без PyTorch) и собирает onefile-исполняемый файл. Окружения `.venv`,
`.venv-dev` и `.venv-full` при этом не изменяются. На Linux нужны `g++` и `patchelf`.
Windows-бинарник собирается в CI.

Проверенная Linux onefile-сборка (Nuitka 4.2.2, ONNX Runtime 1.27.0, обе встроенные
модели) занимает **≈162 MiB**; standalone-каталог — около 499 MiB.

CI (`.github/workflows/build.yaml`) запускает сборку на `push` в `main`/`master` и на
PR в `develop`; сборки, меняющие только `openspec/**`, пропускаются.

---

## 🧷 Тесты

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev

# Быстрые тесты (без slow и GUI)
UV_PROJECT_ENVIRONMENT=.venv-dev uv run --no-sync pytest -m "not slow and not gui"

# GUI-тесты responsiveness/worker (нужен доступный Qt display/offscreen backend)
UV_PROJECT_ENVIRONMENT=.venv-dev uv run --no-sync pytest -m gui

# Полный набор
UV_PROJECT_ENVIRONMENT=.venv-dev uv run --no-sync pytest
```

Маркеры: `slow` (нужны реальные изображения), `gui` (нужен QApplication),
`hypothesis` (property-based тесты). CI регулярно запускает тесты `not slow and
not gui` на Linux и Windows и часть GUI worker-тестов на Windows с offscreen backend; полный GUI heartbeat acceptance-набор локально запускайте командой выше.

---

## 🗂 Структура проекта

```text
bacterial-colony-area/
├── main.py                     Точка входа: QApplication, тема, регистрация моделей
├── pyproject.toml              Зависимости, extras, console scripts
├── analysis/                   Поиск чашки, предобработка, детекция колоний (OpenCV)
│   ├── colony_detector.py      Чашка (отражения → Hough), бинаризация, компоненты
│   ├── image_processor.py      Зелёный канал, CLAHE, медианный blur
│   └── geometry.py, params.py, results.py
├── ui/                         Окна PyQt6, shared background operations, стили
├── labeling/                   Разметка масок: локальный рендеринг, async I/O, ZIP
├── utils/                      Загрузка изображений, расчёт площади, конфиг, логи
├── testing/                    Оценка: CPU/memory-aware pipeline, telemetry, отчёты
│   ├── registry.py             @register_algorithm / register_algorithm_instance
│   ├── classic_algorithms.py   4 классических профиля
│   ├── onnx_algorithm.py       ONNX-адаптер + регистрация моделей из models/
│   ├── tiled_onnx_algorithm.py MobileNet: тайловый CPU-инференс
│   └── dataset.py, metrics.py, runner.py, baseline.py, evaluator.py, dashboard.py
├── train/                      Данные, аугментация, обучение, оценка, экспорт ONNX
│   ├── main.py                 Подкоманды train / train-compare / train-colony
│   ├── dataset_manifest.py     Контракт данных (DatasetManifest)
│   ├── dataset_adapters.py     ManifestAdapter / LabelingAdapter / PairsAdapter
│   ├── dataset_split.py, augment.py, training_pipeline.py
│   ├── models/                 unet, unet_small, mobilenet_v3_small_unet
│   └── reporter.py, dashboard/ FastAPI + WebSocket дашборд
├── models/                     Встроенные *.onnx, упаковываются в бинарник
├── scripts/                    nuitka_build.py, nuitka_flags.py, build_nuitka.sh
├── test_images/                Тестовые снимки и эталонные маски
├── datasets/                   Исходные внешние датасеты (например, 22022540)
├── evaluations/                Отчёты пакетных прогонов evaluate
├── tests/                      Pytest-тесты (unit, GUI, property-based)
├── docs/                       Документация (см. ниже)
├── openspec/                   Спецификации изменений
└── AGENTS.md                   Служебные инструкции для ИИ-агентов
```

---

## 📖 Документация

| Файл | О чём |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | **Полное руководство**: GUI по шагам, все CLI-флаги, форматы данных, обучение, сборка, ограничения |
| [docs/architecture.md](docs/architecture.md) | **Архитектура кода**: слои, модули, сигнатуры, потоки данных, точки расширения, подводные камни |
| [docs/algorithms.md](docs/algorithms.md) | **Алгоритмы детекции**: классика и ONNX-адаптеры, контракт модели, как добавить свой алгоритм |
| [docs/dataset-training-pipeline.md](docs/dataset-training-pipeline.md) | Путь от сырого COCO-датасета до обученной модели в приложении |
| [docs/datasets.md](docs/datasets.md) | Официальные источники, версии, цитирование, лицензии и связь датасетов с локальными папками |
| [docs/testing-quality.md](docs/testing-quality.md) | Метрики, реестр алгоритмов, батчи и потоки, кэширование, telemetry, структура отчётов |
| [models/README.md](models/README.md) | Как добавить свою ONNX-модель в `models/` |
| [AGENTS.md](AGENTS.md) | Правила работы с репозиторием (для ИИ-агентов и контрибьюторов) |

---

## ⚠️ Ограничения

- **Маски разметчика всегда в PNG.** Загрузчики тестовых пар ищут маску с тем же
  расширением, что у изображения. Для JPG/TIFF-исходника подготовьте одинаковые
  расширения либо укажите точные пути в `dataset.json`.
- **GUI тестирования принимает legacy, importer и `dataset.json` manifest-структуры.**
  CLI-режим `report` по-прежнему использует обычные пары через `TestDataset`;
  `baseline`/`evaluate` и обучение также читают манифест.
- **Обучение бинарное.** Multiclass-маски не поддерживаются: loss BCE+Dice, бинаризация
  порогом 0.5.
- **COCO bbox → эллипс — приблизительная разметка** (weak labels), а не точный контур
  колонии.
- **Масштабы inference разные:** встроенные `NN:colony_seg` и MobileNet работают
  по перекрывающимся тайлам полного разрешения; универсальный ONNX-адаптер
  уменьшает кадр до размера модели и растягивает маску обратно.
- **Скорость NN-зависит от CPU и размера снимка.** Модель работает асинхронно в
  фоновом потоке, окно показывает прогресс тайлов.
- **Кнопка «💾 Сохранить»** в окне анализа сохраняет отображаемый pixmap области
  просмотра, поэтому при масштабировании разрешение может быть ниже исходного.
- **Приложение полностью на русском языке.**

---

## 📄 Лицензия

Проект с открытым исходным кодом.
