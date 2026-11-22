# Пайплайн конвертации датасета для обучения модели

Инструкция описывает подготовку и распространение обучающего набора COCO 22022540,
а затем обучение из проверенного распакованного snapshot. `train-colony` не запускает
импорт COCO и не изменяет файлы данных. Первичная запись датасета, её версия, статья
и условия использования указаны в [каталоге внешних датасетов](datasets.md#annotated-dataset-for-deep-learning-based-bacterial-colony-detection-22022540).

## 0. Общая схема

```text
datasets/22022540/                       COCO-источник, read-only
  ├── 369 размеченных JPG
  └── annot_COCO.json (56 865 боксов)
       │
       ├─ prepare-colony-training (preflight, masks, split, lossless WebP variants)
       ▼
datasets/work/22022540-training-webp/   полный prepared snapshot, локальный
       │ pack-colony-training (tar.zst level 19, полный SHA-256 round-trip)
       ▼
datasets/archives/22022540-training-webp.tar.zst + обычные JSON sidecars
       │ git lfs pull → unpack-colony-training (новый checkout)
       ▼
datasets/work/22022540-training-webp/   полный распакованный snapshot
       │ train-colony (проверка до baseline/optimizer, данные read-only)
       ▼
train/runs/<new-run>/                    checkpoint + paired test report
```

Каждый black-background вариант имеет размер исходного снимка и сохраняет пиксели
точно (lossless WebP); исходный JPG копируется байт-в-байт, маска остаётся PNG.
Source и вариант наследуют один subset. Подготовка и обучение разделены; случайные
512×512 patches и аугментации создаются только в памяти. Публикация требует отдельного
ручного решения владельца после проверки лицензии и Git LFS квоты.

---

## 1. Что лежит в источнике `datasets/22022540`

Файлы детекции (не маски!):

| Файл | Содержимое |
|---|---|
| `annot_COCO.json` | **Канонический источник.** `images` (file_name, width, height, id) + `annotations` (bbox `[x,y,w,h]`, image_id, category_id). 369 изображений, 56 865 колоний. |
| `annot_tab.tsv` / `.csv` | Тот же набор координат в табличном виде (избыточно). |
| `annot_YOLO.zip` | 369 `.txt`, нормализованные `<class> <cx> <cy> <w> <h>`. |
| `annot_VOC_XML.zip` | 369 `.xml`, `<object><bndbox>...`. |

Три формата дублируют разметку — они нужны **только для верификации**, а не для импорта.

---

## 2. Низкоуровневый импорт COCO: `CocoBboxImporter` (`train/dataset_adapters.py`)

> `import-22022540` остаётся для общего/ручного импорта. Это не flow распространения
> обучающего архива: для training snapshot используйте `prepare-colony-training`,
> затем `pack-colony-training`; обучающий `train-colony` читает только распакованный snapshot.

CLI-вход — `train/importer_cli.py`, скрипт `import-22022540`
(объявлен в `pyproject.toml`):

```bash
uv run import-22022540 \
  --data-root datasets/22022540 \   # обязательный, источник (read-only)
  --output /tmp/ds_22022540 \
  --split --seed 42 \
  --black-background \
  [--crop]                          # доп. обрезки по чашке (kind=cropped)
  [--verify]                        # сверка COCO vs YOLO/VOC
```

Вызывается `CocoBboxImporter.build(data_root, output_dir, crop, split, seed,
black_background)`. Чёрнофоновые варианты требуют `split=True`.

### 2.1 Чтение COCO и группировка боксов

```python
raw = json.loads(coco_path.read_text(...))
images_by_id = {im["id"]: im for im in raw["images"]}
boxes_by_image: dict[int, list[(category, bbox)]] = {}
for ann in raw["annotations"]:
    boxes_by_image.setdefault(ann["image_id"], []).append(...)
```

Каждому изображению сопоставляется список его боксов.

### 2.2 Растризация бокса → маска (`_rasterize_ellipse`)

Это то, что превращает «прямоугольник» в «колонию». Данные детекции содержат
**бокс вокруг круглой колонии**. Простая заливка прямоугольника захватила бы
фон в углах, поэтому заливается наклонённый на 0° **эллипс, вписанный в бокс**:

```python
x, y, w, h = bbox
center = (int(x + w/2), int(y + h/2))
axes   = (max(int(w/2), 1), max(int(h/2), 1))
cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
```

- Центр = середина бокса `(x+w/2, y+h/2)`.
- Полуоси = `w/2`, `h/2` → эллипс касается всех четырёх сторон бокса.
- Результат — бинарная маска **255 = колония, 0 = фон** (размера исходного снимка).

Для снимка несколько боксов склеиваются: `mask |= _rasterize_ellipse(…)`.

### 2.3 Пропуск «пустых» снимков

```python
boxes = boxes_by_image.get(image_id)
if not boxes:
    continue            # нет боксов → ни маски, ни записи в манифесте
```

### 2.4 Геометрия чашки Петри (`_find_dish`)

```python
from analysis.colony_detector import ColonyDetector
_, petri = ColonyDetector().detect_petri_dish(image)
return {"cx": petri.cx, "cy": petri.cy, "r": petri.radius}
```

- Переиспользуется существующий авто-поиск чашки (блики → контур →
  `minEnclosingCircle`, при неудаче HoughCircles).
- Чашка ищется при `--crop` или `--black-background`. Для обычного source-only-импорта
  поиск по-прежнему пропускается ради скорости, и `dish` остаётся `None`.
- В режиме `--black-background` невалидная/отсутствующая геометрия приводит к ошибке
  с ID source sample; запись не пропускается молча.

### 2.5 Материализация файлов

```bash
<output>/
├── source/sp01_img01.jpg        # исходник
├── source/sp01_img01_mask.png   # бинарная GT-маска (255=колония)
├── augmented/                   # только при --black-background
│   └── sp01_img01_black_background.webp
└── ...
```

Чёрный вариант имеет исходный размер: пиксели внутри круга чашки остаются
неизменными, всё за пределами круга становится нулевым (чёрным). Он записывается
как lossless WebP (OpenCV quality=101), затем декодируется и сравнивается
попиксельно с массивом до записи; GT-маска разделяется с оригиналом.

При `--crop` дополнительно:

```bash
cropped/XXX_cropped.jpg           # обрезка по кругу чашки, фон вне круга = 0
cropped/XXX_cropped_mask.png
```

Внешность круга гасится по маске `(dx²+dy²) > r² → 0`.

### 2.6 Манифест `dataset.json` (`_write_dataset_json`)

Схема — та же, что читает `ManifestAdapter` (`training-data-contract`):

```json
{
  "name": "ds_22022540",
  "origin": "external",
  "mask_mode": "binary",
  "storage": "copy",
  "samples": [
    {
      "id": "sp01_img01",
      "kind": "source",
      "image": "source/sp01_img01.jpg",
      "mask": "source/sp01_img01_mask.png",
      "subset": "train",
      "dish": {"cx": 1841, "cy": 1479, "r": 1879}
    },
    {
      "id": "sp01_img01__black_background",
      "kind": "augmented",
      "image": "augmented/sp01_img01_black_background.webp",
      "mask": "source/sp01_img01_mask.png",
      "subset": "train",
      "source_id": "sp01_img01"
    }
  ]
}
```

Поля: `kind` (`source|cropped|augmented`), пути **относительные** от корня,
опциональные `subset`, `dish` и `source_id` (для ссылки варианта на исходник).

### 2.7 Верификация (`--verify`, `_verify_counts`)

Контроль, что растризация не потеряла колонии. Сверка числа боксов на снимок
из трёх независимых форматов:

- **COCO** — счёт идёт из `annotations`.
- **YOLO** — число строк в `{stem}.txt`.
- **VOC** — число тегов `<object>` в `{stem}.xml` (не строк файла!).

```python
if abs(coco_cnt - ref_cnt) > 5:
    log.warning("Расхождение аннотаций %s: COCO=%d, дубль=%d", ...)
```

В этом датасете все три формата согласованы → 0 расхождений, что подтверждает
корректность масок.

### 2.8 Парная подготовка чёрнофоновых вариантов

```bash
uv run import-22022540 \
  --data-root datasets/22022540 \
  --output /tmp/ds_22022540_black \
  --split --seed 42 --black-background --verify
```

Импорт сначала назначает source ID стратифицированным группам train/val/test по
категории COCO и записывает этот source-level split в `split.json`. Только затем
создаются `kind=augmented` записи: каждому `source` соответствует один
полноразмерный вариант с тем же `subset`, той же GT-маской и стабильным
`source_id`. Поэтому оригинал и его чёрный вариант не могут попасть в разные
выборки. Подготовка пишет только в отдельный `--output`; каталог источника
остаётся read-only.

---

## 3. Общий контракт манифеста (generic `train`)

> Этот раздел описывает общий адаптер `load_manifest` и команду `train`.
> Воспроизводимый COCO `train-colony` workflow использует полный `split.json`,
> `validate_prepared_dataset` и patch-loader; см. последовательность в разделе 7.

Датасет готов, теперь его потребляет ядро обучения без знания про COCO.

### 3.1 `load_manifest(data_root)` (`train/dataset_adapters.py`)

Фабрика выбирает адаптер по структуре папок, по приоритету:

```python
if (root/"dataset.json").is_file():  manifest = ManifestAdapter()
elif (root/"source").is_dir():       manifest = LabelingAdapter()
elif (root/"images").is_dir():       manifest = PairsAdapter()
else: raise FileNotFoundError(...)
```

Для вывода работает `dataset.json` → **`ManifestAdapter`**. Он де-сериализует
JSON обратно в `DatasetManifest` + `SampleRecord` (включая `DishGeometry`) и
валидирует: известные `origin`/`storage`/`kind`, существование файлов
`image`/`mask`, `mask_mode` (только `binary`).

### 3.2 `make_datasets(...)` (`train/dataset.py`)

```python
manifest = load_manifest(data_root)
train_records, val_records = _subsets_from_manifest(manifest, val_split, seed)
train_ds = ColonyDataset(train_records, ...)
val_ds = ColonyDataset(val_records, augment=False)
```

Разбиение train/val:
- если в записях есть явные `subset` — используется манифест;
- иначе **seed-сплит** от `val_split=0.2` (для 369 пар → 296 train / 73 val,
  воспроизводимо от `seed=42`).

### 3.3 Загрузка одного сэмпла (`ColonyDataset.__getitem__`)

```python
image = cv2.imread(...)                        # BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask  = cv2.imread(..., IMREAD_GRAYSCALE)
mask  = (mask > 127).astype(np.float32)        # нормализация маски к 0/1

# аугментация (train): flip / поворот
# ресайз (для dense-чашек ключевой шаг):
image = cv2.resize(image, (img_size, img_size), INTER_LINEAR)
mask  = cv2.resize(mask,  (img_size, img_size), INTER_NEAREST)

# нормализация ImageNet:
image = (image/255 - MEAN) / STD                # MEAN=[.485,.456,.406] STD=[.229,.224,.225]
# тензоры: image -> [3,H,W], mask -> [1,H,W]
```

Здесь **основная «настройка датасета под проект»**: полноразмерные снимки
(2800–5600 px) ресайзятся до `img_size` (512), а маски — билинейно/наеarest
так, чтобы колонии сохранялись.

---

## 4. Общая команда `train` (не COCO LFS workflow)

Команда ниже относится к generic training API; для `22022540` используйте
`train-colony` из раздела 7, который требует заранее распакованный полный snapshot.

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.main train \
  --model unet --data-root /tmp/ds_22022540 \
  --img-size 512 --epochs 200
```

`--data-root` передаётся в конфиг (`train/main.py`), путь идёт в `make_datasets`.
Потери BCE+Dice, ранняя остановка по IoU на валидации. Результаты:

```
train/runs/unet_<ts>/
├── summary.json               # метрики по эпохам
├── report.html                # Plotly-отчёт
├── tensorboard/
└── checkpoints/
    ├── best.pt / best.onnx    # лучшая по валидации
    └── last.pt / last.onnx
```

---

## 5. Шаг экспорта в single-file ONNX

`best.onnx` из шага обучения по умолчанию хранит веса **внешне**
(`best.onnx.data`, ~53 МБ). Для чистого bundled-файла переэкспортируется модель
из `best.pt` с `external_data=False`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.export_single \
  --checkpoint train/runs/unet_<ts>/checkpoints/best.pt \
  --output train/runs/example_new_run/colony_candidate.onnx --img-size 512
```

`--output` должен указывать на отдельный новый путь; не записывайте candidate в
`models/colony_seg.onnx`. Promotion — отдельное явное действие после ревью report.

> `train/export_single.py` — вспомогательный скрипт переэкспорта best.pt
> в самодостаточный single-file ONNX.

Сигнатура: вход `['batch',3,512,512]` (batch dynamic), выход `['batch',1,512,512]`,
opset 18. Один файл без внешних данных.

---

## 6. Шаг интеграции как алгоритм

### 6.1 Авто-регистрация bundled-модели

`models/colony_seg.onnx` сканируется при старте:

```python
# testing/onnx_algorithm.py — production model special case
if model_file.stem == "colony_seg":
    algos.append(TiledOnnxModelAlgorithm(
        model_path=str(model_file), name="NN:colony_seg",
        img_size=512, stride=384, threshold=0.5, output_is_logits=True,
    ))
else:
    algos.append(OnnxModelAlgorithm(
        model_path=str(model_file), name=f"NN:{model_file.stem}", ...
    ))
```

→ в реестре появляется алгоритм **`NN:colony_seg`**, как у классических.

### 6.2 Инференс (`TiledOnnxModelAlgorithm.detect`)

```python
for tile in tiles(image, size=512, stride=384):
    tensor = normalize_rgb(tile)              # ImageNet mean/std, [1,3,512,512]
    logits = session.run(None, {input: tensor})[0][0,0]
    probabilities = sigmoid(logits)
    accumulate(probability_sum, probability_count, probabilities, tile.position)
mask = (probability_sum / probability_count >= 0.5).astype(uint8) * 255
```

Полноразмерный кадр не сжимается целиком: каждый 512×512 тайл нормализуется так
же, как обучающие патчи; соседние тайлы перекрываются на 128 px. Логиты
преобразуются сигмоидой в вероятности, усредняются в областях перекрытия, затем
порог 0.5 формирует маску исходного размера. Инференс в приложении выполняется
на CPU через ONNX Runtime.

---

## 7. End-to-End: подготовить → упаковать → получить/распаковать → обучить

### 7.1 Подготовить и упаковать на машине с локальным COCO

```bash
export UV_PROJECT_ENVIRONMENT=.venv-full
uv sync --extra full
uv run --no-sync prepare-colony-training \
  --data-root datasets/22022540 \
  --output datasets/work/22022540-training-webp \
  --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --black-background
uv run --no-sync pack-colony-training \
  --data-root datasets/work/22022540-training-webp \
  --archive datasets/archives/22022540-training-webp.tar.zst
```

Preflight проверяет все COCO-изображения с аннотациями до создания snapshot; source JPG
копируются байт-в-байт, маски сохраняются как PNG, black-background варианты —
полноразмерные pixel-exact lossless WebP. Seed/split и пути манифеста проверяются до
упаковки. Архиватор использует Zstandard level 19 и полностью сверяет SHA-256 после
round-trip; sidecar JSON содержит копии метаданных и фактический размер/коэффициент.
Проверьте `.archive.json` и per-file limit вашего LFS-плана до публикации. Ни одна из
команд подготовки/упаковки не выполняет Git add, commit или push.

### 7.2 Явно опубликовать (только после проверки владельцем)

Источник версии 3 указывает [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([Figshare DOI](https://doi.org/10.6084/m9.figshare.22022540.v3)); сохраняйте атрибуцию
авторов и ссылку на лицензию, перепроверьте локальную копию перед публикацией. По
[GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
(проверено 2026-11-08), GitHub Free/Pro перечисляет 10 GiB storage + 10 GiB download
bandwidth; Team/Enterprise Cloud — 250 GiB + 250 GiB. Лимит одного файла: Free/Pro 2 GB,
Team 4 GB, Enterprise Cloud 5 GB ([официальные лимиты](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)).
Квоты и бюджет зависят от плана/аккаунта и могут измениться; новые архивные версии
учитывают свой полный storage-size. Не публикуйте, если файл превышает лимит или если
квота/лицензия/видимость репозитория не подходят.

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

Только `*.tar.zst` хранится в LFS; JSON sidecars остаются обычными Git-файлами.
Эти команды — ручная инструкция, а не автоматический шаг. Git LFS необходимо
[установить](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
и активировать (`git lfs install`). На Fedora установите пакет командой `sudo dnf install git-lfs`; другие ОС см. в [официальной инструкции](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

### 7.3 Получить и распаковать на чистом checkout

После того, как владелец опубликовал архив, клонируйте репозиторий с установленным
Git LFS. `git lfs pull` получает объект; без него на месте `.tar.zst` может быть
небольшой LFS pointer.

```bash
git lfs install
git clone https://github.com/Egel68/bacterial-colony-area.git training-checkout
cd training-checkout
git lfs pull
git lfs ls-files
test -s datasets/archives/22022540-training-webp.tar.zst
export UV_PROJECT_ENVIRONMENT=.venv-full
uv sync --extra full
uv run --no-sync unpack-colony-training \
  --archive datasets/archives/22022540-training-webp.tar.zst \
  --output datasets/work/22022540-training-webp
```

Распаковка в новый каталог проверяет SHA-256 архива, sidecar-метаданные, manifest, split
и каждый файл/путь. См. [Git LFS collaboration](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage).

### 7.4 Обучить только из полного распакованного snapshot

```bash
export UV_PROJECT_ENVIRONMENT=.venv-full
uv run --no-sync train-colony \
  --data-root datasets/work/22022540-training-webp \
  --output train/runs/22022540_black_background_seed42_your-new-run \
  --model unet --black-background --no-pretrained --seed 42 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --epochs 100 --batch-size 2 --patches-per-image 4 --img-size 512 \
  --cpu-benchmark
```

Перед запуском выберите новый/пустой `--output`, отдельно от snapshot; U-Net black-background
обучение требует CUDA и не использует CPU fallback. `train-colony` проверит `dataset.json`,
`split.json` и все пары до baseline/optimizer, будет использовать сохранённый split, не
импортирует COCO и не записывает images/masks. Patches и аугментации 512×512 создаются
только в памяти. Checkpoint выбирается по validation, test остаётся held-out. Новый
ONNX и отчёты сохраняются в `<output>/runs/...`; `models/colony_seg.onnx` не меняется
автоматически.

Основные флаги `train-colony` (сверены с `train-colony --help`): `--data-root`, `--output`,
`--model {mobilenet_v3_small_unet,unet}`, `--black-background`, `--seed`,
`--train-ratio`, `--val-ratio`, `--test-ratio`, `--epochs`, `--batch-size`,
`--patches-per-image`, `--img-size` (поддерживается 512), `--no-pretrained`,
`--resume-run`, `--cpu-benchmark`, `--promote-model` и `--allow-overwrite`. Пара `--black-background`
требует `--model unet` и несовместима с `--resume-run`; promotion модели — только отдельным
явным путём после ревью отчёта.

---

## Ключевые точки, где «настраивается датасет под проект»

1. **Формат боксов → бинарные маски** (эллипс, не прямоугольник) —
   `_rasterize_ellipse`.
2. **Разные форматы сводятся к единому контракту** через `dataset.json` +
   `ManifestAdapter` — ядро обучения не менялось.
3. **Обучение и production-инференс используют один масштаб:** обучение берёт
   патчи 512×512, приложение покрывает исходный кадр тайлами 512×512 с
   шагом 384; целое изображение не сжимается.
4. **Модель решает только «есть колония»** (binary mask, BCE+Dice, порог 0.5),
   а не классификацию вида — production регистрирует её через
   `TiledOnnxModelAlgorithm` и исполняет на CPU.

---

## Связанные файлы

| Файл | Роль |
|---|---|
| `train/dataset_adapters.py` | `CocoBboxImporter`, `ManifestAdapter`, `load_manifest` |
| `train/importer_cli.py` | CLI `import-22022540` + `--verify` |
| `train/export_single.py` | Переэкспорт best.pt → single-file ONNX |
| `train/dataset_manifest.py` | Контракт `DatasetManifest` / `SampleRecord` / `DishGeometry` |
| `train/dataset.py` | `make_datasets`, `ColonyDataset`, seed-сплит |
| `train/main.py` | CLI обучения, `--data-root` |
| `testing/onnx_algorithm.py` | `OnnxModelAlgorithm`, special tiled registration, `register_bundled_models` |
| `models/colony_seg.onnx` | Готовая модель (bundled, `NN:colony_seg`) |
