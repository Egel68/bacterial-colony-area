# Пайплайн конвертации датасета для обучения модели

Инструкция описывает полный путь: от сырого датасета `datasets/22022540`
(COCO-bbox) до обученной модели, интегрированной в приложение как алгоритм
`NN:colony_seg`.

## 0. Общая схема

```
datasets/22022540/                 ── ИСХОДНИК · только чтение
├── spXX_imgYY.jpg       (369 фото)
├── annot_COCO.json      (56 865 боксов, COCO)
├── annot_tab.tsv|csv    (дубль формата)
├── annot_YOLO.zip       (369 .txt, дубль)
└── annot_VOC_XML.zip    (369 .xml, дубль)
        │
        ▼   train/importer_cli.py  →  CocoBboxImporter.build()
ИМПОРТ (растризация боксов→маски, materialize)
        │
        ▼
<output>/                        ── САМОДОСТАТОЧНЫЙ датасет (~607 МБ)
├── dataset.json            (манифест: name/origin/mask_mode/storage/samples)
├── source/spXX_imgYY.jpg   (копия фото)
└── source/spXX_imgYY_mask.png  (бинарная маска 255=колония)
        │
        ▼   load_manifest() → ManifestAdapter
        │   name            ── контракт данных (ядро не знает про COCO/папки)
        ▼   make_datasets() → ColonyDataset
ОБУЧЕНИЕ  train.main train → summary.json + report.html
        │
        ▼   best.pt → export_single.py
ЭКСПОРТ  models/colony_seg.onnx
        │
        ▼   register_bundled_models()
ИНТЕГРАЦИЯ OnnxModelAlgorithm.detect(image) → uint8 mask (NN:colony_seg)
```

Ключевой принцип: **ядро обучения не знает, откуда данные**. Оно потребляет
только манифест. Весь формат «COCO-боксы → маски» спрятан в одном импортёре.

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

## 2. Шаг импорта: `CocoBboxImporter` (`train/dataset_adapters.py`)

CLI-вход — `train/importer_cli.py`, скрипт `import-22022540`
(объявлен в `pyproject.toml`):

```bash
uv run import-22022540 \
  --data-root datasets/22022540 \   # обязательный, источник (read-only)
  --output /tmp/ds_22022540         # куда писать результат
  [--crop]                          # доп. обрезки по чашке (kind=cropped)
  [--verify]                        # сверка COCO vs YOLO/VOC
```

Вызывается `CocoBboxImporter.build(data_root, output_dir, crop)`.

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
- **Нюанс реализации:** чашка ищется **только при `--crop`**. Для source-only-импорта
  (нужен только для обучения по полным чашкам) поиск пропускается ради скорости —
  геометрия в `dish` тогда `None`. Это осознанное упрощение (см. design, D4/D5).

### 2.5 Материализация файлов

```bash
<output>/
├── source/sp01_img01.jpg        # копия исходника
├── source/sp01_img01_mask.png   # бинарная маска (255=колония)
└── ...                          # ×369
```

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
      "dish": {"cx": 1841, "cy": 1479, "r": 1879}
    }
  ]
}
```

Поля: `kind` (`source|cropped`), пути **относительные** от корня,
опциональные `subset` и `dish`.

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

---

## 3. Шаг чтения: манифест → обучение

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

## 4. Шаг обучения

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
  --output models/colony_seg.onnx --img-size 512
```

> `train/export_single.py` — вспомогательный скрипт переэкспорта best.pt
> в самодостаточный single-file ONNX.

Сигнатура: вход `['batch',3,512,512]` (batch dynamic), выход `['batch',1,512,512]`,
opset 18. Один файл без внешних данных.

---

## 6. Шаг интеграции как алгоритм

### 6.1 Авто-регистрация bundled-модели

`models/colony_seg.onnx` сканируется при старте:

```python
# testing/onnx_algorithm.py
for model_file in models_dir.glob("*.onnx"):
    algos.append(OnnxModelAlgorithm(model_file, name=f"NN:{model_file.stem}", ...))
```

→ в реестре появляется алгоритм **`NN:colony_seg`**, как у классических.

### 6.2 Инференс (`OnnxModelAlgorithm.detect`)

```python
rgb = cvtColor(image, BGR2RGB)
resized = resize(rgb, (512,512))              # img_size по умолчанию 512
tensor = (resized/255 - MEAN)/STD             # та же нормализация, что в обучении
raw = session.run(None, {input}: tensor)[0][0,0]   # [1,1,512,512] → [512,512]
raw = resize(raw, (w,h), INTER_LINEAR)        # назад к размеру входного фото
binary = (raw > 0.5).astype(uint8)*255        # порог 0.5 → 0/255
```

Именно **совпадение препроцессинга (ресайз в 512, ImageNet-нормализация,
BGR→RGB) между обучением и инференсом** делает модель совместимой с текущим
решением без нового кода. Выход — бинарная маска 255/0, та же, что у
классических алгоритмов.

---

## 7. End-to-End (одна строка на шаг)

```bash
# 1) импорт
uv run import-22022540 --data-root datasets/22022540 --output /tmp/ds_22022540 --verify
# 2) обучение
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.main train \
  --model unet --data-root /tmp/ds_22022540 --img-size 512 --epochs 200
# 3) single-file ONNX в models/
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m train.export_single \
  --checkpoint train/runs/unet_<ts>/checkpoints/best.pt \
  --output models/colony_seg.onnx --img-size 512
# 4) проверка как алгоритм
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -c \
  "from testing.onnx_algorithm import OnnxModelAlgorithm as A; \
   import cv2; print(A('models/colony_seg.onnx').detect(cv2.imread('datasets/22022540/sp01_img01.jpg')).shape)"
```

---

## Ключевые точки, где «настраивается датасет под проект»

1. **Формат боксов → бинарные маски** (эллипс, не прямоугольник) —
   `_rasterize_ellipse`.
2. **Разные форматы сводятся к единому контракту** через `dataset.json` +
   `ManifestAdapter` — ядро обучения не менялось.
3. **Ресайз полных чашек 2800–5600 px → `img_size` + ImageNet-нормализация**
   согласованы между `ColonyDataset` и `OnnxModelAlgorithm.detect`.
4. **Модель решает только «есть колония»** (binary mask, BCE+Dice, порог 0.5),
   а не классификацию вида — поэтому подключается штатным `OnnxModelAlgorithm`.

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
| `testing/onnx_algorithm.py` | `OnnxModelAlgorithm`, `register_bundled_models` |
| `models/colony_seg.onnx` | Готовая модель (bundled, `NN:colony_seg`) |