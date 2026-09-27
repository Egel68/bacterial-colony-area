# unet-training Specification

## Purpose
TBD - created by archiving change ml-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Training configuration

Система SHALL предоставлять `TrainingConfig` (dataclass) с полями: `data_root=train/data`, `img_size=512`, `batch_size=8`, `epochs=200`, `lr=1e-3`, `weight_decay=1e-5`, `val_split=0.2`, `num_workers=4`, `seed=42`, `patience=30`, `augment=True`, `model_name="unet"`, `run_dir=train/runs`, `device="cuda"`, `dashboard=False`, `dashboard_port=8765`.

#### Scenario: Default config constructs
- **КОГДА** создаётся `TrainingConfig()` без аргументов
- **ТОГДА** все поля имеют значения по умолчанию из спецификации

### Requirement: Dataset loading and split

Система SHALL загружать пары «изображение+маска» через контракт данных (`training-data-contract`): `make_datasets` использует `load_manifest(data_root)` (адаптеры: манифест, сессия разметки `source/masks` + `cropped/cropped_masks`, легаси `images/masks`) и разбивает записи на train/val. Если запись имеет `subset` из манифеста — разделение SHALL следовать манифесту; иначе SHALL применяться воспроизводимое перемешивание от `seed=42` (`numpy.default_rng`) по `val_split`. При `mask_mode != "binary"` система SHALL выбрасывать ошибку о неподдерживаемом режиме. Каждый элемент датасета SHALL возвращать нормализованное RGB-изображение `[C,H,W]` float32 (mean/std ImageNet) и бинарную маску `[1,H,W]`, resize до `img_size` (interpolation LINEAR для изображения, NEAREST для маски). При пустом наборе система SHALL выбрасывать `FileNotFoundError`.

#### Scenario: Labeling session pairs loaded
- **КОГДА** `data_root` — сессия разметки (`source/X.png` + `masks/X_mask.png`, `cropped/Y_cropped.png` + `cropped_masks/Y_cropped_mask.png`) без манифеста
- **ТОГДА** `make_datasets` SHALL загрузить обе пары (source и cropped) через `LabelingAdapter` как отдельные элементы датасета

#### Scenario: Manifest subsets respected
- **КОГДА** `dataset.json` содержит записи с `subset: "train"` и `subset: "val"`
- **ТОГДА** разделение train/val SHALL следовать `subset` из манифеста, а не `val_split`

#### Scenario: Legacy format still supported
- **КОГДА** `data_root` — легаси-структура `images/` + `masks/` с совпадающими именами
- **ТОГДА** `make_datasets` SHALL загрузить пары через `PairsAdapter`

#### Scenario: No pairs raises
- **КОГДА** ни один адаптер не вернул пар
- **ТОГДА** `load_manifest` и `make_datasets` SHALL выбросить `FileNotFoundError` с сообщением про отсутствие пар в `data_root`

### Requirement: Model registry and architectures

Система SHALL предоставлять реестр моделей через `@register_model` (`train/models/__init__.py`), `get_model(name, **params)` SHALL возвращать инстанс модели с переданными параметрами архитектуры, `list_models()` — сортированный список имён. Для неизвестного имени система SHALL выбрасывать `ValueError` с перечнем доступных моделей. Реестр SHALL содержать минимум две зарегистрированные CNN-архитектуры сегментации: `unet` (4 down/up блока, DoubleConv с BatchNorm2d) и одну дополнительную (например, `unet_small` с уменьшенным числом каналов). Каждая архитектура SHALL наследовать `BaseSegmenter` и быть параметризуемой через конструктор.

#### Scenario: Unknown model raises
- **КОГДА** `get_model("nope")` с не зарегистрированным именем
- **ТОГДА** система SHALL выбросить `ValueError` с сообщением "Unknown model"

#### Scenario: Multiple architectures listed
- **КОГДА** зарегистрированы `unet` и `unet_small`
- **ТОГДА** `list_models()` SHALL вернуть `["unet", "unet_small"]`, а `get_model("unet_small")` SHALL вернуть инстанс второй архитектуры

#### Scenario: Architecture params passed
- **КОГДА** `get_model("unet_small", features=(32, 64, 128, 256))`
- **ТОГДА** возвращаемая модель SHALL использовать переданные параметры каналов

### Requirement: Loss and metrics

Модель SHALL вычислять лосс как BCE + Dice (`smooth=1e-6`). `compute_metrics` SHALL возвращать `loss`, `iou`, `dice`, `precision`, `recall` по бинаризации предсказаний порогом `>threshold` (по умолчанию 0.5).

#### Scenario: Perfect prediction on identity
- **КОГДА** `compute_metrics(pred, target)` с pred и target, совпадающими после сигнатуры (>0.5) на ненулевом объекте
- **ТОГДА** `iou`/`dice`/`precision`/`recall` SHALL равняться 1.0 (с точностью `smooth=1e-6`)

### Requirement: Training loop

`run_training` SHALL запускать цикл из `cfg.epochs`, оптимизируя AdamW `lr=cfg.lr, weight_decay=cfg.weight_decay` с `CosineAnnealingLR(T_max=epochs)`. Система SHALL сохранять в `run_dir/{model}_{timestamp}/`: чекпоинты в `checkpoints/` — `best.pt`+`best.onnx` (лучший по валидационному `iou`) и `last.pt`+`last.onnx` после каждой эпохи, логи тензора в `tensorboard/`, и `summary.json` (model, epochs, best_epoch, best_iou, train/val samples, img_size). При отсутствии улучшения более `cfg.patience` эпох система SHALL остановиться досрочно. Если `dashboard=True`, система SHALL запустить веб-дашборд на порту `dashboard_port`.

#### Scenario: Train and validation history recorded
- **КОГДА** `run_training(cfg)` отработает на минимальном датасете
- **ТОГДА** SHALL вернуться `summary`, `train_hist`, `val_hist` (длина = числу эпох), а в `run_dir` SHALL существовать `summary.json` и `checkpoints/best.pt`

### Requirement: ONNX экспорт и загрузка

Модель SHALL экспортироваться в ONNX (`opset=18`, dynamic batch axis), и `load_onnx(path)` SHALL возвращать `onnxruntime.InferenceSession`. Предобработка `preprocess` SHALL resize в `img_size`, нормировать на ImageNet и добавить batch-dims.

#### Scenario: ONNX export produces valid session
- **КОГДА** модель экспортируется в файл `best.onnx`
- **ТОГДА** `load_onnx("best.onnx")` SHALL создать инференс-сессию без исключений

