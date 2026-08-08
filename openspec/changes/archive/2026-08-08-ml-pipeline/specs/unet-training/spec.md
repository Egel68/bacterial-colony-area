## ADDED Requirements

### Requirement: Training configuration

Система SHALL предоставлять `TrainingConfig` (dataclass) с полями: `data_root=train/data`, `img_size=512`, `batch_size=8`, `epochs=200`, `lr=1e-3`, `weight_decay=1e-5`, `val_split=0.2`, `num_workers=4`, `seed=42`, `patience=30`, `augment=True`, `model_name="unet"`, `run_dir=train/runs`, `device="cuda"`, `dashboard=False`, `dashboard_port=8765`.

#### Scenario: Default config constructs
- **КОГДА** создаётся `TrainingConfig()` без аргументов
- **ТОГДА** все поля имеют значения по умолчанию из спецификации

### Requirement: Dataset loading and split

Система SHALL загружать пары `image/mask` из `data_root/images/` + `data_root/masks/` (имена файлов совпадают), отбрасывая пары без маски. `make_datasets` SHALL разделять данные на train/val по `val_split` с воспроизводимым перемешиванием от `seed=42` (`numpy.default_rng`), каждый элемент датасета SHALL возвращать нормализованное RGB-изображение `[C,H,W]` float32 (mean/std ImageNet) и бинарную маску `[1,H,W]`, resize до `img_size` (interpolation LINEAR для изображения, NEAREST для маски). При пустом наборе система SHALL выбрасывать `FileNotFoundError`.

#### Scenario: No pairs raises
- **КОГДА** в данных нет ни одной пары «изображение+маска»
- **ТОГДА** `make_datasets` SHALL выбросить `FileNotFoundError` с сообщением про отсутствие пар в `data_root`

### Requirement: Model registry and U-Net

Система SHALL предоставлять реестр моделей через `@register_model` (`train/models/__init__.py`), `get_model(name)` SHALL возвращать инстанс модели, `list_models()` — сортированный список имён. Для неизвестного имени система SHALL выбрасывать `ValueError` с перечнем доступных моделей. Архитектура `UNet` (4 down/up блока, DoubleConv с BatchNorm2d) SHALL наследовать `BaseSegmenter`.

#### Scenario: Unknown model raises
- **КОГДА** `get_model("nope")` с не зарегистрированным именем
- **ТОГДА** система SHALL выбросить `ValueError` с сообщением "Unknown model"

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