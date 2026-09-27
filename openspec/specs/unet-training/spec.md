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

Система SHALL загружать пары «изображение+маска» через контракт данных (`training-data-contract`): `make_datasets` использует `load_manifest(data_root)` (адаптеры: манифест, сессия разметки `source/masks` + `cropped/cropped_masks`, легаси `images/masks`) и возвращает только train/validation records. Если manifest содержит `subset="train"` и/или `subset="val"`, train-набор SHALL включать только train-записи, validation — только val-записи; существующие test-записи SHALL NOT перераспределяться в train/validation. Если subset train/val отсутствуют, SHALL применяться воспроизводимое seed-разделение записей по `val_split`. При `mask_mode != "binary"` система SHALL выбрасывать ошибку о неподдерживаемом режиме. Каждый элемент стандартного датасета SHALL возвращать нормализованное RGB `[C,H,W]` float32 (ImageNet mean/std) и бинарную маску `[1,H,W]`, resized до `img_size` (LINEAR для изображения, NEAREST для маски). При пустом train/validation наборе SHALL возвращаться понятная ошибка; записи test без train или validation SHALL NOT включаться неявно в обучение.

#### Scenario: Labeling session pairs loaded
- **КОГДА** `data_root` — сессия разметки (`source/X.png` + `masks/X_mask.png`, `cropped/Y_cropped.png` + `cropped_masks/Y_cropped_mask.png`) без манифеста
- **ТОГДА** обучающий конвейер SHALL обнаружить обе пары и воспроизводимо распределить записи source/cropped между train и validation

#### Scenario: Manifest subsets respected
- **КОГДА** `dataset.json` содержит записи с `subset: "train"` и `subset: "val"`
- **ТОГДА** разделение train/validation SHALL следовать `subset` из манифеста, а не `val_split`

#### Scenario: Manifest subsets respected without test leakage
- **КОГДА** `dataset.json` содержит записи с `subset: "train"`, `subset: "val"` и `subset: "test"`
- **ТОГДА** train/validation наборы SHALL содержать только свои subset, SHALL NOT переназначать test-записи, а `val_split` SHALL не менять заданные subset

#### Scenario: Legacy format still supported
- **КОГДА** `data_root` — легаси-структура `images/` + `masks/` с совпадающими именами
- **ТОГДА** обучающий конвейер SHALL загрузить пары и применить воспроизводимое разбиение на train/validation с учётом seed и `val_split`

#### Scenario: No pairs raises
- **КОГДА** ни один адаптер не вернул пар
- **ТОГДА** `load_manifest` и `make_datasets` SHALL выбросить `FileNotFoundError` с сообщением про отсутствие пар в `data_root`

#### Scenario: No train or validation samples raises
- **КОГДА** данные отсутствуют или фиксированный manifest не содержит требуемой train/validation части
- **ТОГДА** обучение SHALL остановиться с понятной ошибкой, а test samples SHALL NOT подставляться в отсутствующую часть

### Requirement: Separate test dataset access

Система SHALL предоставлять отдельный доступ к записям `subset="test"` для итоговой оценки. Test-выборка SHALL NOT включаться в train/validation и SHALL NOT влиять на выбор чекпоинта.

#### Scenario: Test samples are loaded separately
- **WHEN** test dataset запрашивается для manifest с train/val/test subset
- **THEN** он SHALL содержать только test-записи и SHALL сохранять их связь с исходным sample ID

#### Scenario: Test data is absent
- **WHEN** test subset отсутствует или пуст
- **THEN** загрузчик SHALL вернуть пустой/отсутствующий test набор с диагностикой, не подмешивая train/validation данные

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

### Requirement: MobileNetV3-Small segmentation model registration

Реестр обучения SHALL предоставлять отдельную архитектуру `mobilenet_v3_small_unet` с энкодером MobileNetV3-Small и компактным декодером бинарной сегментации. Архитектура SHALL наследовать существующий контракт `BaseSegmenter`, поддерживать одноканальные logits для существующего BCE+Dice обучения и принимать параметр для включения/отключения ImageNet-pretrained инициализации энкодера.

#### Scenario: Architecture is listed and constructible
- **WHEN** загружается реестр моделей обучения
- **THEN** реестр SHALL содержать архитектуру `mobilenet_v3_small_unet`, а выбор этой архитектуры SHALL создавать модель, совместимую с обучающим конвейером

#### Scenario: Pretrained weights can be disabled
- **WHEN** архитектура создаётся с отключённым pretrained флагом
- **THEN** модель SHALL создаваться без сетевой загрузки весов и сохранять тот же формат входа/выхода

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
