## MODIFIED Requirements

### Requirement: Dataset loading and split

Система SHALL загружать пары «изображение+маска» через контракт данных (`training-data-contract`): `make_datasets` использует `load_manifest(data_root)` (адаптеры: манифест, сессия разметки `source/masks` + `cropped/cropped_masks`, легаси `images/masks`) и возвращает только train/validation records. Если manifest содержит `subset="train"` и/или `subset="val"`, train-набор SHALL включать только train-записи, validation — только val-записи; существующие test-записи SHALL NOT перераспределяться в train/validation. Если subset train/val отсутствуют, SHALL применяться воспроизводимое seed-разделение записей по `val_split`. Каждый элемент стандартного датасета SHALL возвращать нормализованное RGB `[C,H,W]` float32 (ImageNet mean/std) и бинарную маску `[1,H,W]`, resized до `img_size` (LINEAR для изображения, NEAREST для маски). При пустом train/validation наборе SHALL возвращаться понятная ошибка; записи test без train или validation SHALL NOT включаться неявно в обучение.

#### Scenario: Labeling session pairs loaded
- **WHEN** `data_root` — сессия разметки (`source/X.png` + `masks/X_mask.png`, `cropped/Y_cropped.png` + `cropped_masks/Y_cropped_mask.png`) без манифеста
- **THEN** `make_datasets` SHALL загрузить обе пары (source и cropped) через `LabelingAdapter` как train/validation записи с воспроизводимым разделением

#### Scenario: Manifest subsets respected without test leakage
- **WHEN** `dataset.json` содержит записи с `subset: "train"`, `subset: "val"` и `subset: "test"`
- **THEN** train/validation наборы SHALL содержать только свои subset, SHALL NOT переназначать test-записи, а `val_split` SHALL не менять заданные subset

#### Scenario: Manifest subsets respected
- **WHEN** `dataset.json` содержит записи с `subset: "train"` и `subset: "val"`
- **THEN** разделение train/validation SHALL следовать subset из манифеста, а не `val_split`

#### Scenario: Legacy format still supported
- **WHEN** `data_root` — легаси-структура `images/` + `masks/` с совпадающими именами
- **THEN** `make_datasets` SHALL загрузить пары через `PairsAdapter` и применить прежнее seed-разделение train/validation

#### Scenario: No train or validation samples raises
- **WHEN** данные отсутствуют или фиксированный manifest не содержит требуемой train/validation части
- **THEN** обучение SHALL остановиться с понятной ошибкой, а test samples SHALL NOT подставляться в отсутствующую часть

#### Scenario: No pairs raises
- **WHEN** ни один адаптер не вернул image-mask пары
- **THEN** `load_manifest` и `make_datasets` SHALL выбросить `FileNotFoundError` с сообщением про отсутствие пар в `data_root`

## ADDED Requirements

### Requirement: Separate test dataset access

Система SHALL предоставлять способ отдельно загрузить записи `subset="test"` для итоговой оценки. Test loader SHALL быть read-only относительно checkpoint selection и SHALL NOT добавлять test samples в train/validation наборы.

#### Scenario: Test samples are loaded separately
- **WHEN** test dataset запрашивается для manifest с train/val/test subset
- **THEN** он SHALL содержать только test-записи и SHALL сохранять их связь с исходным sample ID

#### Scenario: Test data is absent
- **WHEN** test subset отсутствует или пуст
- **THEN** загрузчик SHALL вернуть пустой/отсутствующий test набор с диагностикой, не подмешивая train/validation данные

### Requirement: MobileNetV3-Small segmentation model registration

Реестр обучения SHALL предоставлять отдельную архитектуру `mobilenet_v3_small_unet` с энкодером MobileNetV3-Small и компактным декодером бинарной сегментации. Архитектура SHALL наследовать существующий контракт `BaseSegmenter`, поддерживать одноканальные logits для существующего BCE+Dice обучения и принимать параметр для включения/отключения ImageNet-pretrained инициализации энкодера.

#### Scenario: Architecture is listed and constructible
- **WHEN** загружается реестр моделей обучения
- **THEN** `list_models()` SHALL содержать `mobilenet_v3_small_unet`, а `get_model()` SHALL создавать совместимую с обучением модель

#### Scenario: Pretrained weights can be disabled
- **WHEN** архитектура создаётся с отключённым pretrained флагом
- **THEN** она SHALL создаваться без загрузки внешних весов и сохранять тот же формат входа/выхода
