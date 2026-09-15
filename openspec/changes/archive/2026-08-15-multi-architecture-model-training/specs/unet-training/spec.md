# unet-training Specification

## MODIFIED Requirements

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