# training-data-contract Specification

## Purpose
Единый контракт данных для обучения: манифест `dataset.json`, адаптеры источников (манифест, сессия разметки, легаси-пары) и фабрика `load_manifest`, чтобы обучение не зависело от конкретных структур папок.
## Requirements
### Requirement: Dataset manifest

Система SHALL описывать датасет манифестом `dataset.json` с полями: `name`, `origin` (`labeling` | `legacy` | `external`), `mask_mode` (`binary` | `multiclass`), `storage` (`reference` | `copy`) и списком `samples`. Каждый sample SHALL содержать `id`, `kind` (`source` | `cropped` | `augmented`), относительные пути `image` и `mask` от корня датасета, опциональные `subset` (`train` | `val` | `test`) и `dish` (`cx`, `cy`, `r`). Записи без маски или с несуществующим файлом SHALL быть отброшены при загрузке.

#### Scenario: Manifest loaded with samples
- **КОГДА** в корне датасета существует валидный `dataset.json` с двумя samples
- **ТОГДА** `load_manifest(data_root)` SHALL вернуть манифест с двумя записями, а пути `image`/`mask` SHALL резолвиться относительно корня

#### Scenario: Invalid mask_mode rejected
- **КОГДА** в манифесте `mask_mode == "multiclass"`, а текущий конвейер поддерживает только `binary`
- **ТОГДА** система SHALL выбросить ошибку с понятным сообщением «multiclass не поддерживается»

### Requirement: Labeling session adapter

Система SHALL предоставлять `LabelingAdapter`, который сканирует структуру сессии разметки и строит манифест: пары `source/*.{ext}` + `masks/{stem}_mask.{ext}` (kind=`source`) и `cropped/*.{ext}` + `cropped_masks/{stem}_cropped_mask.{ext}` (kind=`cropped`). Адаптер SHALL отбрасывать пары без маски и работать в режиме `storage="reference"` (без копирования файлов). При отсутствии обеих структур адаптер SHALL возвращать пустой манифест.

#### Scenario: Session pairs become manifest
- **КОГДА** `data_root` содержит `source/X.png` + `masks/X_mask.png` и `cropped/Y_cropped.png` + `cropped_masks/Y_cropped_mask.png`
- **ТОГДА** `LabelingAdapter.build(data_root)` SHALL вернуть манифест с двумя записями (kind `source` и `cropped`) и `mask_mode="binary"`

#### Scenario: Pair without mask skipped
- **КОГДА** в `source/` есть `Z.png`, но нет `masks/Z_mask.png`
- **ТОГДА** запись для `Z.png` SHALL отсутствовать в манифесте

### Requirement: Pairs adapter for legacy format

Система SHALL предоставлять `PairsAdapter`, который строит манифест из легаси-структуры `data_root/images/` + `data_root/masks/` по совпадающим именам файлов (`kind="source"`), `mask_mode="binary"`.

#### Scenario: Legacy pairs adapter builds manifest
- **КОГДА** `data_root` содержит `images/X.png` и `masks/X.png`
- **ТОГДА** `PairsAdapter.build(data_root)` SHALL вернуть манифест с записью `X.png` → `X.png`

#### Scenario: Legacy pair missing mask
- **КОГДА** `images/X.png` существует, а `masks/X.png` — нет
- **ТОГДА** манифест SHALL не содержать запись `X.png`

### Requirement: Manifest loader factory

Система SHALL предоставлять фабрику `load_manifest(data_root)` с приоритетом адаптеров: `ManifestAdapter` (если есть `dataset.json`), иначе `LabelingAdapter`, иначе `PairsAdapter`. При пустом результате фабрика SHALL выбрасывать `FileNotFoundError` с сообщением про отсутствие пар в `data_root`.

#### Scenario: Factory resolves labeling session
- **КОГДА** в `data_root` нет `dataset.json`, но есть `source/` и `masks/`
- **ТОГДА** `load_manifest(data_root)` SHALL использовать `LabelingAdapter`

#### Scenario: Factory resolves legacy pairs
- **КОГДА** в `data_root` нет ни `dataset.json`, ни `source/`, но есть `images/` и `masks/`
- **ТОГДА** `load_manifest(data_root)` SHALL использовать `PairsAdapter`

#### Scenario: No pairs raises
- **КОГДА** ни один адаптер не вернул пар
- **ТОГДА** `load_manifest(data_root)` SHALL выбросить `FileNotFoundError`