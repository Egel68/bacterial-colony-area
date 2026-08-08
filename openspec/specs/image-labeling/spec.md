# image-labeling Specification

## Purpose
TBD - created by archiving change labeling-and-testing. Update Purpose after archive.
## Requirements
### Requirement: File list filtered by supported extensions

Система SHALL формировать список изображений сессии, включая только файлы с расширениями из `SUPPORTED_EXTENSIONS` (`utils.image_loader`): `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`. Расширение SHALL сравниваться без учёта регистра. Если директория не существует, список SHALL быть пустым.

#### Scenario: Unsupported formats are hidden
- **КОГДА** в директории находятся `.png`, `.jpg`, `.webp`, а также `.gif` и `.svg`
- **ТОГДА** `list_image_files` SHALL вернуть только `.png`, `.jpg`, `.webp`

#### Scenario: Only unsupported files present
- **КОГДА** в директории есть только файл `.gif`
- **ТОГДА** `list_image_files` SHALL вернуть пустой список

#### Scenario: Missing directory
- **КОГДА** директория не существует
- **ТОГДА** `list_image_files` SHALL вернуть пустой список

### Requirement: Session directory structure

Для режима «исходники» (`source`) система SHALL использовать поддиректорию `source/` сессии как рабочую директорию. Для режима «обрезки» (`cropped`) система SHALL использовать поддиректорию `cropped/`. Если соответствующая поддиректория отсутствует, система SHALL создать её перед использованием. Для масок система SHALL использовать `masks/` (для source) или `cropped_masks/` (для cropped), создавая директорию при необходимости.

#### Scenario: Source dir as working directory
- **КОГДА** пользователь находится в режиме «исходники»
- **ТОГДА** `get_current_dir(session, "source")` SHALL вернуть `source/` (даже если в ней нет файлов)

#### Scenario: Cropped dir as working directory
- **КОГДА** пользователь находится в режиме «обрезки»
- **ТОГДА** `get_current_dir(session, "cropped")` SHALL вернуть `cropped/`

#### Scenario: Missing source dir auto-created
- **КОГДА** в сессии отсутствует `source/`
- **ТОГДА** система SHALL создать `source/` перед загрузкой списка файлов и вернуть её

#### Scenario: Mask directory selection
- **КОГДА** запрашивается маска для source
- **ТОГДА** `get_mask_dir(session, "source")` SHALL вернуть `masks/`

### Requirement: Crop by Petri dish

Система SHALL обрезать изображение по чашке: квадрат `[cx±r, cy±r]` с ограничением границ изображения, и заливать фон за пределами круга нулями (чёрным). Выходной кадр SHALL иметь 3 канала.

#### Scenario: Crop inside image
- **КОГДА** чашка целиком помещается в изображение
- **ТОГДА** `crop_by_petri` SHALL вернуть квадрат размером `2r×2r` с чёрным фоном вне круга

#### Scenario: Crop clamped at image edge
- **КОГДА** центр чашки близко к краю и обрезка выходит за границы
- **ТОГДА** SHALL применяться обрезка к границам (`max(0, cx−r)` и т. п.), не выходя за пределы изображения

### Requirement: Mask save/load

Система SHALL сохранять маску через `cv2.imwrite` как `uint8`-изображение, создавая родительскую директорию. Функция `load_mask` SHALL возвращать маску, только если она существует и имеет ожидаемый размер shape; иначе `None`.

#### Scenario: Save mask creates parent directory
- **КОГДА** `save_mask(mask, path)` в несуществующей родительской директории
- **ТОГДА** директория SHALL быть создана и файл записан

#### Scenario: Load mask shape mismatch
- **КОГДА** маска существует, но её shape не совпадает с ожидаемым
- **ТОГДА** `load_mask` SHALL вернуть `None`

### Requirement: Session export to ZIP

Система SHALL упаковывать всю папку сессии в ZIP с относительными путями от родителя сессии.

#### Scenario: Export includes all files
- **КОГДА** сессия содержит `source/`, `masks/`, `cropped/`, `cropped_masks/`
- **ТОГДА** ZIP SHALL содержать все файлы с путями вида `session/...`

