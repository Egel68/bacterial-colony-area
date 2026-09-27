# coco-bbox-importer Specification

## Purpose
Импорт внешнего датасета в формате детекции (COCO bbox) в контракт данных проекта: растризация боксов колоний в бинарные маски, создание файлов `kind=source`/`cropped` и материализация самодостаточной папки датасета с `dataset.json`, читаемой существующим `ManifestAdapter`.

## ADDED Requirements

### Requirement: Rasterize colony boxes to binary masks

Система SHALL растризовывать боксы колоний из COCO-аннотаций (`datasets/22022540/annot_COCO.json`) в бинарные маски размера исходного изображения, где пиксели колоний = 255, фон = 0. Маски SHALL сохраняться рядом с изображениями и попадать в манифест как пары `image` + `mask`.

#### Scenario: Mask matches image size
- **КОГДА** импортируется изображение `spXX_imgYY.jpg` размером W×H с N боксами колоний
- **ТОГДА** создаётся маска размера W×H, где заливка каждого бокса даёт 255, остальное 0, и `N >= 1`

#### Scenario: Image without colonies skipped
- **КОГДА** у изображения в аннотациях нет ни одного бокса
- **ТОГДА** файл маски не создаётся, а запись о таком изображении SHALL отсутствовать в манифесте

### Requirement: Import dish geometry metadata

Система SHALL сохранять в манифест геометрию чашки Петри (`dish`: `cx`, `cy`, `r`), вычисляемую автоматически по контуру чашки на исходном изображении, для записей `kind=source`.

#### Scenario: Dish geometry attached to records
- **КОГДА** импортируется полноразмерное изображение чашки
- **ТОГДА** запись `kind=source` SHALL содержать `dish.{cx,cy,r}` внутри границ изображения

### Requirement: Produce cropped variant

Система SHALL опционально формировать вариант `kind=cropped`: обрезанное по чашке изображение и соответствующая маска с чёрным фоном вне круга (модель по образцу пайплайна разметки), и помечать записи как `cropped`.

#### Scenario: Cropped records created
- **КОГДА** включён режим создания обрезков
- **ТОГДА** для каждой исходной записи дополнительно создаются `{stem}_cropped` (изображение + маска) с `kind=cropped`

#### Scenario: Cropping respects dish radius
- **КОГДА** создаётся обрезок
- **ТОГДА** вне круга с центром `(cx,cy)` и радиусом `r` пиксели SHALL быть чёрными

### Requirement: Materialize self-contained dataset

Система SHALL материализовать результаты в самодостаточную папку со структурой, читаемой `ManifestAdapter`: файлы `image`/`mask` физически скопированы (`storage="copy"`), а в корне папки создан валидный `dataset.json` со схемой `training-data-contract`, `mask_mode="binary"`, `origin="external"`.

#### Scenario: dataset.json readable by ManifestAdapter
- **КОГДА** импорт завершён
- **ТОГДА** `load_manifest(<output_dir>)` SHALL вернуть манифест со всеми записями, используя `ManifestAdapter`

#### Scenario: Output path is deterministic
- **КОГДА** источник и параметры импорта не меняются
- **ТОГДА** повторный импорт в один и тот же выходной каталог SHALL давать одинаковый набор записей и не дублировать файлы

### Requirement: CLI entrypoint

Система SHALL предоставлять CLI `import-22022540` с параметрами: путь к источнику (`datasets/22022540`), выходной каталог, флаг `--crop` для варианта `cropped`, чтобы запускать импорт без GUI.

#### Scenario: CLI runs import
- **КОГДА** `uv run import-22022540 --data-root datasets/22022540 --output <dir> --crop` запускается из окружения `.venv-full`
- **ТОГДА** создаётся самодостаточный датасет в `<dir>` с `dataset.json` и масками

#### Scenario: CLI reports summary
- **КОГДА** импорт завершается успешно
- **ТОГДА** CLI SHALL вывести количество обработанных изображений, созданных записей (source/cropped) и путь к `dataset.json`