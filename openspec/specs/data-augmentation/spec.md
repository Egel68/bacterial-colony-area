# data-augmentation Specification

## Purpose
TBD - created by archiving change ml-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Augmentation pipeline

Система SHALL генерировать аугментированные пары изображение-маска из `test_images/cropped/` и `test_images/cropped_masks/`, выдавая `AUGMENTATIONS_PER_IMAGE = 15` вариантов на каждый исходник, у которого есть маска. Система SHALL применять геометрические преобразования (`Rotate(180°)`, `HorizontalFlip`, `VerticalFlip`, `Affine` со scale 0.9–1.1 и translate ±5%) и пиксельные (`RandomBrightnessContrast`, `GaussNoise`, `Blur`, `HueSaturationValue`) синхронно к изображению и маске (`additional_targets={"mask": "image"}`). Результат SHALL записываться в `train/data/images/` и `train/data/masks/` парами с одинаковыми именами `{stem}_aug_{index:03d}.png`, при этом маска SHALL бинаризоваться порогом `>127`.

#### Scenario: 15 пар на исходник с синхронными путями
- **КОГДА** в `test_images/cropped/` есть один исходник с парной маской
- **ТОГДА** для каждого из 15 индексов SHALL быть создана пара `images/{stem}_aug_000.png` + `masks/{stem}_aug_000.png` (индексы 000..014), имя-маска в `masks/` SHALL совпадать с именем-картинкой в `images/`

#### Scenario: Маска без пары пропускается
- **КОГДА** для исходника не существует `_{stem}_mask.png`
- **ТОГДА** система SHALL пропустить этот исходник и не создавать для него пары

### Requirement: Аугментационный seed

Система SHALL использовать глобальный `SEED = 42`, а индекс пары задаёт подсеются для `random.seed(SEED + index)`, обеспечивая воспроизводимость набора.

#### Scenario: Reproducibility with same seed
- **КОГДА** аугментация прогоняется дважды с одним и тем же seed
- **ТОГДА** результаты (последовательности индексов, имена файлов) SHALL совпадать

### Requirement: Black-background image variants
Для каждого исходного sample 22022540 с валидной геометрией чашки система SHALL создавать один полноразмерный вариант, заменяя чёрным только пиксели за пределами круга чашки. Пиксели внутри круга, исходное изображение и бинарная GT-маска SHALL оставаться неизменными; маска варианта SHALL совпадать с исходной попиксельно.

#### Scenario: Variant changes only the exterior background
- **WHEN** обрабатывается source sample с изображением, бинарной маской и геометрией чашки
- **THEN** создаётся изображение того же размера с чёрными пикселями вне круга, неизменёнными пикселями внутри круга и маской, попиксельно равной исходной

#### Scenario: Missing dish geometry is reported
- **WHEN** выбранный для преобразования source sample не содержит валидной геометрии чашки
- **THEN** подготовка SHALL завершиться ошибкой с ID sample и SHALL NOT сообщать об успешном создании полного датасета после молчаливого пропуска sample

### Requirement: Variant provenance and split inheritance
Каждый чёрнофоновый вариант SHALL быть записан как kind=augmented со стабильным ID, связанным с ID исходника, и SHALL наследовать его subset. Разбиение SHALL быть закреплено по ID исходников до генерации вариантов; при одинаковых входе и seed состав групп и идентификаторы вариантов SHALL быть воспроизводимыми.

#### Scenario: Variants remain in their source subset
- **WHEN** исходный sample получает subset train, val или test до генерации чёрного варианта
- **THEN** исходная запись и её чёрнофоновый вариант SHALL иметь один и тот же subset и SHALL NOT попадать в разные выборки

#### Scenario: Repeated augmentation is reproducible
- **WHEN** подготовка выполняется повторно для тех же исходных данных с тем же seed
- **THEN** соответствия source ID → augmented ID и subset SHALL совпадать

### Requirement: Complete paired dataset before evaluation or training
Для 22022540 pipeline SHALL полностью материализовать все исходные записи и по одному чёрнофоновому варианту на source ID, зафиксировать split до вариантов и проверить парность, маски, файлы и subset-наследование до запуска baseline evaluation или обучения. Неполный или невалидный dataset SHALL приводить к ошибке до оценки/обучения, а не к частичному запуску.

#### Scenario: Training starts only after complete materialization
- **WHEN** запуск black-background experiment получает размеченный COCO dataset
- **THEN** baseline evaluation и обучение SHALL начинаться только после проверки полного набора source ID и соответствующих augmented ID, причем каждая пара SHALL иметь один subset

#### Scenario: Reuse validates the complete prepared dataset
- **WHEN** явно запрошен `--reuse-prepared-dataset` для уже созданного output-каталога
- **THEN** pipeline SHALL до baseline evaluation проверить seed/ratios, покрытие всех COCO source ID, наличие файлов, ровно один вариант на источник и совпадение subset и маски; неполный/несовпадающий dataset SHALL быть отклонён

