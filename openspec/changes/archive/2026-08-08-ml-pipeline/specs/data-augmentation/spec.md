## ADDED Requirements

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