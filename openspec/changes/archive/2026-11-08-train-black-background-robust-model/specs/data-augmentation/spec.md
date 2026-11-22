# Spec Delta

## ADDED Requirements

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
