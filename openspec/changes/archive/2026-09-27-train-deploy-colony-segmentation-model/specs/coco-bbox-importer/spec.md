## ADDED Requirements

### Requirement: Optional deterministic train/validation/test assignments

`CocoBboxImporter` SHALL поддерживать режим подготовки воспроизводимого split: принимать seed и доли выборок (по умолчанию seed 42 и приблизительно 70/15/15), назначать subset группам исходных COCO-изображений со стратификацией по категории, и записывать его во все производные записи этой группы. Выходной `dataset.json` SHALL оставаться читаемым `ManifestAdapter`; отдельные метаданные SHALL фиксировать seed, доли и список идентификаторов групп в каждой выборке. При неизменном источнике и параметрах повторная подготовка SHALL давать те же назначения и SHALL NOT изменять входные файлы. Обычный импорт без режима split SHALL сохранять текущую совместимость.

#### Scenario: Split is deterministic and persisted
- **WHEN** COCO-источник дважды импортируется с одинаковыми долями и seed
- **THEN** назначения subset и метаданные split SHALL совпадать и оставаться читаемыми через `load_manifest`

#### Scenario: Source and derived variants share assignment
- **WHEN** для COCO-изображения создаются source- и cropped-записи с включённым split
- **THEN** обе записи SHALL иметь один subset, определённый исходным COCO-изображением

#### Scenario: Split import keeps source read-only
- **WHEN** COCO-импорт выполняется с разбиением
- **THEN** файлы исходных изображений и аннотаций SHALL остаться неизменными, а все новые данные SHALL быть записаны в `output_dir`
