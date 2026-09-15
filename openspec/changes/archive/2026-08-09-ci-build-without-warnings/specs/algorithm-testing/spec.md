## MODIFIED Requirements

### Requirement: Test dataset loading

Система SHALL загружать парные данные из `test_images/`: для каждой пары из `source/` + `masks/` строится sample с `source_image`, `source_mask`; при наличии пары в `cropped/` + `cropped_masks/` — также `cropped_image` и `cropped_mask`. Отсутствующие пары SHALL пропускаться. Загрузчик (`TestDataset`) SHALL NOT быть распознан pytest как тестовый класс: он SHALL иметь `__test__ = False`, чтобы запуск тестов SHALL NOT порождать `PytestCollectionWarning`.

#### Scenario: Pair with cropped variant
- **КОГДА** в датасете есть source+mask и cropped+cropped_mask
- **ТОГДА** `TestDataset` SHALL содержать sample с заполненными cropped-полями

#### Scenario: Loader not collected by pytest
- **КОГДА** pytest собирает тесты (`tests/`)
- **ТОГДА** класс `TestDataset` в `testing/dataset.py` SHALL NOT коллектироваться как тестовый, и в сводке SHALL NOT появляться `PytestCollectionWarning: cannot collect test class 'TestDataset'`