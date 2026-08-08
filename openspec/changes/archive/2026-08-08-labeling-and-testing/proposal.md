## Why

Режим ручной разметки (Labeling) и фреймворк тестирования алгоритмов реализованы и используются, но их контракты не задокументированы в спеках: нет capability `image-labeling` и `algorithm-testing`. Это мешает ревью, регрессионному тестированию и воспроизводимости сравнения алгоритмов.

## What Changes

- Добавить capability `image-labeling`: ручная разметка масок, управление папками сессии (source/cropped, masks/cropped_masks), обрезка по чашке, сохранение/загрузка масок, экспорт сессии в ZIP, фильтрация файлов по `SUPPORTED_EXTENSIONS`.
- Добавить capability `algorithm-testing`: реестр алгоритмов (`@register_algorithm`), интерфейс `BaseDetectionAlgorithm`, 4 классических алгоритма, загрузка тестовых пар `TestDataset`, метрики (IoU/Dice/F1/Precision/Recall/Accuracy), прогон `run_all`, HTML-отчёт.
- Зафиксировать сценарии и сопоставить их с существующим кодом и тестами (reverse documentation, без изменения логики).

## Capabilities

### New Capabilities
- `image-labeling`: Ручная разметка изображений чашек Петри: список файлов по допустимым расширениям, папки сессии (source/cropped), обрезка по чашке с заливкой фона, сохранение/загрузка масок, экспорт сессии в ZIP.
- `algorithm-testing`: Фреймворк сравнения алгоритмов детекции на ground-truth: реестр алгоритмов, интерфейс, метрики, прогон на датасете, HTML-отчёт.

### Modified Capabilities
<!-- Нет существующих specs, все capabilities новые -->

## Impact

- `ui/controllers/labeling_controller.py`, `labeling/labeling_window.py`
- `testing/` (interface, registry, classic_algorithms, dataset, metrics, runner, dashboard, __main__)
- `utils/image_loader.py` — через `SUPPORTED_EXTENSIONS`
- Внешних зависимостей не добавляется.