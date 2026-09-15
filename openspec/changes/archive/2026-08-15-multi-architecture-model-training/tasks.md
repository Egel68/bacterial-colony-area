## 1. Контракт данных

- [x] 1.1 Создать `train/dataset_manifest.py`: схема манифеста (`name`, `origin`, `mask_mode`, `storage`, `samples` с `id`/`kind`/`image`/`mask`/`subset`/`dish`), dataclass-ы записей и манифеста.
- [x] 1.2 Валидация манифеста: пути относительны к `data_root`, файлы `image`/`mask` существуют, `mask_mode` принимает `binary` (задел: `multiclass` — понятная ошибка).
- [x] 1.3 Тесты/проверка: `load_manifest` собирает манифест из валидного `dataset.json`, отбрасывает записи без маски.

## 2. Адаптеры источников

- [x] 2.1 Реализовать `ManifestAdapter`, `LabelingAdapter`, `PairsAdapter` с единым интерфейсом `build(data_root) -> DatasetManifest` в `train/dataset_adapters.py`.
- [x] 2.2 Реализовать фабрику `load_manifest(data_root)`: приоритет `dataset.json` → сессия разметки (`source/`) → легаси (`images/`); при пустоте — `FileNotFoundError`.
- [x] 2.3 `LabelingAdapter`: сканирует `source/*` + `masks/{stem}_mask`, `cropped/*` + `cropped_masks/{stem}_cropped_mask`, `storage="reference"`, без копирования.
- [x] 2.4 Проверка: `load_manifest(test_images)` находит source+cropped пары; `load_manifest(train/data)` работает через `PairsAdapter`.

## 3. Чтение контракта в `make_datasets`

- [x] 3.1 Переписать `make_datasets`/`ColonyDataset` на контракт: загрузка через `load_manifest`, разделение по `subset` из манифеста (фолбэк — seed-сплит `val_split`).
- [x] 3.2 Проверка `mask_mode`: при `multiclass` — ошибка с понятным сообщением; при пустом наборе — `FileNotFoundError`.
- [x] 3.3 Убедиться, что выход датасета не изменился: RGB float32 `[C,H,W]` (mean/std ImageNet) + бинарная маска `[1,H,W]`, resize LINEAR/NEAREST.

## 4. Множественные архитектуры

- [x] 4.1 Добавить зарегистрированную CNN-архитектуру `unet_small` в `train/models/` (уменьшенные каналы, наследует `BaseSegmenter`).
- [x] 4.2 Прокинуть параметры архитектуры через `get_model(name, **params)` → конструктор (`n_channels`, `n_classes`, `features`).
- [x] 4.3 Убедиться, что `list_models()` возвращает `["unet", "unet_small"]`, а неизвестное имя бросает `ValueError`.

## 5. Метрики в реальном времени и устройство

- [x] 5.1 Подтвердить/оставить работу `progress_callback` и TensorBoard для новых архитектур в `run_training` (метрики train и val каждую эпоху).
- [x] 5.2 Убедиться, что подпись колбэка для compare-режима содержит имя архитектуры (без изменения сигнатуры ядра).
- [x] 5.3 Сохранить/покрыть выбор устройства: `cuda` при доступности, иначе `cpu`; `to_onnx` не падает на CPU.

## 6. Конвейер обучения и оценки нескольких архитектур

- [x] 6.1 Создать `train/compare.py` с `run_train_compare(architectures, cfg, data_root, eval_root)`: для каждой архитектуры клонировать конфиг (`model_name`), вызвать `run_training`, собрать пути `checkpoints/best.onnx`.
- [x] 6.2 Оценка каждой `best.onnx` через `OnnxModelAlgorithm` + `run_algorithm` на `TestDataset(eval_root)` (по умолчанию `test_images`), усреднение IoU/Dice/Precision/Recall/F1 с указанием числа снимков.
- [x] 6.3 Сформировать сводный отчёт `compare.json` и `compare.html` в `train/runs/compare_{timestamp}/`.
- [x] 6.4 Добавить CLI-под-команду `train-compare` в `train/main.py`: `--architectures`, `--data-root`, `--eval-root`, общие параметры; поле `architectures` в `TrainingConfig`.

## 7. Документирование задела (импортёры, multiclass)

- [x] 7.1 Задокументировать в `AGENTS.md` идею контракта данных и импортёров внешних датасетов (COCO/VOC/произвольные пары): интерфейс `@register_importer`, стратегии `reference`/`copy`, `mask_mode` `multiclass` — как задел на будущее.
- [x] 7.2 Оставить в коде (`load_manifest`, схема манифеста) комментарии-точки расширения для импортёров и multiclass, чтобы задел не терялся.

## 8. Проверка и качество

- [x] 8.1 Ручной прогон `uv run train.main train-compare --architectures unet,unet_small --epochs 2` на `test_images/` — run-папки и отчёты создаются, ошибок нет.
- [x] 8.2 Нерегресс: `make_datasets(train/data, ...)` (легаси `PairsAdapter`) и `--model unet` работают как раньше.