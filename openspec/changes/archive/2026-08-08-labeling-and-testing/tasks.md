## 1. Сверка спеки `image-labeling` с кодом

- [x] 1.1 Проверено: `LabelingController.SUPPORTED_EXTENSIONS` перенесён в `utils/image_loader.py`; `list_image_files` фильтрует по нему без учёта регистра
- [x] 1.2 Проверено: `get_current_dir`/`get_mask_dir` выбирают source/cropped и masks/cropped_masks с фоллбеком на корень сессии
- [x] 1.3 Проверено: `crop_by_petri` обрезает по чашке и заливает фон чёрным, с клампингом к границам
- [x] 1.4 Проверено: `save_mask` создаёт родительскую директорию; `load_mask` возвращает `None` при несоответствии shape

## 2. Тесты на сценарии `image-labeling` (tests/test_labeling_controller.py)

- [x] 2.1 `list_image_files` фильтрует `.gif`/`.svg`
- [x] 2.2 Только `.gif` → пустой список
- [x] 2.3 Директория отсутствует → пустой список
- [x] 2.4 `get_current_dir` выбор source/ vs фоллбэк корень
- [x] 2.5 `get_mask_dir` выбор папок масок
- [x] 2.6 `crop_by_petri` форма `2r×2r` и чёрный фон
- [x] 2.7 `crop_by_petri` ограничение у границ
- [x] 2.8 `export_session_to_zip` содержит все файлы

## 3. Тесты на сценарии `algorithm-testing`

- [x] 3.1 Существующие `test_registry.py` (register/get/list/errors) покрывают реестр
- [x] 3.2 `test_each_returns_mask` (не-cроpped) и новый `test_each_returns_cropped_mask` покрывают интерфейс `detect(..., is_cropped=True)`
- [x] 3.3 `test_metrics.py` покрывают perfect prediction и disjoint
- [x] 3.4 `test_mean_metrics` покрывает `_mean_metrics([])` и агрегацию
- [x] 3.5 Проверить, что `generate_report` создаёт файл (быстрый unit-тест)

## 4. Верификация

- [x] 4.1 `uv run pytest -q` — все тесты зелёные
- [x] 4.2 `uv run ruff check` — чистый код
- [x] 4.3 `openspec validate --all` — спеки валидны