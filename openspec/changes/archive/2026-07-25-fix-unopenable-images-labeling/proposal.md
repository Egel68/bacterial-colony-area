## Why

В режиме разметки (Labeling) некоторые изображения не открываются — выпадает ошибка «Не удалось загрузить файл», хотя эти же файлы открываются в стандартных редакторах изображений. Причина: `QPixmap(path)` из PyQt6 не поддерживает некоторые варианты форматов (CMYK JPEG, TIFF с LZW/Deflate-сжатием, progressive JPEG и др.), в то время как OpenCV (`cv2.imread`) справляется с ними.

Пользователи не могут размечать такие изображения, что замедляет сбор ground-truth данных.

## What Changes

- Добавить fallback-загрузку через `cv2.imread` в `utils/image_loader.py`, если `QPixmap` вернул `isNull()`.
- Добавить fallback-загрузку через `cv2.imdecode` (из буфера) для `load_image_grayscale`.
- Улучшить сообщения об ошибках — указывать, какая именно попытка загрузки не удалась и какой формат у файла.
- Убедиться, что `_on_auto_detect` и `_on_crop` в `labeling_window.py` показывают пользователю ошибку вместо silent return.
- (Опционально) Добавить поддержку формата PNG с палитрой (индексированные цвета).

## Capabilities

### New Capabilities
- `robust-image-loading`: Надёжная загрузка изображений с fallback-механизмом (QPixmap → OpenCV), поддержка широкого спектра форматов и кодеков.

### Modified Capabilities
<!-- Нет существующих specs в проекте, все capabilities новые -->

## Impact

- `utils/image_loader.py` — основной файл изменений
- `labeling/labeling_window.py` — исправление silent return в `_on_auto_detect` и `_on_crop`
- Зависимости: OpenCV (`cv2`) уже используется в проекте, новых зависимостей не требуется
