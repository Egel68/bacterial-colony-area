## 1. Обновление SPEC и DESIGN

- [x] 1.1 Дополнить spec требованием: любая ошибка в цепочке QPixmap→QImage→numpy (включая reshape от row padding) — fallback
- [x] 1.2 Обновить spec: добавить scenario про QImage row-alignment padding
- [x] 1.3 Обновить design: блок QPixmap→numpy заворачивается в try/except Exception

## 2. Исправление core fallback в utils/image_loader.py

- [x] 2.1 `_load_image_pixmap`: обернуть блок QPixmap→QImage→reshape→copy в try/except Exception → return None при любой ошибке; исправить reshape на использование bytesPerLine
- [x] 2.2 `_load_image_grayscale_pixmap`: тот же try/except Exception + bytesPerLine
- [x] 2.3 `load_image`: оставить как есть (уже правильно: вызывает _load_image_pixmap, если None → _load_image_opencv)
- [x] 2.4 Проверить, что reshape с bytesPerLine работает для padded (870, 873) и unpadded (824, 868) изображений

## 3. Исправление silent error handling в labeling_window.py

- [x] 3.1 `_on_auto_detect`: silent return → QMessageBox.warning
- [x] 3.2 `_on_crop`: silent return → QMessageBox.warning

## 4. Верификация

- [x] 4.1 Запустить тест с файлами, у которых padding (ширины 870, 873): reshape не падает, все 6/6 изображений загружаются
- [x] 4.2 Проверить, что файлы без padding (ширины 824, 868) всё ещё грузятся через QPixmap (без регрессии)
- [x] 4.3 Проверить, что auto-detect и crop показывают диалог при ошибке, а не silent return (ранее проверено)
