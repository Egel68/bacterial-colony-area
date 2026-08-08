## Context

Изображения в режиме разметки загружаются через `QPixmap(path)` → `QImage` → `numpy`. Есть две независимые причины отказа:

1. **QPixmap не декодирует формат** (CMYK JPEG, LZW TIFF и т.д.) — Qt не использует libjpeg-turbo/libpng, а свои плагины.
2. **QImage имеет row-alignment padding** — Qt выравнивает строки пикселов до 4 байт. Если `qimage.bytesPerLine() != width × 3`, то `np.frombuffer(ptr).reshape((h, w, 3))` падает с `ValueError`, потому что размер буфера не совпадает с ожидаемым.

OpenCV (`cv2.imread`) использует libjpeg-turbo, libtiff, libpng и не имеет проблемы row padding (возвращает плотный массив). OpenCV уже есть в зависимостях проекта.

## Goals / Non-Goals

**Goals:**
- Все изображения, которые открываются в стандартных редакторах, должны открываться и в режиме разметки
- Прозрачность для пользователя: fallback не требует ручных действий
- Сообщения об ошибках должны быть информативными
- Любая ошибка в цепочке QPixmap → QImage → numpy (включая reshape из-за padding) приводит к fallback, а не к падению

**Non-Goals:**
- Добавление новых форматов (GIF, SVG, HEIC, RAW и т.д.) — только поддержка существующих через fallback
- Изменение пайплайна загрузки в анализе (analysis_window.py) — там отдельный код с `load_image`, он тоже получит fallback автоматически
- Переработка архитектуры загрузки изображений

## Decisions

1. **Любая ошибка QPixmap → fallback**
   - Весь блок загрузки через QPixmap/convertToFormat/reshape оборачивается в try/except Exception.
   - При ЛЮБОМ исключении (ValueError от reshape, RuntimeError от Qt и т.д.) — возвращаем None, переходим к OpenCV.
   - `cv2.imread` при ошибке возвращает `None` — проверяем и кидаем `ValueError` только если оба метода вернули null.

2. **Fallback в `load_image_grayscale`**
   - То же самое: QPixmap → `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`.

3. **Обработка ошибок в labeling_window**
   - `_on_auto_detect` и `_on_crop`: заменить `except ValueError: return` на `except ValueError: QMessageBox.warning(...)`.

4. **Кэширования не требуется**
   - OpenCV читает с диска при каждом fallback — это нормально, т.к. загрузка происходит один раз при выборе файла.

## Risks / Trade-offs

- **[Производительность]** OpenCV читает файл заново при fallback → *Приемлемо: fallback срабатывает редко (только для проблемных файлов), загрузка однократная.*
- **[Цветовой профиль]** OpenCV не применяет ICM/ICC (цветовые профили) → *QPixmap их тоже не применяет, поведение не меняется.*
- **[Прозрачность (alpha)]** OpenCV загружает 3-канальный BGR, теряя альфа-канал → *QPixmap тоже не сохранял альфу (конвертация в RGB888), поведение не меняется.*
- **[EXIF-ориентация]** OpenCV не учитывает EXIF-поворот → *QPixmap тоже не учитывает; если потребуется — отдельная задача.*
- **[Широкий catch Exception]** try/except Exception скрывает ошибки программиста (например, ImportError) → *Применяется только к узкому блоку QPixmap→numpy, где известен полный набор возможных исключений.*
