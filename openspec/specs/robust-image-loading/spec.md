# Robust Image Loading

## Purpose

Надёжная загрузка изображений в режиме разметки с fallback-механизмом (QPixmap → OpenCV), поддержка широкого спектра форматов и кодеков. Если стандартный путь через Qt не может декодировать файл, система обязана (SHALL) попробовать OpenCV, прежде чем сообщить об ошибке.

## Requirements

### Requirement: Fallback image loading with QPixmap → OpenCV

Система SHALL сначала пытаться загрузить изображение через `QPixmap`. **Любой сбой** в цепочке QPixmap → QImage → numpy считается сбоем загрузки и SHALL запускать fallback на OpenCV. К сбоям относятся (но не ограничиваясь): `QPixmap::isNull()`, `QImage::isNull()`, `bytesPerLine != width × bytesPerPixel` (выравнивание строк Qt), ошибки reshape в numpy, ошибки доступа к памяти.

Если путь через QPixmap не сработал, система SHALL пробовать `cv2.imread` для того же пути. Fallback на OpenCV SHALL выполняться независимо от того, какой именно сбой произошёл в пути QPixmap.

#### Scenario: Image loads successfully via QPixmap
- **КОГДА** вызывается `load_image(path)` с корректным путём к стандартному PNG/JPEG, ширина которого не вызывает выравнивания строк (то есть `bytesPerLine == width × 3`)
- **ТОГДА** функция SHALL вернуть изображение как BGR numpy-массив, загруженный через QPixmap

#### Scenario: Image fails QPixmap but loads via OpenCV fallback
- **КОГДА** вызывается `load_image(path)` с корректным путём к файлу, который QPixmap не может декодировать (например, CMYK JPEG, TIFF с LZW-сжатием)
- **ТОГДА** функция SHALL вызвать `cv2.imread` и, при успехе, вернуть изображение как BGR numpy-массив

#### Scenario: Image fails due to QImage row-alignment padding, OpenCV succeeds
- **КОГДА** вызывается `load_image(path)` с корректным PNG, ширина которого даёт `bytesPerLine > width × 3` после `convertToFormat(RGB888)` (например, width=870 → bytesPerLine=2612 против 3×870=2610)
- **ТОГДА** путь QPixmap SHALL завершиться ошибкой, и функция SHALL успешно перейти к fallback на `cv2.imread`

#### Scenario: Any exception from the QPixmap path triggers fallback
- **КОГДА** вызывается `load_image(path)` и любое исключение (ValueError, RuntimeError и т. п.) возникает при конвертации QPixmap → QImage → numpy
- **ТОГДА** исключение SHALL быть перехвачено, путь QPixmap SHALL быть отброшен, и SHALL выполняться попытка `cv2.imread`

#### Scenario: Both QPixmap and OpenCV fail to load
- **КОГДА** оба способа загрузки не срабатывают
- **ТОГДА** функция SHALL бросить `ValueError` с описательным сообщением, включающим путь и указание, что оба способа не сработали

### Requirement: Robust numpy conversion from QImage

Система SHALL корректно читать пиксельные данные из `QImage`, даже когда присутствует выравнивание строк (`bytesPerLine > width × bytesPerPixel`). Numpy-массив SHALL строиться построчно с шагом `bytesPerLine`, извлекая ровно `width` пикселей в строке.

#### Scenario: QImage with row padding converts correctly
- **КОГДА** QImage имеет `bytesPerLine > width × 3` из-за выравнивания строк Qt
- **ТОГДА** numpy-массив SHALL содержать ровно `height × width × 3` элементов, без байтов выравнивания

### Requirement: Fallback image loading for grayscale

Функция `load_image_grayscale` SHALL использовать тот же механизм fallback: сначала QPixmap, затем `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`. Правило «любой сбой запускает fallback» SHALL применяться так же.

#### Scenario: Grayscale fallback loads successfully
- **КОГДА** вызывается `load_image_grayscale(path)` и QPixmap не срабатывает, но OpenCV успешен
- **ТОГДА** функция SHALL вернуть одноканальный uint8-массив, загруженный через OpenCV в градациях серого

#### Scenario: Grayscale QImage with row padding falls back
- **КОГДА** вызывается `load_image_grayscale(path)` и данные QPixmap имеют `bytesPerLine > width × 1` после `convertToFormat(Grayscale8)`
- **ТОГДА** функция SHALL перейти к fallback на `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`

### Requirement: User-visible error on auto-detect and crop failure

Окно разметки SHALL показывать предупреждение (dialog) когда `_on_auto_detect` или `_on_crop` не могут загрузить текущее изображение, вместо молчаливого возврата.

#### Scenario: Auto-detect with unloadable image
- **КОГДА** пользователь нажимает «🔍 Авто-поиск» и текущее изображение не может быть загружено
- **ТОГДА** SHALL показываться `QMessageBox.warning`: «Не удалось загрузить {path.name}»

#### Scenario: Crop with unloadable image
- **КОГДА** пользователь нажимает «✂️ Обрезать по чашке» и текущее изображение не может быть загружено
- **ТОГДА** SHALL показываться `QMessageBox.warning`: «Не удалось загрузить {path.name}»

### Requirement: Supported image formats

Система SHALL поддерживать как минимум следующие форматы: PNG, JPEG, JPG, BMP, TIFF, TIF, WebP. Дополнительные форматы, поддерживаемые OpenCV (например, PXM, JPEG 2000), SHALL также работать через путь fallback. Загрузчик SHALL публиковать список допустимых расширений как `SUPPORTED_EXTENSIONS` в `utils/image_loader.py`, и фильтрация списка файлов в интерфейсе SHALL использовать именно этот единый источник.

#### Scenario: Open unsupported format
- **КОГДА** выбран файл с расширением, отсутствующим в `SUPPORTED_EXTENSIONS` (например, `.gif`, `.svg`)
- **ТОГДА** файл SHALL NOT появляться в списке файлов
