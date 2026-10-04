## ADDED Requirements

### Requirement: Image decoding is safe to run in a background thread

Загрузка полноразмерных изображений вне GUI-потока SHALL использовать потокобезопасный decoder и SHALL NOT вызывать `QPixmap` в worker. Декодированные цветовые изображения SHALL сохранять публичный контракт BGR `uint8`-массива, grayscale-загрузка — одноканальный `uint8`; при ошибке основного worker-safe decoder SHALL использоваться fallback на OpenCV до сообщения об ошибке. Существующий порядок QPixmap → OpenCV для вызовов, выполняемых в GUI-потоке, SHALL оставаться совместимым. Порядок попыток должен сохранять историческую способность декодировать все уже поддержанные файлы: fallback на OpenCV используется, если worker-safe Qt reader недоступен или не справился.

#### Scenario: Background worker decodes a supported image
- **WHEN** PNG, JPEG, BMP, TIFF или WebP загружается worker-потоком
- **THEN** decoder SHALL вернуть тот же shape/channel/dtype/color-order контракт, а worker SHALL NOT вызывать `QPixmap`

#### Scenario: Background decoder fails but OpenCV can load the file
- **WHEN** worker-safe Qt decoder не может загрузить допустимый файл, но OpenCV может
- **THEN** система SHALL выполнить OpenCV fallback и вернуть корректный массив вместо ошибки

#### Scenario: Both background decoding paths fail
- **WHEN** worker-safe Qt decoder и OpenCV не могут загрузить файл
- **THEN** операция SHALL вернуть описательную ошибку, которую UI может показать после возврата результата в GUI-поток
