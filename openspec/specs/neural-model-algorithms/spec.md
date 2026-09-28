# neural-model-algorithms Specification

## Purpose
Поддержка нейросетевых алгоритмов распознавания колоний: универсальный ONNX-адаптер с инференсом через ONNX Runtime, загрузка внешних и встроенных (bundled) весов моделей с регистрацией в общем реестре алгоритмов.
## Requirements
### Requirement: Neural model algorithm adapter

Система SHALL предоставлять универсальный адаптер `OnnxModelAlgorithm`, реализующий `BaseDetectionAlgorithm`, для запуска ONNX-моделей как алгоритмов детекции. Адаптер SHALL принимать путь к `.onnx`-файлу, размер входного изображения (`img_size`, по умолчанию 512) и порог бинаризации (`threshold`, по умолчанию 0.5). Метод `detect(image, is_cropped=False)` SHALL выполнять предобработку (BGR→RGB, resize до `img_size`, нормализация по mean/std ImageNet), инференс через ONNX Runtime и возвращать бинарную `uint8`-маску размера входного изображения.

#### Scenario: Detect returns mask
- **КОГДА** `detect(image)` вызывается для BGR-изображения с загруженной валидной ONNX-моделью
- **ТОГДА** возвращается маска `ndim=2`, `dtype=uint8`, shape совпадает с входным изображением

#### Scenario: Threshold applied
- **КОГДА** выход модели порогово обрабатывается с `threshold=0.5`
- **ТОГДА** значения маски SHALL быть только 0 или 255

#### Scenario: Missing onnxruntime handled
- **КОГДА** `onnxruntime` не установлен в окружении
- **ТОГДА** адаптер SHALL выбросить понятную ошибку с подсказкой установки (без падения всего приложения)

### Requirement: External model weights loading

Система SHALL загружать веса модели из внешнего файла `.onnx`, путь к которому указывается пользователем (GUI — файловый диалог, CLI — параметр). Такая модель SHALL регистрироваться в реестре динамически во время выполнения, не требуя пересборки исполняемого файла.

#### Scenario: External model registered at runtime
- **КОГДА** пользователь выбирает внешний файл `weights.onnx` в GUI
- **ТОГДА** соответствующий алгоритм появляется в списке алгоритмов и может быть запущен в прогоне

#### Scenario: Invalid model file reported
- **КОГДА** указанный файл не существует или не является валидной ONNX-моделью
- **ТОГДА** GUI SHALL показать сообщение об ошибке и не регистрировать алгоритм

### Requirement: Bundled model weights

Система SHALL поддерживать встроенные (bundled) веса моделей, упакованные в исполняемый файл при сборке Nuitka. Встроенные модели SHALL регистрироваться в реестре автоматически при запуске приложения наравне с классическими алгоритмами.

#### Scenario: Bundled model registered at startup
- **КОГДА** приложение запущено и содержит встроенные `.onnx`-файлы
- **ТОГДА** соответствующие алгоритмы SHALL присутствовать в `list_algorithms()` без действий пользователя

#### Scenario: Bundled model absent at runtime
- **КОГДА** встроенных моделей нет (например, запуск из исходников без файлов)
- **ТОГДА** приложение SHALL работать без ошибок, просто не показывая эти алгоритмы

### Requirement: CPU-only ONNX Runtime session

Сессии ONNX Runtime для нейросетевых моделей SHALL использовать `CPUExecutionProvider` и SHALL NOT зависеть от GPU/CUDA-провайдеров. Если модель повреждена или CPU provider недоступен, приложение SHALL показать понятную ошибку и сохранить возможность пользоваться классическими алгоритмами.

#### Scenario: Inference uses CPU provider
- **WHEN** нейросетевой алгоритм инициализирует валидную ONNX-модель
- **THEN** session SHALL использовать CPU provider независимо от доступности ускорителей

#### Scenario: CPU model is unavailable
- **WHEN** runtime не может загрузить ONNX-модель или CPU provider
- **THEN** приложение SHALL вернуть понятную диагностическую ошибку и SHALL NOT завершить весь процесс

### Requirement: Overlapping full-resolution inference for bundled model

Система SHALL предоставлять для новой bundled-модели tiled inference полного входного кадра без resize исходника. Тайлы SHALL иметь размер 512×512, шаг 384 пикселя и дополнение на правой/нижней границах; выходные вероятности SHALL усредняться в зонах перекрытия и бинаризоваться порогом 0.5 после склейки. Результат SHALL быть `uint8`-маской 0/255 исходного размера. Операция SHALL выдавать progress callback с индексом/количеством тайлов.

#### Scenario: Tiled detector covers non-divisible image dimensions
- **WHEN** кадр не кратен шагу/размеру тайла
- **THEN** крайние пиксели SHALL покрываться дополненными тайлами, а выходная маска SHALL иметь исходную форму

#### Scenario: Overlap is merged as probabilities
- **WHEN** область пикселей предсказана несколькими соседними тайлами
- **THEN** вероятности предсказаний SHALL усредняться до бинаризации, чтобы не создавать швов на границах частей изображения

#### Scenario: Tile progress is emitted
- **WHEN** tiled inference обрабатывает изображение
- **THEN** progress callback SHALL сообщать продвижение обработки вплоть до общего количества тайлов
