## ADDED Requirements

### Requirement: CPU-only ONNX Runtime session

Нейросетевые модели приложения SHALL создавать ONNX Runtime session с `CPUExecutionProvider` и SHALL NOT зависеть от GPU/CUDA-провайдера. Если CPU provider недоступен или модель повреждена, приложение SHALL сообщить понятную ошибку, не прекращая работу классических алгоритмов.

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
- **THEN** вероятности SHALL объединяться до бинаризации, чтобы скрыть швы между тайлами

#### Scenario: Tile progress is emitted
- **WHEN** tiled inference обрабатывает изображение
- **THEN** progress callback SHALL сообщать продвижение обработки вплоть до общего количества тайлов
