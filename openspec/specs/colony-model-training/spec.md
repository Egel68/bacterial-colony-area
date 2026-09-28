# colony-model-training Specification

## Purpose

Определяет воспроизводимую подготовку скачанного датасета, обучение компактной сегментационной модели с контролем переобучения и итоговую оценку/export артефактов для CPU-инференса в приложении.

## Requirements

### Requirement: Reproducible source-level train/validation/test split

Пайплайн обучения SHALL формировать фиксированные выборки `train`, `val` и `test` до нарезки изображений на обучающие тайлы. По умолчанию целевые доли SHALL быть приблизительно 70/15/15, seed — 42. Для COCO-датасета разделение SHALL выполняться на уровне исходного изображения/чашки и стратифицироваться по категории бактерии; все представления одного исходника (включая cropped-варианты и производные тайлы) SHALL относиться только к одной выборке. При неизменных входных аннотациях, пропорциях и seed назначения групп SHALL быть одинаковыми при каждом запуске независимо от порядка перечисления файлов. Назначения, seed, пропорции и список идентификаторов SHALL сохраняться в метаданных датасета. Исходный датасет SHALL оставаться неизменным.

#### Scenario: Split is stable across repeated preparation
- **WHEN** подготовка выполняется повторно на тех же COCO-аннотациях с теми же параметрами и seed
- **THEN** каждая исходная чашка SHALL принадлежать той же выборке, а метаданные разбиения SHALL содержать те же назначения

#### Scenario: Derived records remain in their source split
- **WHEN** для исходного снимка создаётся cropped-вариант или обучающие тайлы
- **THEN** все производные записи SHALL сохранить subset исходного снимка и SHALL NOT пересекать границы train/val/test

#### Scenario: Source data is not modified
- **WHEN** запускается импорт и подготовка обучающего датасета
- **THEN** файлы источника и COCO-аннотации SHALL NOT изменяться; производные данные SHALL записываться только в выходной каталог

### Requirement: Test subset isolation and validation-based checkpoint selection

Обучение SHALL вычислять градиенты только на `train`, а выбор чекпоинта и early stopping SHALL выполнять только по `val`. Записи с `subset="test"` SHALL быть исключены из обучающих и validation DataLoader. После фиксации лучшего по validation чекпоинта система SHALL оценить test-набор отдельно и SHALL NOT использовать его метрики для настройки весов, выбора эпохи, порога или гиперпараметров.

#### Scenario: Test records are excluded from training
- **WHEN** обучающий манифест содержит записи `train`, `val` и `test`
- **THEN** ни один `test` sample SHALL NOT появиться в обучающем или validation DataLoader

#### Scenario: Test is evaluated only after model selection
- **WHEN** обучение завершено и лучший чекпоинт выбран по validation
- **THEN** система SHALL вычислить итоговые метрики на test отдельным шагом, не меняя выбранный чекпоинт

### Requirement: Compact pretrained colony segmenter

Система SHALL предоставлять регистрируемую PyTorch-архитектуру бинарной сегментации `mobilenet_v3_small_unet` с энкодером MobileNetV3-Small и лёгким сегментационным декодером. Специализированная команда обучения SHALL по умолчанию инициализировать энкодер опубликованными ImageNet-весами; SHALL быть возможность отключить pretraining для offline/контрольного запуска. Обучение SHALL использовать маски COCO, преобразованные импортёром в бинарные маски, и SHALL сохранять выбор архитектуры, настройки и seed в отчёте запуска.

Обучение SHALL использовать CUDA/GPU, если CUDA-устройство доступно, и SHALL поддерживать CPU fallback при отсутствии GPU. Это не меняет deployment-контракт: проверка ONNX и производственный инференс SHALL выполняться только через ONNX Runtime `CPUExecutionProvider`, без зависимости на CUDA или PyTorch.

#### Scenario: Pretrained architecture is available
- **WHEN** `mobilenet_v3_small_unet` запрашивается через реестр архитектур
- **THEN** модель SHALL создаваться для одноканального бинарного выхода и SHALL поддерживать ImageNet-pretrained encoder

#### Scenario: Training can run without pretrained download
- **WHEN** пользователь отключает pretrained initialization
- **THEN** обучение SHALL стартовать без сетевой загрузки ImageNet-весов

### Requirement: Overfit-aware native-resolution patch training

Новая модель SHALL обучаться на patch-тайлах исходного разрешения 512×512, не масштабируя всю чашку до 512×512 перед обучением. Позиции обучающих patch-тайлов SHALL варьироваться между эпохами, а validation patches SHALL выбираться воспроизводимо и не использовать аугментации. Пайплайн SHALL применять регуляризацию/аугментации и ограничение обучения по validation (early stopping или эквивалент), чтобы снижать переобучение на небольшом наборе из 369 исходников. Для конфигурации по умолчанию SHALL быть предусмотрена цель — не более 60 секунд на эпоху на RTX 2060 6 GB; фактическое время эпох SHALL записываться, а число patch-итераций и batch size SHALL настраиваться для соблюдения этого бюджета без изменения split.

#### Scenario: Training preserves colony pixel scale
- **WHEN** исходный снимок имеет размер больше 512×512
- **THEN** модель SHALL обучаться на его исходных 512×512 patch-тайлах, а не на полном снимке, уменьшенном до 512×512

#### Scenario: Validation controls early stopping
- **WHEN** validation-метрика перестаёт улучшаться в течение настроенного patience
- **THEN** обучение SHALL завершиться досрочно и оставить чекпоинт с лучшей validation-метрикой

#### Scenario: Epoch timing is recorded
- **WHEN** завершается эпоха обучения
- **THEN** длительность SHALL попасть в историю запуска, а доступный benchmark SHALL позволять проверить целевой бюджет ≤60 секунд на RTX 2060

### Requirement: Reproducible training command and evaluation artifacts

Система SHALL предоставлять CLI-команду, которая по скачанному COCO-датасету выполняет split и импорт масок, обучает выбранную компактную модель, выбирает checkpoint только по validation, экспортирует его в отдельный ONNX-файл и оценивает модель на held-out test. Исходная модель `models/colony_seg.onnx` SHALL оставаться нетронутой; новый артефакт SHALL иметь отдельное имя и путь вывода. Отчёт SHALL включать train/validation/test IoU и Dice, precision и recall, seed и ID разделённых групп, архитектуру и настройки обучения, номер лучшей эпохи, длительности эпох, размер ONNX-файла и CPU latency по полным test-изображениям. Замеры CPU SHALL охватывать подготовку, все тайлы, склейку маски и полный исходный кадр; система SHALL фиксировать характеристики CPU и не SHALL предполагать наличие CUDA при runtime-инференсе.

#### Scenario: One command produces deployable model and report
- **WHEN** CLI запускается на корректном `datasets/22022540` в full-окружении
- **THEN** создаются материализованный датасет с фиксированным split, run-отчёт, validation-selected ONNX-модель и финальные метрики test

#### Scenario: CPU benchmark processes full test images
- **WHEN** измеряется задержка экспортированной модели
- **THEN** каждый учитываемый замер SHALL включать инференс всех тайлов полного исходного снимка на CPU, а в отчёте SHALL быть указаны размеры изображений и использованный CPU

#### Scenario: Existing model is preserved
- **WHEN** новая модель экспортируется для UI
- **THEN** `models/colony_seg.onnx` SHALL остаться неизменённой, а новый файл SHALL быть сохранён отдельно
