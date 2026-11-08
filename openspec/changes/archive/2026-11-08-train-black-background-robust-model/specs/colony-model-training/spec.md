# Spec Delta

## ADDED Requirements

### Requirement: Scratch U-Net training with background variants
Эксперимент SHALL обучать архитектуру unet со случайной инициализацией без pretrained-весов и без resume-checkpoint. Оптимизация SHALL выполняться обязательно на CUDA GPU; если CUDA недоступна, запуск SHALL завершиться понятной ошибкой и SHALL NOT переключаться на CPU. В градиентное обучение SHALL попадать только subset=train; checkpoint SHALL выбираться по val, а subset=test SHALL быть исключён из обучения и выбора checkpoint. Фактически использованное CUDA-устройство SHALL фиксироваться в summary/report.

#### Scenario: Model starts from random initialization
- **WHEN** запускается black-background эксперимент с архитектурой unet
- **THEN** исходным состоянием весов SHALL быть случайная инициализация, а запуск SHALL обходиться без загрузки pretrained-весов или checkpoint

#### Scenario: Only training groups affect learned weights
- **WHEN** манифест содержит оригиналы и чёрнофоновые варианты с subset train, val и test
- **THEN** градиенты SHALL вычисляться только по train; val SHALL использоваться только для выбора checkpoint, а test SHALL быть исключён из optimizer и выбора checkpoint и использоваться только для предусмотренных baseline и post-training оценок

#### Scenario: CUDA is mandatory for optimizer updates
- **WHEN** запускается black-background experiment без доступного CUDA-устройства
- **THEN** выполнение SHALL остановиться с ясной ошибкой до optimizer updates и SHALL NOT незаметно обучаться на CPU; при успешном запуске summary SHALL указывать CUDA device

### Requirement: Matched baseline and retrained evaluation
После полной материализации и проверки dataset эксперимент SHALL до обучения оценить текущую bundled U-Net на held-out test-ID для исходных и чёрнофоновых вариантов. После обучения scratch-U-Net SHALL быть оценена на этих же held-out ID, масках и обоих вариантах фона. Baseline и post-training результаты SHALL использовать одинаковый source-ID набор. Отчёт SHALL указывать, известны ли ID, использованные при обучении исходной модели; неизвестное пересечение SHALL маркироваться как unknown, а не как независимая test-оценка.

#### Scenario: Baseline black-background metrics precede training
- **WHEN** полный dataset подготовлен и проверен
- **THEN** baseline SHALL быть оценена на test source ID для original и black вариантов до первого optimizer update; CPU-инференс baseline допускается

#### Scenario: Models are compared on matching samples
- **WHEN** сформированы baseline и post-training результаты
- **THEN** каждый результат SHALL содержать одинаковый набор source ID и соответствующие оценки original-background и black-background вариантов

#### Scenario: Original model training membership is unavailable
- **WHEN** система не может установить ID sample, использованных при обучении текущей модели
- **THEN** отчёт SHALL явно указывать unknown для независимости baseline test-набора и SHALL NOT выдавать эти метрики за подтверждённую оценку на неиспользованных ранее sample

### Requirement: Pixel confusion matrix and precision-recall reporting
Для каждой пары модель/фон итоговый отчёт SHALL содержать суммарные pixel-level TP, FP, FN и TN, а также precision и recall, вычисленные из этих сумм. Матрица SHALL иметь строки actual background/colony и столбцы predicted background/colony; отчёт SHALL сохранять соответствующие значения по каждому sample ID.

#### Scenario: Aggregate metrics agree with the confusion matrix
- **WHEN** итоговая бинарная маска сопоставляется с GT-маской на всех test sample заданного фона
- **THEN** матрица SHALL быть [[TN, FP], [FN, TP]], precision SHALL равняться TP/(TP+FP), а recall SHALL равняться TP/(TP+FN) с определённой системой обработкой нулевого знаменателя

#### Scenario: Per-sample errors are traceable
- **WHEN** формируется отчёт по held-out набору
- **THEN** для каждого source ID и фонового варианта SHALL быть доступны TP, FP, FN, TN, precision и recall
