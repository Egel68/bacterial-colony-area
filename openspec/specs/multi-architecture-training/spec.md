# multi-architecture-training Specification

## Purpose
Обучение и сравнение нескольких CNN-архитектур сегментации колоний: выбор архитектуры через конфиг и CLI, обучение с CPU-fallback, сбор метрик в реальном времени, конвейер «обучи и сравни» с итоговым сводным отчётом и экспортом в ONNX для инференса в `testing/`.
## Requirements
### Requirement: Architecture selection in configuration

Система SHALL предоставлять выбор архитектуры модели через `TrainingConfig` (поле `model_name`) и через CLI (`--model`). Обучение SHALL стартовать с архитектурой из конфига, используя реестр моделей из `unet-training`.

#### Scenario: Custom architecture selected via config
- **КОГДА** `TrainingConfig(model_name="unet_small")` передаётся в `run_training`
- **ТОГДА** система SHALL обучать модель, созданную через `get_model("unet_small")`

#### Scenario: Default architecture remains unet
- **КОГДА** создаётся `TrainingConfig()` без аргументов
- **ТОГДА** `model_name` SHALL равняться `"unet"`

### Requirement: Training on CUDA with CPU fallback

Система SHALL обучать модели на устройстве `cfg.device`, если оно доступно (`torch.cuda.is_available()`), иначе — на `cpu`. Экспорт в ONNX SHALL выполняться на активном устройстве.

#### Scenario: CUDA available uses GPU
- **КОГДА** `cfg.device == "cuda"` и CUDA доступна
- **ТОГДА** обучение и веса (включая `to_onnx`) SHALL выполняться на GPU-устройстве

#### Scenario: CUDA unavailable falls back to CPU
- **КОГДА** `cfg.device == "cuda"`, но CUDA недоступна
- **ТОГДА** система SHALL обучать на `cpu` без ошибок

### Requirement: Real-time metrics collection

Система SHALL собирать метрики (loss, IoU, Dice, precision, recall, lr) по каждой эпохе обучения и доставлять их в реальном времени: через `progress_callback` (rich-прогресс или веб-дашборд) и через TensorBoard (`SummaryWriter`). Колбэк SHALL вызываться после каждой эпохи с метриками train и val.

#### Scenario: Metrics emitted per epoch
- **КОГДА** обучение идёт с `progress_callback` на минимальном датасете
- **ТОГДА** колбэк SHALL вызываться после каждой эпохи и получать `train_metrics`, `val_metrics`, `elapsed` с ключами `loss`, `iou`, `dice`, `precision`, `recall`

#### Scenario: TensorBoard scalars written
- **КОГДА** обучение завершается хотя бы одну эпоху
- **ТОГДА** в `run_dir/tensorboard/` SHALL быть записаны scalar-метрики `train/*` и `val/*`

### Requirement: Multi-architecture train-and-compare

Система SHALL предоставлять конвейер (`run_train_compare`, CLI-команда `train-compare`) для обучения и оценки нескольких архитектур: принимает список архитектур (например `["unet", "unet_small"]`), источник данных (через контракт `training-data-contract`) и корень тестовых пар; по каждой архитектуре запускает `run_training` с сохранением `best.onnx`, затем оценивает каждую `best.onnx` на тестовых парах через `TestDataset` из `testing/` и `compute_segmentation_metrics`, и формирует сводный отчёт (JSON + HTML) со средними метриками по архитектурам.

#### Scenario: Trains all requested architectures
- **КОГДА** `run_train_compare(["unet", "unet_small"], cfg, data_root, eval_root)` выполняется на минимальном датасете с малым числом эпох
- **ТОГДА** SHALL быть создано по run-директории для каждой архитектуры с `checkpoints/best.onnx`

#### Scenario: Comparison summary emitted
- **КОГДА** обе архитектуры обучены и оценены на тестовых парах
- **ТОГДА** SHALL быть сформирован сводный отчёт, содержащий по каждой архитектуре средние `iou`, `dice`, `precision`, `recall`, `f1`

### Requirement: ONNX export for CPU inference

Система SHALL экспортировать обученную модель каждой архитектуры в ONNX (`opset=18`, dynamic batch axis) через существующий `BaseSegmenter.to_onnx`, пригодный для инференса на CPU в фреймворке `testing/` через `OnnxModelAlgorithm`.

#### Scenario: Best ONNX is loadable as algorithm
- **КОГДА** для каждой обученной архитектуры существует `checkpoints/best.onnx`
- **ТОГДА** `OnnxModelAlgorithm(best.onnx)` SHALL создаться и `detect(image)` SHALL вернуть бинарную `uint8`-маску без исключений