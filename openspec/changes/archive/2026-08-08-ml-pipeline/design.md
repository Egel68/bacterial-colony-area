## Context

`train/` — конвейеры аугментации (`augment.py`) и обучения U-Net (`train.py`, `dataset.py`, `config.py`, `models/`). Это обратная документация: контракты существующего кода фиксируются в спеках; поведение не меняется. Модели регистрируются через `@register_model` в `train/models/__init__.py`.

## Goals / Non-Goals

**Goals:**
- Специфицировать `data-augmentation` (тафы, seed, синхронность пары) и `unet-training` (конфиг, dataset+split, реестр моделей, U-Net, лосс, цикл обучения, чекпоинты, ONNX).
- Соотнести сценарии с кодом и (где возможно без torch) с тестами.

**Non-Goals:**
- Изменение гиперпараметров, архитектуры, команд CLI.
- Тестирование `train/` в обычном прогоне (torch ~5 ГБ, dev-окружение) — риски этапа 3/4 плана; сценарии проверяются код-ревью, не pytest в runtime.

## Decisions

1. **Реестр моделей остаётся в `train/models/__init__.py`** (`@register_model`) — не переносим, в отличие от `@register_algorithm` в `testing/`. Спек ссылается прямо на `train.models`.

2. **Две capability-спеки в одном change `ml-pipeline`.** `data-augmentation` и `unet-training` — общий домен обучения, один propose/archive цикл (Вариант А плана).

3. **Ключевой контракт пары — синхронность имён.** В `augment.py` маска пишется в `masks/{stem}_aug_{index:03d}.png`, картинка — в `images/` тот же `stem`. `ColonyDataset` вычисляет маску как `masks_dir / p.name`, поэтому имена обязаны совпадать.

4. **Нормализация в `dataset.py` и `preprocess` — ImageNet (mean/std).** Значения жёстко заданы в коде, не конфиг.

5. **`n_val = max(1, int(len*full*val_split))`** — при пустом/единичном наборе тренинг может работать с 1 валидационным образцом; не добавляем мягких случаев в спек, но Фиксация в `make_datasets`.

## Risks / Trade-offs

- **[Тестируемость без torch]** — сценарии `unet-training` нельзя прогнать в runtime-окружении (нет torch). → *tasks фиксируют сверку код-ревью; при желании добавить `@pytest.mark.dev`.*
- **[Seed и воспроизводимость]** — аугментация зависит от `random.seed(SEED+index)`, но про `albumentations` implementations относится к py-порядку вызовов; внутри запуска НЕ гарантируется побайтовая идентичность между версиями albumentations → *в спеке фиксируем воспроизводимость имён/индексов, не пикселей.*
- **[Early stopping T_max vs patience]** — `CosineAnnealingLR(T_max=epochs)` не рестартует; преждевременный останов не управляет эпохой LR → *не специфицируем, это оптимизация.*
- **[`summary.json` поля]** — содержат `{model, epochs, best_epoch, best_iou, train_samples, val_samples, img_size}`; не включаем в спеке все tboard-скалары.

## Migration Plan

Нет изменений кода — только докирование. `summary.json`/чекпоинты существующих run не затронуты.

## Open Questions

- Нужно ли специфицировать буферизацию Plotly-отчёта и дашборд FastAPI подробнее? Пока — поверхностно (порт 8765).
- Переводить ли `retain` маски (flip+rot в `dataset._augment`) в интеграцию с настройкой `augment`? Держим как есть — `augment=True` включает inner-аугментацию.