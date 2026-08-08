## Why

Модуль `train/` реализует два независимых конвейера: аугментацию данных (`augment.py`, CLI `uv run augment`) и обучение U-Net (`train.py` + `models/` + `dataset.py` + `config.py`, CLI `uv run train.main train`). Оба конвейера существуют в коде, но их контракты не специфицированы. Цель — зафиксировать поведение в capability-спеках `data-augmentation` и `unet-training` и подтвердить сценарии существующими тестами (в части, протестируемой без dev-окружения с torch).

## What Changes

- Добавляются две capability-спеки в `openspec/specs/`:
  - `data-augmentation` — генерация 15× аугментированных пар изображение-маска.
  - `unet-training` — загрузка датасета, конфигурация, модель U-Net, цикл обучения с чекпоинтами и early stopping.
- Поведение кода НЕ меняется. Меняются только спецификации (reverse-documentation).
- Файлы-источники: `train/augment.py`, `train/config.py`, `train/dataset.py`, `train/train.py`, `train/models/base.py`, `train/models/unet.py`, `train/models/__init__.py`.

## User Impact

- Пользователи CLI не затронуты: команды `uv run augment` и `uv run train.main train` работают как раньше.
- Разработчики получают проверяемые контракты для конвейеров обучения.

## Out of Scope

- Обновление гиперпараметров или архитектуры модели.
- Перенос `@register_model` из `train/models/__init__.py` (остаётся в train-модуле, в отличие от `@register_algorithm` в `testing/`).
- GUI-дашборд во время обучения специфицируется только поверхностно (порт 8765, потом не расширяем).