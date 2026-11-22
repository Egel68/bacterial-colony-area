# Bacteria Colony Area Analyzer v3.0.1

## Новое

- **Воспроизводимый пайплайн обучающих датасетов.** Добавлены отдельные команды `prepare-colony-training`, `pack-colony-training` и `unpack-colony-training`: preflight COCO-изображений, маски, фиксированный split, black-background варианты, детерминированный lossless `.tar.zst` и проверка SHA-256/manifest.
- **Git LFS для датасетов.** Подготовленный набор `22022540` хранится в Git LFS одним архивом; JSON-манифесты и sidecars остаются в обычном Git. Документированы установка LFS, `git lfs pull`, распаковка, лицензия и квоты.
- **Обучение из готового snapshot.** `train-colony` принимает только полный распакованный набор, проверяет ссылки до baseline/optimizer и не создаёт изображения или маски во время обучения.

## Улучшения

- **Обновлённая модель `NN:colony_seg`.** Production-веса заменены на scratch U-Net, устойчивую к исходному и чёрному фону: IoU 0.4812/0.4721, Dice 0.6497/0.6414 на парном held-out наборе из 54 source ID.
- **Экспериментальное сравнение фонов.** Добавлены отчёт, confusion matrix и воспроизводимые метрики для original/black-background.
- **Документация и тесты.** Обновлены README, AGENTS и руководства по датасетам и обучению; добавлены тесты подготовки, архива, распаковки и раннего отказа.

## Обновление

Скачайте `BacteriaAnalyzer-Linux.zip` или `BacteriaAnalyzer-Windows.zip` со страницы релиза и запустите `BacteriaAnalyzer` / `BacteriaAnalyzer.exe`. Для запуска из исходников используйте `uv sync`, затем `uv run bacteria-analyzer`.

> Для получения обучающего архива установите Git LFS и выполните `git lfs pull`; без LFS вместо `.tar.zst` может остаться pointer.
