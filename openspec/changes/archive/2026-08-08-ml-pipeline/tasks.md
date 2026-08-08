## 1. Сверка спеки `data-augmentation` с кодом (train/augment.py)

- [x] 1.1 `AUGMENTATIONS_PER_IMAGE = 15`, `SEED = 42`, пути CROPPED/OUT
- [x] 1.2 Геометрический + пиксельный Compose с `additional_targets={"mask": "image"}`
- [x] 1.3 Имена `{stem}_aug_{index:03d}.png` пишутся в images/ и masks/ синхронно
- [x] 1.4 `_finalize`: маска бинаризуется `>127`
- [x] 1.5 Пары без маски пропускаются (`_load_image_mask`)

## 2. Сверка спеки `unet-training` с кодом (код-ревью без torch-прогона)

- [x] 2.1 `TrainingConfig` — все 17 полей с дефолтами
- [x] 2.2 `make_datasets` — rng(seed), `n_val=max(1, int(len*val_split))`, `FileNotFoundError` при пустом наборе
- [x] 2.3 `ColonyDataset` — resize INTER_LINEAR/NEAREST, normalize ImageNet, `augment` inner flip/rot всех
- [x] 2.4 `register_model`/`get_model`/`list_models`, ошибка Unknown model; `UNet` 4-level
- [x] 2.5 `get_loss` = BCE + Dice(smooth=1e-6); `compute_metrics` возвращает loss/iou/dice/precision/recall
- [x] 2.6 `run_training` — AdamW, CosineAnnealingLR(T_max=epochs), early stopping patience, best/last по val_iou
- [x] 2.7 Чистый `summary.json` поля; чекпоинты в `run_dir/checkpoints/`, tensorboard/
- [x] 2.8 `to_onnx` opset 18, `load_onnx` → InferenceSession; `preprocess` ImageNet+resize

## 3. Тесты на сценарии

- [ ] 3.1 (опц.) Быстрый unit-тест бинаризации `>127` и имён (без torch) — требует флейка; решено не добавлять, чтобы не тянуть torch
- [ ] 3.2 (опц.) Прогон `train.models` registry через dev-окружение — НЕ входит в задачу, сценарии закрыты ревью

## 4. Верификация

- [x] 4.1 `openspec validate --type change ml-pipeline` → valid
- [x] 4.2 `openspec archive ml-pipeline --yes` → добавлено 2+ capabilities в main specs
- [x] 4.3 `openspec validate --all` → всё валидно

## 5. Итог для diff-редистрибуции

- [x] 5.1 `train/*.py`, `train/models/*.py`, `tests/` не изменялись
- [x] 5.2 `openspec/changes/ml-pipeline/{proposal.md,design.md,tasks.md,specs/}` готовы