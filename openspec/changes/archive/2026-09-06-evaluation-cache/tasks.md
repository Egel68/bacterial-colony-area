## 1. Dataset signature utility

- [x] 1.1 Implement `compute_dataset_signature(data_root: str) -> str` в отдельном модуле `testing/cache.py`: для каждого файла изображения и маски (source + cropped) вычислить `relpath:filesize`, отсортировать лексикографически, конкатенировать, SHA256 → hex. Verifiable: запуск на `datasets/22022540_imported` возвращает 64-символьную строку и не меняется при повторном вызове.
- [x] 1.2 Написать тест `tests/test_evaluation_cache.py`: test_signature_stable — сигнатура не меняется при повторном вызове; test_signature_changes — сигнатура меняется при добавлении/удалении файла.

## 2. Cache persistence layer

- [x] 2.1 Реализовать `load_cache(cache_dir: str, signature: str) -> dict | None` и `save_cache(cache_dir: str, signature: str, data: dict) -> None` в `testing/cache.py`. Формат: `{"dataset_root": ..., "dataset_signature": ..., "created": ..., "algorithms": {name: {source: {...}, cropped: {...}}}}`. Сохранение — атомарно через `path.write_text` (перезапись существующего файла). Verifiable: запись → чтение возвращает те же данные.
- [x] 2.2 Написать тест `tests/test_evaluation_cache.py`: cache_roundtrip — записать и прочитать; cache_nonexistent — `load_cache` возвращает None для отсутствующего файла.

## 3. Modify `run_baseline` to use cache

- [x] 3.1 Изменить сигнатуру `run_baseline(dataset, use_cropped=True, cache=None)`. Перед циклом по алгоритмам проверять `cache.get(algo_name)`. Если есть — `log.info("Skipping %s (cached)", algo_name)` и добавить entry из кэша. Если нет — выполнить детектирование и записать entry в cache (по месту). Verifiable: первый запуск на датасете — все 4 алгоритма отрабатывают; второй запуск — все 4 пропущены.
- [x] 3.2 Убедиться, что частичный кэш работает: если в кэше 2 из 4 алгоритмов, то 2 пропускаются, 2 пересчитываются. Verifiable: ручная проверка логов.

## 4. Modify `run_evaluate` for cache integration

- [x] 4.1 В `run_evaluate(config)`: после загрузки датасета вычислить `signature = compute_dataset_signature(config["data_root"])`, загрузить кэш из `evaluations/.cache/<signature>.json`. Если `cache` есть — передать в `run_baseline`. После `run_baseline` сохранить обновлённый кэш. Verifiable: повторный запуск `uv run evaluate --data-root datasets/22022540_imported` — алгоритмы из первого запуска пропущены.

## 5. Add `--clear-cache` CLI flag

- [x] 5.1 В `testing/__main__.py`: добавить `--clear-cache` в парсер. В `_run_evaluate(args)`: перед загрузкой датасета, если `args.clear_cache`, вычислить сигнатуру и удалить `.cache/<signature>.json`. Verifiable: запуск с `--clear-cache --data-root datasets/22022540_imported` — кэш удалён, все алгоритмы пересчитаны.

## 6. Verify full pipeline

- [x] 6.1 Запустить `uv run evaluate --data-root test_images` — проверить, что кэш создаётся и повторный запуск пропускает алгоритмы.
- [x] 6.2 Запустить `uv run evaluate --data-root datasets/22022540_imported --no-cropped` — убедиться, что кэш работает и с `--no-cropped`.
- [x] 6.3 Запустить `uv run evaluate --data-root datasets/22022540_imported --clear-cache` — убедиться, что кэш очищен и прогон идёт с нуля.