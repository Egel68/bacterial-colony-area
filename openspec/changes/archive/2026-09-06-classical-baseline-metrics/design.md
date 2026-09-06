## Context

См. proposal.md — Why. Текущий `TestDataset` (testing/dataset.py) ожидает маски в отдельной директории `masks/`, а `CocoBboxImporter` (train/dataset_adapters.py) пишет маски в той же директории `source/` с суффиксом `_mask`. Для прогона на полном датасете 22022540 нужна прослойка, которая понимает обе структуры и агрегирует метрики в машиночитаемый JSON.

## Goals / Non-Goals

**Goals:**
- Загружать датасет в структуре CocoBboxImporter-output (source/*.jpg + source/*_mask.png, опционально cropped/*_cropped.jpg + cropped/*_cropped_mask.png)
- Загружать датасет в legacy-структуре `test_images/` (source/ + masks/)
- Прогонять 4 классических алгоритма с UI-параметрами
- Агрегировать метрики отдельно по source и cropped выборкам
- Экспортировать JSON-отчёт с mean/std метрик
- CLI с флагами --data-root, --output, --no-cropped

**Non-Goals:**
- Не менять существующие алгоритмы, метрики, runner или реестр
- Не добавлять новые метрики
- Не генерировать HTML-отчёт (он уже есть в testing/dashboard.py)
- Не подготавливать датасет (работаем с уже импортированным CocoBboxImporter-ом)

## Decisions

| Decision | Choice | Alternatives considered |
|---|---|---|
| Модуль размещения | Новый `testing/baseline.py` | Расширение `testing/__main__.py` — модуль обеспечивает изоляцию и переиспользование из кода |
| Загрузка датасета | `BaselineDataset` с авто-детекцией структуры (приоритет: `dataset.json` → `source/` + `masks/` → `source/` + `source/*_mask.png`) | Использовать `load_manifest` из train/ — это смешивает train- и test-слои, недопустимо для CI-сборки |
| Формат отчёта | JSON (не CSV) — вложенная структура удобнее для программного сравнения, один файл вместо нескольких |
| CLI точка входа | Новый entry point в pyproject.toml (`uv run baseline`) плюс программный вызов из `testing/__main__.py` с `--mode baseline` | Отдельный исполняемый файл — entry point проще, меньше кода |
| Повторное использование | `compute_segmentation_metrics` из testing/metrics.py, `_mean_metrics` из testing/runner.py (без изменений) |

## Risks / Trade-offs

- [Dataset compatibility] CocoBboxImporter может создать датасет без cropped-вариантов (`--crop` не указан) — BaselineDataset обрабатывает source-only корректно, cropped-секция в JSON отсутствует.
- [Performance] 369 source-снимков × 4 алгоритма = 1476 прогонов по ~0.5–2 с каждый → до ~50 мин выполнения. Mitigation: прогресс-бар (tqdm) в CLI, возможность прервать и перезапустить.
- [Memory] OpenCV imread для 369 полноразмерных снимков может занять >2 ГБ. Mitigation: загрузка ленивая, по одной паре за раз (iterable).