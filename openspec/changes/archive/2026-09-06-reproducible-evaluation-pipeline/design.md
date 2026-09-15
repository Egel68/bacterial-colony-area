## Context

См. proposal.md — Why. Существующий `testing/baseline.py` умеет запускать 4 классических алгоритма и экспортировать JSON, но результат идёт в произвольную директорию, не привязан к версии кода, нет автоматической HTML-визуализации, невозможно воспроизвести старый прогон. Текущий `dashboard.py` умеет генерировать HTML, но используется только из CLI `test-algorithms`.

## Goals / Non-Goals

**Goals:**
- Единый entry point `uv run evaluate` с поддержкой YAML-конфига
- Автоматическое создание `evaluations/<run-id>/` с report.json, report.html, config.yaml, run_info.json
- Запись git-commit-id в run_info.json
- Интеграция импорта датасета (CocoBboxImporter) как опционального шага pipeline
- Возможность запустить только прогон (без импорта) на уже готовом датасете
- Совместимость с существующим `run_baseline` и `compute_segmentation_metrics`

**Non-Goals:**
- Не менять алгоритмы, метрики, `TestDataset`, `BaselineDataset`
- Не добавлять новые метрики
- Не реализовывать веб-дашборд с историей прогонов (только статический HTML)
- Не интегрировать CI/CD прогоны (только локальный запуск)

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Конфиг-формат | YAML (через PyYAML как опциональная зависимость в `[project.optional-dependencies]`) | YAML читается человеком, допускает комментарии, поддерживает вложенность; fallback на JSON если PyYAML не установлен |
| Идентификатор прогона | `YYYY-MM-DD_HH-MM-SS` (строка) | Просто, уникально, сортируемо, человекочитаемо |
| Директория результатов | `evaluations/<run-id>/` в корне репозитория | Позволяет отслеживать в git, привязана к коду, а не к /tmp |
| Git-track | `evaluations/` в git, игнорировать только бинарные артефакты | История прогонов — часть репозитория |
| Модуль | Новый `testing/evaluator.py` | Не смешивать с существующим baseline-модулем; evaluator оркестрирует, baseline считает |
| Импорт в pipeline | Вызов `CocoBboxImporter().build()` напрямую из evaluator | Избежать subprocess; всё в одном Python-процессе |

## Risks / Trade-offs

- [Run time] Полный pipeline (импорт 369 изображений + 4 алгоритма) занимает ~60 мин — без прогресс-бара пользователь может подумать, что зависло. Mitigation: tqdm и логирование шагов.
- [Git tracking] `evaluations/` будет расти с каждым прогоном (JSON + HTML ~10 KB на прогон). Это приемлемо, т.к. 1000 прогонов займут ~10 MB. Mitigation: gitignore на HTML (можно перегенерировать из JSON), хранить только JSON.
- [PyYAML availability] В runtime-окружении PyYAML может отсутствовать. Mitigation: fallback на JSON-парсинг; YAML-конфиг — опциональная возможность.