## Context

В проекте уже есть работающий CLI-фреймворк тестирования (`uv run test-algorithms`):

- `testing/interface.py` — ABC `BaseDetectionAlgorithm` (`name`, `description`, `detect(image, is_cropped=False)`);
- `testing/registry.py` — класс-реестр с `@register_algorithm` (ключ = имя класса);
- `testing/classic_algorithms.py` — 4 классических алгоритма на `ColonyDetector`;
- `testing/dataset.py` — `TestDataset` читает `source/`, `masks/`, `cropped/`, `cropped_masks/`;
- `testing/metrics.py`, `testing/runner.py`, `testing/dashboard.py` — метрики, прогон, HTML-отчёт.

Ограничения текущей архитектуры:
1. Реестр хранит **только классы**, `get_algorithm(name)` создаёт экземпляр без аргументов. Невозможно зарегистрировать алгоритм, которому нужен путь к весам.
2. `run_all` всегда гоняет **весь реестр** — нет выбора подмножества.
3. Нет нейросетевых алгоритмов, хотя `train/predict.py` уже умеет ONNX-инференс.
4. Нет GUI; единственный путь — CLI.
5. Отчёт всегда содержит per-image таблицы; нет пары-сравнения «A vs B».

Окружения: `onnxruntime` есть только в `.venv-full` (extras `full`), в `.venv`/`.venv-build` его нет. Бинарь собирается Nuitka из `.venv-build`. `.onnx`-файлы трекаются в git (конвенция проекта), `.pt` — нет.

## Goals / Non-Goals

**Goals:**
- Единый интерфейс `BaseDetectionAlgorithm` остаётся точкой интеграции для классических и нейросетевых алгоритмов.
- Реестр умеет хранить как классы, так и готовые экземпляры (для моделей с весами).
- Поддержка внешних `.onnx`-весов (без пересборки) и встроенных (упакованных в бинарь) моделей.
- GUI-окно «Тестирование алгоритмов»: выбор датасета, чекбоксы алгоритмов, кнопка загрузки модели, сводная таблица, парное сравнение, экспорт отчёта, фон-прогон.
- CLI расширяется флагами `--algorithms`, `--per-snapshot`, `--compare`, остаётся совместимым.

**Non-Goals:**
- Не обучаем модели — только подключаем готовые `.onnx`.
- Не меняем `.venv` (runtime) — onnxruntime подгружается lazily.
- Не добавляем REST/веб-API тестирования.
- Не меняем формат и семантику существующих классических алгоритмов.

## Decisions

### D1. Двухуровневый реестр: классы + экземпляры

`testing/registry.py` расширяется: помимо словаря классов добавляется словарь готовых экземпляров `_INSTANCES`. `@register_algorithm` по-прежнему регистрирует класс; новый `register_algorithm_instance(name, instance)` — экземпляр. `get_algorithm(name)` SHALL искать в обоих словарях, `list_algorithms()` и `get_algorithm_descriptions()` объединяют оба списка.

**Почему не только классы:** модели с весами требуют параметр `model_path` в конструкторе; реестр классов не может их создать без данных. **Альтернатива (отвергнута):** подкласс `OnnxModelAlgorithm` на каждый файл весов — требует динамическое создание классов и теряет связь с GUI.

### D2. `OnnxModelAlgorithm` — универсальный адаптер NN

Новый файл `testing/onnx_algorithm.py` с `OnnxModelAlgorithm(BaseDetectionAlgorithm)`:

- Конструктор: `model_path`, `name`, `description`, `img_size=512`, `threshold=0.5`.
- `detect()` повторяет логику `train/predict.py:33` (`predict_single`): BGR→RGB, resize, нормализация ImageNet, инференс, бинаризация `> threshold` → `uint8` 0/255, resize маски обратно к входному shape.
- `onnxruntime` импортируется **внутри метода** (lazy) с понятным `RuntimeError` при отсутствии — модель не ломает запуск приложения без ML-окружения.

**Почему ONNX:** модель уже экспортируется в ONNX при обучении (`best.onnx`), инференс не требует torch/CUDA, `onnxruntime` — лёгкая зависимость. **Альтернативы (отвергнуты):** прямые torch-модели (тянут CUDA в runtime), C-плагины (усложнение сборки).

### D3. Источники весов: внешние и встроенные

- **Внешние:** GUI-кнопка «Загрузить модель (.onnx)» → `register_algorithm_instance(...)` на лету; путь сохраняется в `QSettings` для последующих запусков. CLI: параметр `--model <path>` регистрирует и включает в прогон.
- **Встроенные:** папка `models/` на уровне репозитория (`.onnx` в git). В `main.py`/`ui/testing_window.py` при старте сканируется `Path(__file__).resolve().parent.parent / "models"`; каждый `.onnx` регистрируется как экземпляр `OnnxModelAlgorithm`. Nuitka: в `scripts/nuitka_flags.py` добавляется `--include-data-files=models/*.onnx=models/`, при этом путь в собранном бинаре переопределяется через `__file__`-зависимый lookup (`sys._MEIPASS` не нужен для onefile Nuitka — Nuitka распаковывает data в каталог рядом с бинарём; уточняется на этапе реализации).

**Почему оба механизма:** пользователь не должен пересобирать приложение для новой модели (внешние веса), но дистрибутив может поставлять готовые веса из коробки (встроенные).

### D4. Расширение runner: подмножество + парное сравнение

`testing/runner.py`:

- `run_all(dataset, algorithms=None)` — принимает список имён; `None` = весь реестр.
- Новая `compare_algorithms(dataset, name_a, name_b)` → `{metric: {sample: winner_name}}` (или ничья), считается на тех же прогонах, что и `run_all`, без повторного инференса.

**Почему в runner, а не GUI:** парное сравнение нужно и CLI (`--compare A B`), и GUI; логика остаётся тестируемой.

### D5. Отчёт: опция per-snapshot + секция сравнения

`testing/dashboard.py`:
- `generate_report(all_results, output_path, include_per_snapshot=True, comparison=None)`.
- При `include_per_snapshot=False` per-algo секции (по-сним) скрываются, остаются сводная таблица и графики.
- При `comparison` (результат D4) добавляется HTML-таблица «победитель по каждому снимку».

### D6. GUI: `ui/testing_window.py` + QThread-прогон

Новое окно (аналогично `AnalysisWindow` по стилю Catppuccin). Кнопка «🧪 Тестирование алгоритмов» в `MainWindow` рядом с «✏️ Разметка...».

Состав окна:
- поле корня датасета (по умолчанию `test_images`) + кнопка выбора папки;
- список алгоритмов с `QCheckBox` (классические + зарегистрированные модели);
- кнопка «Загрузить модель (.onnx)»;
- опции: «Детализация по снимкам» (checkbox), «Парное сравнение» (два комбобокса A/B);
- кнопка «▶ Запустить», `QProgressBar` + статус;
- сводная `QTableWidget` (алгоритм × mean IoU/Dice/F1/Precision/Recall);
- кнопка «💾 Экспорт отчёта»;
- секция/вкладка парного сравнения (таблица победителей).

Прогон — через `QThread` (наследник `QThread` или `QObject`-worker, перемещённый в поток), сигналы `progress(int)` и `finished(dict)`. UI не блокируется (spec `algorithm-testing-gui`: "Testing runs in blocking off-UI thread").

**Почему QThread:** приложение синхронное PyQt6; простой `run_all` в main thread заморозит UI на минуты (детекция дорогая). **Альтернатива (отвергнута):** `QThreadPool`/`QRunnable` — избыточно для одного последовательного прогона.

### D7. CLI: обратная совместимость

`testing/__main__.py` получает флаги `--algorithms` (через запятую), `--model` (повторяемый, внешние веса), `--per-snapshot` (по умолчанию включён, `--no-per-snapshot` выключает), `--compare A,B`. Без флагов поведение идентично текущему.

## Risks / Trade-offs

- [Встроенные веса увеличивают размер бинаря (каждый `.onnx` ~80 МБ)] → Веса кладутся в `models/` по желанию; бинарь без них не ломается (пустой список моделей). Документировать размер.
- [Путь к `models/` в собранном Nuitka-бинаре может отличаться от исходников] → На этапе реализации проверить фактический путь распаковки Nuitka data и добавить fallback-поиск (несколько кандидатов). Отмечено как пункт в tasks.
- [Дублирование логики препроцессинга между `train/predict.py` и `OnnxModelAlgorithm`] → Обе функции тонкие; риск расхождения сведён к минимуму одним источником нормализации (константы ImageNet), допустимо до рефакторинга `train/predict.py` для импорта.
- [onnxruntime отсутствует в runtime-окружении] → lazy-import + понятное сообщение; GUI показывает модели как недоступные без падения (spec `neural-model-algorithms`: "Missing onnxruntime handled").
- [Парное сравнение трактует разные типы variant (source/cropped) как независимые снимки] → Сравнение ведётся внутри одного variant (source vs source, cropped vs cropped); документировать в UI подписи.

## Migration Plan

Обратная совместимость сохраняется: классические алгоритмы, `run_all(dataset)`, `generate_report(all_results, path)` без новых опций работают как раньше. Обновление по шагам:

1. Реестр (D1) + `OnnxModelAlgorithm` (D2) + встроенные веса (D3) — ядро, покрывается unit-тестами.
2. Runner (D4) + dashboard (D5) — CLI-функции, тесты.
3. CLI-флаги (D7) — тест через `--help` и прогон.
4. GUI (D6) + кнопка в `MainWindow` — pytest-qt smoke-тест открытия.
5. Nuitka: `--include-data-files` для `models/`.

Откат: изменить код-точку входа `main.py` не требуется; каждая стадия аддитивна.

## Open Questions

- Где хранить выбор пользователя (последний корень датасета, последние выбранные алгоритмы) — в `QSettings`? Предположительно да, уточняется в GUI-стадии.
- Фактический путь встроенных весов внутри Nuitka onefile — проверить на реальной сборке (блокер для шага 4 сборки, не для кода).