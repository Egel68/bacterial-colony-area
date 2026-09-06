## 1. Реестр алгоритмов (D1)

- [x] 1.1 Расширить `testing/registry.py`: добавить словарь `_INSTANCES`, функцию `register_algorithm_instance(name, instance)`; `get_algorithm`, `list_algorithms`, `get_algorithm_descriptions` объединяют классы и экземпляры
- [x] 1.2 Написать unit-тесты реестра: регистрация экземпляра, запрос экземпляра по имени, перечисление смешанного списка, `ValueError` для неизвестного имени

## 2. ONNX-адаптер (D2, spec: neural-model-algorithms)

- [x] 2.1 Создать `testing/onnx_algorithm.py` с `OnnxModelAlgorithm(BaseDetectionAlgorithm)` (ctor: `model_path`, `name`, `description`, `img_size=512`, `threshold=0.5`)
- [x] 2.2 Реализовать `detect()`: lazy-import `onnxruntime`, BGR→RGB, resize, ImageNet-нормализация, инференс, бинаризация `> threshold` → `uint8` 0/255, resize маски к входному shape (логика из `train/predict.py`)
- [x] 2.3 Выбрасывать понятный `RuntimeError` при отсутствии `onnxruntime` в окружении
- [x] 2.4 Unit-тесты с фиктивным ONNX-моделью (синтетический сеанс или monkeypatch): маска того же shape, порог 0/255, ошибка при отсутствии onnxruntime

## 3. Встроенные и внешние веса (D3, spec: neural-model-algorithms)

- [x] 3.1 Добавить папку `models/` уровня репозитория (с `.gitignore`-плацебо + README-комментарий)
- [x] 3.2 Создать функцию сканирования встроенных моделей (`scan_bundled_models() -> list[OnnxModelAlgorithm]` по `models/*.onnx`)
- [x] 3.3 Зарегистрировать встроенные модели при старте (вызов из `main.py` и/или `ui/main_window.py`)
- [x] 3.4 CLI: параметр `--model <path>` регистрирует внешний `.onnx` и включает в прогон
- [x] 3.5 Nuitka: добавить `--include-data-files=models/*.onnx=models/` в `scripts/nuitka_flags.py`; провести/проверить путь распаковки в собранном бинаре (фолбэк-поиск путей)

## 4. Runner: подмножество и парное сравнение (D4, spec: algorithm-testing)

- [x] 4.1 `run_all(dataset, algorithms=None)` — принимать список имён (None = весь реестр); соответствующий unit-тест
- [x] 4.2 Новая `compare_algorithms(dataset, name_a, name_b)` → победитель по снимку/метрике с учётом variant (source vs source, cropped vs cropped); unit-тест на маленьком датасете

## 5. Отчёт: опции и сравнение (D5, spec: algorithm-testing, algorithm-testing-gui)

- [x] 5.1 `generate_report(..., include_per_snapshot=True, comparison=None)` — прячет per-algo секции при `include_per_snapshot=False`
- [x] 5.2 Добавить HTML-секцию «Сравнение алгоритмов» в `dashboard.py` при `comparison`
- [x] 5.3 Тесты: отчёт без per-snapshot не содержит per-algo таблиц; отчёт со сравнением содержит таблицу победителей

## 6. CLI (D7)

- [x] 6.1 `--per-snapshot` / `--no-per-snapshot` флаги в `testing/__main__.py`
- [x] 6.2 `--compare A,B` флаг — вызов `compare_algorithms` и передача в отчёт
- [x] 6.3 Проверить обратную совместимость: `uv run test-algorithms` без флагов работает как раньше

## 7. GUI (D6, spec: algorithm-testing-gui)

- [x] 7.1 Создать `ui/testing_window.py`: поле корня датасета + выбор папки, список алгоритмов с чекбоксами (классические + модели), кнопка «Загрузить модель (.onnx)»
- [x] 7.2 Добавить опции «Детализация по снимкам» и «Парное сравнение» (комбобоксы A/B)
- [x] 7.3 Реализовать фон-прогон через QThread (сигналы `progress` / `finished`); UI не блокируется, прогресс-бар
- [x] 7.4 Сводная `QTableWidget` (алгоритм × mean IoU/Dice/F1/Precision/Recall) + секция парного сравнения
- [x] 7.5 Кнопка «Экспорт отчёта» → `QFileDialog` → `generate_report`
- [x] 7.6 Кнопка «🧪 Тестирование алгоритмов» в `MainWindow` рядом с кнопкой разметки
- [x] 7.7 pytest-qt smoke-тест: окно открывается, список алгоритмов заполнен, пустой датасет показывает ошибку

## 8. Сборка и документация

- [x] 8.1 Убедиться, что `.venv`/`.venv-build` не тянут `onnxruntime` (lazy-import в моделях)
- [x] 8.2 Обновить AGENTS.md (новая секция про GUI-тестирование и подключение моделей) и README.md
- [x] 8.3 Прогнать `uv run test-algorithms` и локальное GUI-открытие на реальном датасете; зафиксировать результат

## 9. Доработки GUI (спека algorithm-testing-gui)

- [x] 9.1 Таблицы результатов и парного сравнения растягиваются на всю ширину окна (stretch-колонки)
- [x] 9.2 Подсказка-инструкция о структуре датасета рядом с полем «Датасет» (source, masks, cropped, cropped_masks)