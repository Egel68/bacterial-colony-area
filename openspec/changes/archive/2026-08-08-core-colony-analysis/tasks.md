## 1. Сверка спеки с кодом

- [x] 1.1 Сверено: `detect_petri_dish` — два пути (блики/200 → min_enclosing_circle, критерии ≥10% и ≤30%), fallback `HoughCircles`, `(None, None)` — поведение соответствует спеки
- [x] 1.2 Сверено: критерии выбора контура и `create_inner_mask`/@margin_percent соответствуют описанным в спеке
- [x] 1.3 Сверено: `AnalysisParams.__post_init__` валидирует диапазоны (sensitivity/contrast/margin/min_size/fill_strength)
- [x] 1.4 Сверено: пайплайн `detect_colonies` повторяет шаги спеки (зелёный канал → CLAHE → median → bg → порог → морфология → components)

## 2. Тесты на сценарии спеки

- [x] 2.1 Добавить/проверить тест: детекция чашки при ярких объектах (assert mask + PetriInfo)
- [x] 2.2 Добавить/проверить тест: fallback на Hough при отсутствии ярких контуров
- [x] 2.3 Добавить/проверить тест: `(None, None)` при пустом изображении (без чашки)
- [x] 2.4 Добавить/проверить тест: `AnalysisParams` out-of-range → AssertionError
- [x] 2.5 Добавить/проверить тест: `detect_colonies` на пустом ROI возвращает нулевую маску
- [x] 2.6 Добавить/проверить тест: AreaCalculator с margin и без — корректные значения коэффициентов
- [x] 2.7 Добавить/проверить тест: `AnalysisController.analyze` возвращает заполненный `AnalysisResult`

## 3. Верификация

- [x] 3.1 `uv run pytest -q` — все тесты зелёные
- [x] 3.2 `uv run ruff check` — чистый код
- [x] 3.3 `openspec validate --all` — спеки валидны