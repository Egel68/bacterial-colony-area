## Why

Алгоритм анализа колоний уже реализован и стабилен, но его контракт не задокументирован: нет спеки на capability `colony-analysis`, отсутствует привязка требований к коду и тестам. Это затрудняет ревью изменений и регрессионное тестирование.

## What Changes

- Добавить новую capability `colony-analysis`: автоматическое определение чашки Петри и подсчёт колоний, параметризация, расчёт площадей.
- Зафиксировать в спеки: пайплайн детекции, критерии отбора контура чашки, диапазоны и валидации параметров, поведение solid_fill, расчёт площадей и покрытия.
- Сопоставить каждый сценарий спеки с существующим кодом и тестами (reverse documentation, без изменения логики).

## Capabilities

### New Capabilities
- `colony-analysis`: Автоматическое определение колоний бактерий на изображении чашки Петри: поиск чашки (блики → minEnclosingCircle, fallback HoughCircles), параметризуемый пайплайн детекции, расчёт площадей и покрытия.

### Modified Capabilities
<!-- Нет существующих specs, все capabilities новые -->

## Impact

- `analysis/colony_detector.py`, `analysis/image_processor.py`, `analysis/geometry.py`, `analysis/params.py`, `analysis/results.py`
- `utils/calculations.py`
- `ui/controllers/analysis_controller.py`
- Внешних зависимостей не добавляется: используются уже имеющиеся OpenCV и NumPy.