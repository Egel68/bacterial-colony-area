# План: спецификации проекта и покрытие сценариев тестами

> Назначение: реализуется в отдельной сессии. Файл — единственный источник задач.
> Решения, принятые до сохранения: **язык спек — русский**, **вариант А для SUPPORTED_EXTENSIONS** (перенос в `utils/image_loader.py`).
> Опция структуры changes (этап 2) — **согласована: Вариант А** (3 changes по доменам). См. Этап 2.

---

## Итоговое состояние (отправная точка)

- `openspec/` создан, **не в git** (untracked `openspec/`, `.opencode/`, `opencode.json`).
- `openspec/specs/` содержит один главный спек `robust-image-loading` (создан вручную, синтаксис уже валиден: `openspec validate --specs` → ✓).
- Код проекта реализован: 56 тестов проходят, `ruff` без ошибок.
- В `openspec/changes/archive/2026-07-25-fix-unopenable-images-labeling/` лежит архивированный change с delta-спекой `robust-image-loading` (написана как `## MODIFIED`, должна быть `## ADDED`).

## Ключевые решения (зафиксированы)

| # | Решение | Значение |
|---|---|---|
| 1 | Язык спек | **русский** (все новые и существующие спеки) |
| 2 | SUPPORTED_EXTENSIONS | **вариант А**: перенос в `utils/image_loader.py` |
| 3 | Git | коммитим `openspec/**`; `opencode.json` НЕ коммитим; `.opencode/node_modules/` в `.gitignore` |
| 4 | Этап 2 (структура changes) | **Вариант А** — 3 changes по доменам (согласовано) |

---

## Этап 0 — Инфраструктура

- [ ] 0.1 `.gitignore`: добавить строки
  ```
  .opencode/node_modules/
  opencode.json
  ```
- [ ] 0.2 Ввести `openspec/**` в git (не коммитить до этапа 6).
- [ ] 0.3 Проверить, что `opencode.json` в корне проекта = только `$schema` (50 байт); сам файл не коммитим.

Результат: репозиторий чист от служебных окружений, спэки версионируются.

---

## Этап 1 — Привести `robust-image-loading` в порядок

### 1.1 Главная спецификация: `openspec/specs/robust-image-loading/spec.md`
- [ ] Структурные заголовки НЕ трогать (обязательны для валидатора): `# Robust Image Loading` / `## Purpose` / `## Requirements` / `### Requirement:` / `#### Scenario:`.
- [ ] Переводу подлежит ТОЛЬКО текстовое содержимое (Purpose, текст требований, WHEN/THEN в сценариях). Оформление WHEN/THEN → «КОГДА/ТОГДА» — унифицировать один раз по проекту.
- [ ] Термины SHALL/WILL/SHALL NOT перевести: ОБЯЗАН / ДОЛЖЕН / НЕ ДОЛЖЕН (одна русская нотация).
- [ ] Убрать опечатки, вставленные при ручном создании в прошлой сессии (проверить текст целиком).
- [ ] Сохранить все 5 требований и **11 сценариев** из delta-спеки (переведёнными): по 5/1/2/2/1 на требование.

### 1.2 Delta-спека в архиве
- [ ] `openspec/changes/archive/2026-07-25-fix-unopenable-images-labeling/specs/robust-image-loading/spec.md`:
  - [ ] Заголовок `## MODIFIED Requirements` → `## ADDED Requirements` (capability была новая).
  - [ ] Перевести текст на русский (единый язык).
- [ ] `openspec/changes/archive/.../.openspec.yaml` — оставить как есть (мета).

### 1.3 Валидация этапа
- [ ] `openspec validate --specs` → 1 passed.

---

## Этап 2 — Capability-спеки ядра (главный разветвлённый выбор)

> Цель: наполнить `openspec/specs/` спеками по реальному функционалу через workflow
> `/opsx-propose` → `/opsx-apply` → `/opsx-sync` → `/opsx-archive`.
> Код УЖЕ существует, поэтому tasks внутри change — это «проверка соответствия
> спеки коду + написание тестов на сценарии», а не написание фичи.

### Варианты структуры ✅ Выбрано: Вариант А

**✅ Вариант А — Множество changes (по домену, 3 штуки)** — *РЕАЛИЗУЕМ*

| Change | Capabilities | Файлы-источники |
|---|---|---|
| `core-colony-analysis` | `colony-analysis` | `analysis/`, `utils/calculations.py`, `ui/controllers/analysis_controller.py` |
| `labeling-and-testing` | `image-labeling`, `algorithm-testing` | `ui/controllers/labeling_controller.py`, `labeling/labeling_window.py`, `testing/` |
| `ml-pipeline` | `data-augmentation`, `unet-training` | `train/` (augment.py, config.py, dataset.py, train.py, models/, dashboard/) |

Плюсы: атомарность (каждый change независимо ревьюится и архивируется), история по доменам, гибкость (один домен медленный — не блокирует другие). Компромисс: 3 прохода propose/archive.

**Вариант Б — Один составной change `core-capabilities`** *(отклонён)*
Плюсы: один цикл propose→archive, меньше ручной работы. Минусы: огромный change (5 capabilities, ~40+ сценариев), трудно ревьюировать и валидировать по частям.

**Вариант В — 5 отдельных changes по одному на capability** *(отклонён)*
Плюсы: максимальная атомарность, чистый history. Минусы: 5× полный цикл propose/archive, избыточная церемония для «обратной документации» существующего кода.

### 2.1 Заготовка содержимого специй (общая для вариантов)

Для каждой capability примерно такой план content (на русском):

#### `colony-analysis`
- Purpose: автоматическое определение колоний на изображении чашки Петри.
- Requirements (файлы: `analysis/colony_detector.py`, `analysis/image_processor.py`, `analysis/geometry.py`, `analysis/params.py`, `analysis/results.py`, `utils/calculations.py`, `ui/controllers/analysis_controller.py`):
  - поиск чашки Петри: два пути (блики→minEnclosingCircle, fallback HoughCircles), критерии отбора контура (площадь ≥10%, близость к центру ≤30% min(h,w));
  - параметризация `AnalysisParams`: sensitivity 0.01..1.0, contrast 0.5..3.0, margin 0..30, min_size 1..1000, solid_fill, fill_strength 1..100 (валидации в `AnalysisParams.__post_init__`);
  - пайплайн детекции: зелёный канал → CLAHE(clip=2×contrast) → median blur → Gaussian bg-вычитание → адаптивный порог (mean+k·std, k=3.0−2.5·sensitivity) → морфология open/close → connected components с фильтром по `min_colony_size`;
  - solid_fill: ядро морф. закрытия fill_strength + заливка заполнением контуров;
  - расчёты `AreaCalculator`: площадь колоний в px и mm², площадь чашки Петри (внутренняя с учётом margin), покрытие %, px_to_mm2 при диаметре 90 мм;
  - `AnalysisResult` (frozen dataclass) и `PetriInfo` с инвариантами.
- Scenario пример: «скрытое изображение с бликами», «fallback на Hough при отсутствии бликов», «валидация параметров вне диапазона → ошибка».

#### `image-labeling`
- Purpose: ручная разметка масок, обрезка по чашке, экспорт.
- Requirements (файл `ui/controllers/labeling_controller.py`, `labeling/labeling_window.py`):
  - список файлов по `SUPPORTED_EXTENSIONS` (после переноса заголовок constants живёт в `utils/image_loader.py`);
  - `get_current_dir`/`get_mask_dir` — выбор source vs cropped и соответствующие папки масок;
  - `crop_by_petri` — извлечение квадрата раскрытия, обнуление фона по круговой маске;
  - `save_mask`/`load_mask` (grayscale, проверка shape);
  - `export_session_to_zip` — рекурсивная упаковка session_dir.
- Scenario: «файлы с `.gif`/`.svg` не показываются в списке», «обрезка не выходит за границы».

#### `algorithm-testing`
- Purpose: сравнение алгоритмов на ground-truth.
- Requirements:
  - `@register_algorithm`/`list_algorithms`/`get_algorithm`/дескрипции;
  - интерфейс `BaseDetectionAlgorithm` (name, description, detect(image, is_cropped));
  - 4 классических алгоритма (Default, HighSensitivity, SolidFill, LowSensitivity) с фикс. параметрами;
  - `TestDataset` — загрузка пар (source+mask, cropped+cropped_mask) из `test_images/`;
  - метрики IoU/Dice/F1/Precision/Recall/Accuracy (TP/FP/FN/TN);
  - `run_all` прогон по всем алгоритмам, `_mean_metrics` μ±σ;
  - HTML-отчёт `dashboard.py` (Chart.js).
- Scenario: «прогон алгоритма на пустом наборе → пустой результат, без падения».

#### `data-augmentation`
- Purpose: генерация обучающего набора для модели (15× на исходник).
- Requirements:
  - геометрические: Rotate/HFlip/VFlip/Affine (scale ±10%, translate ±5%), пиксельные: BrightnessContrast, GaussNoise, Blur, HueSaturationValue;
  - пары image+mask консистентно (additional_targets), маска после ауг бинаризуется (mask > 127);
  - выход `train/data/images/` + `train/data/masks/`, имена `{stem}_aug_{idx:03d}.png`, seed 42.
- Scenario: «для 1 пары и 15 аугментаций на выходе ровно 15 пар, пары лежат синхронно».

#### `unet-training`
- Purpose: обучение U-Net сегментации колоний.
- Requirements:
  - `TrainingConfig` (img_size, batch_size, epochs, lr, val_split, patience, model_name=unet, device, dashboard);
  - `ColonyDataset`/`make_datasets` — resize 512, нормализация (ImageNet), train/val split 0.2 по seed 42, отключаемая inner-аугментация (flip/rot);
  - модель `Unet` (BaseSegmenter): forward/sigmoid loss BCE+Dice, compute_metrics (IoU/Dice/P/R), ONNX-экспорт (opset 18), preprocess/load_onnx;
  - `run_training`: цикл, CosineAnnealingLR, early stopping (patience), лучший по val_IoU, tensorboard-логи, ckpt `best.pt/onnx`, `last.pt/onnx`, `summary.json`, Plotly-отчёт;
  - dashboard (FastAPI+Chart.js, порт 8765).
- Scenario: «обучение на 1-2 синтетических парах отрабатывает → создаются все файлы run_dir».

### 2.2 Процедура для каждого change (по выбранному варианту)

- [ ] `/opsx-propose "<имя изменения> ..."` (или `openspec new change <имя>` + скрины по инструкциям CLI) → proposal/design/tasks + delta-спеки.
- [ ] `openspec status --change ...` — убедиться, что `applyRequires` = ["specs","design","proposal","tasks"].
- [ ] Спеки писать НА РУССКОМ (решение 1).
- [ ] Дизайн брать из реального кода (строки, модули), как в Этапе 1.1.
- [ ] tasks: для каждого scenario спеки — задача «добавить/проверить тест» + приемочный критерий (см. Этап 3).
- [ ] `/opsx-sync` (переписывает main specs), затем финальный `/opsx-archive` (move+валидация).
- [ ] После каждого change — `openspec validate --all`.

---

## Этап 3 — Заполнить пробелы тестового покрытия

> Все тесты — фреймворк/фикстуры из `tests/conftest.py` (фикстуры blank/binary/synthetic есть).

### 3.1 `utils/image_loader` (`tests/test_image_loader.py`)
- [ ] Тест row-padding: построить синтетическую QImage (или фикстуру) с `bytesPerLine > w×3`, проверить, что `_load_image_pixmap` возвращает None → `load_image` переходит к cv2 fallback (используем синтетический PNG на диск в tmp_path).
- [ ] Тест сообщения об ошибке: `load_image(несуществующий)` → `ValueError`, сообщение содержит путь и фразу «QPixmap и OpenCV».

### 3.2 Новый `tests/test_labeling_controller.py`
- [ ] `list_image_files`: папка с `.png`, `.jpg`, `.webp`, `.gif`, `.svg` → в списке только SUPPORTED, `.gif/.svg` отсутствуют.
- [ ] Папка из одного `.gif` → пустой список.
- [ ] `crop_by_petri`: обрезка с круговой маской у краёв → выходная форма `<2r+1 x 2r+1>`, фон (рядом с кругом) = 0.
- [ ] `get_current_dir`/`get_mask_dir`: сессия с `source/` vs без неё — корректный выбор директории.
- [ ] `export_session_to_zip`: временная сессия → zip содержит все файлы.

### 3.3 `tests/test_classic_algorithms.py`
- [ ] Параметризованный тест 4 алгоритмов с `is_cropped=True` (обрезка-маска возвращается, shape=исходная, dtype=uint8) — аналог существующего `test_each_returns_mask`, но для cropped-ветки.

### 3.4 Сверка: каждый `Сценарий` из каждой спеки → минимум один тест
- [ ] Пройтись по всем delta/main сценариям и внести в таблицу соответствия (файл: `docs/spec-scenario-matrix.md` — создать, gitignore не трогает docs).

---

## Этап 4 — SUPPORTED_EXTENSIONS (вариант А)

> Перед стартом сессии проверено: дублирующий список форматов есть в **двух** местах —
> `ui/controllers/labeling_controller.py:17` и `ui/main_window.py:30` (`SUPPORTED_FORMATS`).
> Единый источник создаётся в `utils/image_loader.py`, оба места импортируют его.

- [ ] Создать в `utils/image_loader.py` модульную константу `SUPPORTED_EXTENSIONS = {".png",".jpg",".jpeg",".bmp",".tiff",".tif",".webp"}`.
- [ ] Перенести/удалить `SUPPORTED_EXTENSIONS` из `ui/controllers/labeling_controller.py` → заменить на `from utils.image_loader import SUPPORTED_EXTENSIONS`.
- [ ] `labeling/labeling_window.py:32` (`SUPPORTED_EXTENSIONS = LabelingController.SUPPORTED_EXTENSIONS`) — заменить на импорт из `utils.image_loader`.
- [ ] `ui/main_window.py`: заменить класс-атрибут `SUPPORTED_FORMATS` (`:30`) на импорт `SUPPORTED_EXTENSIONS` (используется в `:130`, `:218`, `:230`) — заменить либо сделать алиас `SUPPORTED_FORMATS = SUPPORTED_EXTENSIONS`.
- [ ] Обновить спеку `robust-image-loading` в части «Supported image formats»: сформулировать «загрузчик публикует список допустимых расширений (`utils.image_loader.SUPPORTED_EXTENSIONS`), фильтрация списка файлов использует его».
- [ ] Решить вопрос: включать ли в список форматы OpenCV (`.pxm`, `.jp2`). По умолчанию — **НЕ расширять**, только перенос; спек «Supported image formats» переформулировать: загрузчик поддерживает указанный список + любые форматы, декодируемые cv2 (fallback), но UI-фильтр использует именно `SUPPORTED_EXTENSIONS`.
- [ ] Тест в 3.2 использует существующий список (из `utils.image_loader`).

---

## Этап 5 — Верификация

- [ ] `uv run pytest -q` — все существующие (56) + новые тесты зелёные.
- [ ] `uv run ruff check` — весь код чист.
- [ ] `openspec validate --all` — все спеки валидны.
- [ ] Проверить, что `.opencode/`, `opencode.json` НЕ попали в изменения (см. Этап 0).

---

## Этап 6 — Коммит

- [ ] `git add` только: `openspec/**`, `utils/image_loader.py`, `ui/controllers/labeling_controller.py`, `labeling/labeling_window.py`, `tests/`, `docs/implementation-plan.md`, `docs/spec-scenario-matrix.md`.
- [ ] НЕ добавлять: `.opencode/`, `opencode.json`, проверить `git status`.
- [ ] Сообщение по конвенции репозитория (git log показывает `feat:`, `fix:`, `stage N:`):
  например первый коммит: `docs(spec): capability specs for analysis/labeling/testing/train`.

---

## Риски и открытые вопросы

1. **torch/albumentations в тестах**: тренировочные тесты (`train/`) без dev-окружения с torch (~5 ГБ) не запускаются. Варианты: тесты только на `train/config.py`/`train/dataset.py` без torch-импортов не получится (module-level imports torch); поэтому либо (а) пометить `@pytest.mark.dev` и исключить из обычного прогона (по аналогии `@pytest.mark.gui`), либо (б) не покрывать `train/` тестами в этой задаче, оставив только спеки.
2. **Структурные заголовки спек**: валидатор требует английские `## Purpose`/`## Requirements`/`### Requirement:`/`#### Scenario:` — переводу подлежит только текст тела. Унифицировать стиль WHEN/THEN → КОГДА/ТОГДА один раз для всех спек и зафиксировать в `openspec/config.yaml` (rules).
3. **Время на Этап 2**: 5 специй × 3..7 сценариев — это основная работа сессии. Выбран вариант А (3 changes по доменам): последовательно реализовать `core-colony-analysis` → `labeling-and-testing` → `ml-pipeline`, каждый доводить до архива перед переходом к следующему.
4. **`SUPPORTED_EXTENSIONS` / `SUPPORTED_FORMATS`**: перед стартом проверено — список форматов продублирован в `ui/controllers/labeling_controller.py:17` **и** `ui/main_window.py:30`; `ui/analysis_window.py` свой список НЕ держит (фильтр диалога сохранения жёстко `"*.png *.jpg"`). Объединяем все через `utils.image_loader` (см. Этап 4).