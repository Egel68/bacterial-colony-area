# Bacterial Colony Area Analyzer

Python 3.13+ / PyQt6 / OpenCV / NumPy desktop app.

## Environments

Два виртуальных окружения для разделения runtime и dev-зависимостей:

| Окружение | Команда | Размер | Назначение |
|---|---|---|---|
| `.venv` (runtime) | `uv sync` | ~600 МБ | Запуск приложения, PyInstaller |
| `.venv-dev` (dev) | `UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev` | ~5 ГБ | Разработка, обучение, аугментация |

## Run

```bash
# Runtime (лёгкое окружение)
uv sync
uv run bacteria-analyzer
```

## Augment

```bash
# Из runtime-окружения (работает без torch)
uv run augment
```

## Разработка и обучение (тяжёлое окружение)

```bash
# Установка dev-окружения (runtime + torch/CUDA)
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev

# Запуск из dev-окружения
UV_PROJECT_ENVIRONMENT=.venv-dev uv run python -m train.augment

# Активация dev-окружения
source .venv-dev/bin/activate
```

## Architecture

| Path | Purpose |
|---|---|
| `main.py` | Entrypoint — sets up `QApplication`, dark theme (`ui/styles.py`, Catppuccin Mocha), launches `MainWindow` |
| `ui/main_window.py` | `MainWindow` — file picker, validates image extensions, opens `AnalysisWindow` |
| `ui/analysis_window.py` | `AnalysisWindow` — image display + parameter sliders, 4 view modes (original/contrast/binary/result-overlay), save output |
| `analysis/image_processor.py` | Static methods: grayscale, CLAHE, green-channel extraction, median blur, resize |
| `analysis/colony_detector.py` | `detect_petri_dish()` via bright-reflection contours → HoughCircles fallback; `detect_colonies()` via green channel + CLAHE + adaptive threshold + morphology |
| `utils/calculations.py` | `AreaCalculator` — defaults to 90mm dish diameter; `calculate_areas()` returns colony count, px/mm² area, coverage % |
| `test_data/` | Sample PNG screenshots |
| `labeling/` | `LabelingWindow` — manual ground-truth mask annotation with configurable brush (draw/erase), file navigation, petri dish auto-detection + crop, saves binary PNG masks |
| `test_images/source/` | Input images for labeling (place originals here) |
| `test_images/masks/` | Output binary masks for source (`{filename}_mask.png`, 255=colony) |
| `test_images/cropped/` | Cropped petri dish images (created via «Обрезать по чашке») |
| `test_images/cropped_masks/` | Output binary masks for cropped images |
| `train/` | Training module: `augment.py`, `dataset.py`, `model.py`, `train.py`, `predict.py` |
| `train/augment.py` | Generates 15 augmented pairs per original (Rotate, Flip, Affine, Brightness/Contrast, Noise, Blur, Hue/Sat) |
| `train/data/images/` | Augmented images for U-Net training |
| `train/data/masks/` | Corresponding augmented masks |

## Detection pipeline

`MainWindow` → `AnalysisWindow(image_path)` → `cv2.imread` → `detect_petri_dish` → `detect_colonies` (reads UI sliders) → `calculate_areas` → display

## UI parameters (re-read on every "Recalculate" click)

- **Sensitivity** (1–100%, mapped to `k = 3.0 - sens*2.5`)
- **Contrast** (0.5–3.0x, maps to CLAHE clip limit)
- **Edge margin** (0–30%, shrinks ROI from dish edge)
- **Min colony size** (px, filtered via connected components)
- **Solid fill** toggle + fill strength (closing kernel radius, for confluent lawns)

## CI

`.github/workflows/build.yaml` — PyInstaller via `uv` on push to `develop`/`main`. Python 3.13, matrix: ubuntu + windows. Собирается runtime-окружение (без torch/CUDA), ~200–350 МБ. No tests, linting, typechecking, or formatting gates.

## What is NOT configured

No tests, no linter, no typechecker, no formatter, no pre-commit hooks. CI does not verify code quality.

## Conventions

- Code comments are in Russian.
- Adding `test/`, lint/config, or CI quality gates is new territory — no existing patterns to follow.
