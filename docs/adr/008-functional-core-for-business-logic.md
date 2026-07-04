# 008. Functional Core for business logic

Date: 2026-07-04

## Context

`AnalysisWindow` (482 lines) and `LabelingWindow` (814 lines) mixed
Qt widget code with image processing logic. This made testing
expensive (requires pytest-qt and a display server) and made the
UI classes fragile.

## Decision

Extract business logic into controller classes with zero Qt imports:
- `AnalysisController`: petri dish detection + colony analysis pipeline
- `LabelingController`: image/mask loading, petri detection, cropping, ZIP export
- UI classes delegate to controllers, keeping only layout + signal wiring

## Consequences

- Controllers are testable without `pytest-qt` or display server
- UI classes are ~40% smaller (AnalysisWindow: 482→372 lines after extraction)
- Clear separation: controller owns state, UI owns rendering
- Same pattern can be applied to future features
- Cannot test controller state changes through UI (intentional — controller is the test surface)
