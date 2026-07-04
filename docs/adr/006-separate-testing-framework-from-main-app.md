# 006. Separate testing framework from main app

Date: 2026-01-25

## Context

Need to benchmark multiple detection algorithms against ground-truth masks.

## Decision

Create `testing/` package as a standalone framework:
- `BaseDetectionAlgorithm` ABC for pluggable algorithms
- `@register_algorithm` decorator for auto-discovery
- `TestDataset` for loading paired images + GT masks
- `runner.py` with `run_all()` and `compute_summary()`
- CLI entry point `test-algorithms` in `pyproject.toml`

## Consequences

- New algorithms can be added with a single decorated class
- Reports are generated as standalone HTML with Chart.js (no server needed)
- Framework is independent of main app — usable in CI without PyQt6
- Registry pattern (side-effect import) required: `testing/__init__.py` imports all algorithms
