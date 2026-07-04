# 007. Typed dataclasses instead of dicts

Date: 2026-07-04

## Context

Core data structures (`PetriInfo`, `AnalysisParams`, analysis results)
were passed as plain dicts with string keys. No validation, no IDE
autocomplete, no documentation of required fields.

## Decision

Replace all dict-based data transfer objects with frozen dataclasses:
- `PetriInfo` (cx, cy, radius, image_shape) with `__post_init__` validation
- `AnalysisParams` (sensitivity, contrast, margin_percent, ...) with range checks
- `AnalysisResult` (colony_count, coverage_percent, ...) as frozen struct
- `AppConfig` / `AnalysisDefaults` for configuration constants

## Consequences

- All callers get guaranteed structure and IDE autocompletion
- Validation at construction time catches invalid values early
- No more `KeyError` risks from misspelled string keys
- Slightly more verbose construction, significantly safer usage
- `dataclasses.frozen=True` ensures immutability across threads
