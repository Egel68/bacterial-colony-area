# 001. Use PyQt6 over PySide6

Date: 2026-01-15

## Context

Desktop GUI framework needed for bacterial colony analyzer.
Two viable options: PyQt6 (Riverbank) and PySide6 (Qt official).

## Decision

Use PyQt6.

## Consequences

- GPL-licensed (acceptable for standalone binary distribution)
- Larger ecosystem of examples and StackOverflow answers
- PySide6 stubs are more complete, but PyQt6 is more battle-tested in scientific image processing apps
- pytest-qt works with both; no practical difference for this project
