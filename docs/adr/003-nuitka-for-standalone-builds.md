# 003. Nuitka for standalone builds

Date: 2026-01-20

## Context

Need a standalone executable that end users can run without Python.

## Decision

Use Nuitka instead of PyInstaller as primary build tool.

## Consequences

- Nuitka compiles to real machine code (via C++ compiler)
- Better PyQt6 support than PyInstaller (fewer hidden imports to debug)
- Slower build time (5-10 min vs 1-2 min for PyInstaller)
- Larger binary (~120 MB vs ~80 MB for PyInstaller) but faster startup
- PyInstaller .spec kept as fallback build option
