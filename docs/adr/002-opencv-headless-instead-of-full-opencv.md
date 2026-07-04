# 002. opencv-python-headless instead of full opencv-python

Date: 2026-01-15

## Context

Image processing requires OpenCV. Full `opencv-python` includes GUI
windowing (HighGui) which we don't use (Qt handles all display).

## Decision

Use `opencv-python-headless` instead of `opencv-python`.

## Consequences

- Smaller dependency footprint (~20 MB saved)
- Fewer system library dependencies
- Both packages share the same core API; zero code impact
