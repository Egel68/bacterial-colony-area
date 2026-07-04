# 004. Qt-based image loader instead of cv2.imread

Date: 2026-02-04

## Context

`cv2.imread()` fails in Nuitka standalone binary because OpenCV I/O
plugins (libjpeg, libpng) are not bundled by Nuitka.

## Decision

Replace `cv2.imread` with `QPixmap`/`QImage`-based loader
in `utils/image_loader.py`.

## Consequences

- Standalone binary loads images correctly without bundling codec libraries
- Slightly slower for trivial files, negligible difference in practice
- Requires `pytest-qt` for testing the loader (GUI test marker)
- No need for `--include-package` workarounds for codecs
