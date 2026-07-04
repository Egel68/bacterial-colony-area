# 005. Split test and dev dependencies

Date: 2026-02-04

## Context

Nuitka build step in CI was installing dev dependencies (`--extra dev`),
which pulled torch~2GB unnecessarily into the build environment.
This slowed CI and risked OOM on smaller runners.

## Decision

Split `pyproject.toml` extras:
- `[test]`: pytest, hypothesis, pytest-qt, pytest-mock, pytest-cov
- `[dev]`: torch, torchvision, + all test deps
- CI uses `--extra test`, not `--extra dev`

## Consequences

- CI build env is ~600 MB instead of ~5 GB
- Faster install and cache-restore times
- Build failure risk due to CUDA/torch OOM eliminated
- Developer must explicitly activate `.venv-dev` for training work
