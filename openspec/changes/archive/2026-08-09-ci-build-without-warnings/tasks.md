## 1. CI Workflow: node24 actions

- [x] 1.1 Обновить `actions/checkout@v4` → `actions/checkout@v7` в `.github/workflows/build.yaml`
- [x] 1.2 Обновить `actions/upload-artifact@v4` → `actions/upload-artifact@v7` (оба места: Linux и Windows)

## 2. CI Workflow: ccache и git-local конфиг

- [x] 2.1 Добавить `ccache` в `apt-get install` шага «Install system dependencies (Linux)» (`g++ patchelf ccache`)
- [x] 2.2 Добавить первый шаг в каждый job: `git config --global init.defaultBranch main` (перед `actions/checkout`), чтобы подавить git-hint

## 3. Устранение PytestCollectionWarning

- [x] 3.1 Добавить атрибут `__test__ = False` в класс `TestDataset` в `testing/dataset.py`
- [x] 3.2 Прогнать `uv run python -m pytest tests/ -m "not slow and not gui"` локально и убедиться, что в сводке нет «1 warning» и `PytestCollectionWarning`

## 4. Валидация «0 warning»

- [x] 4.1 Прогнать локально ruff: `uv run ruff check . && uv run ruff format --check .`
- [x] 4.2 Запустить CI (push в `develop`), скачать логи обоих job-ов и выполнить grep `warning|error|deprecat|PytestCollection|ccache|Node.js 20 is deprecated` — результат SHALL быть пустым (исключая технические команды `tar --warning=no-unknown-keyword`)
- [x] 4.3 Проверить, что в разделе `ANNOTATIONS` run нет сообщений уровня `error`, а артефакты `BacteriaAnalyzer-Linux`/`BacteriaAnalyzer-Windows` загружены