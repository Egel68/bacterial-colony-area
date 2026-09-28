## Why

CI-сборка в GitHub Actions (run #31299365276) завершается успешно, но в логах присутствуют предупреждения: Node.js 20 deprecation (`actions/checkout@v4`, `actions/upload-artifact@v4`), `PytestCollectionWarning` от класса `TestDataset`, `Nuitka-Scons: not using ccache` и git-hint про дефолтную ветку `master`. Требование пользователя: лог сборки «идеальный» — без warning, error и прочих неточностей.

## What Changes

- Обновить `actions/checkout` и `actions/upload-artifact` с `@v4` на версии с рантаймом `node24` (v6+/v7), устранив deprecation-предупреждения Node.js 20 и производные `[DEP0040] punycode` / `[DEP0169] url.parse`.
- Установить `ccache` на Linux-runner, чтобы Nuitka перестал писать `Nuitka-Scons:WARNING: not using ccache`.
- Подавить git-hint про `init.defaultBranch=master`, настроив `git config --global init.defaultBranch main` до шага checkout.
- Устранить `PytestCollectionWarning` про класс `TestDataset`: переименовать класс (или добавить `__test__ = False`), чтобы pytest не пытался коллектировать его как тестовый.
- Сохранить текущий набор steps, флагов Nuitka и артефактов — только устранение предупреждений.

## Capabilities

### New Capabilities
- `warning-free-ci-build`: гарантия, что CI-сборка Nuitka (ubuntu + windows) не порождает предупреждений в логах — deprecations от Actions, Nuitka/ccache, git-hints и предупреждений pytest.

### Modified Capabilities
- `nu-packaging`: требование «CI build matrix and quality gates» дополняется критерием чистоуровневого лога — сборка SHALL завершаться без known warnings (Node 20 deprecation, ccache, pytest collection).
- `algorithm-testing`: требование «Test dataset loading» — загрузчик (`TestDataset`) SHALL NOT экранироваться собранием как pytest test-класс и SHALL NOT порождать `PytestCollectionWarning`.

## Impact

- `.github/workflows/build.yaml` — версии actions (checkout, upload-artifact), установка `ccache`, git config для `init.defaultBranch`.
- `testing/dataset.py` и `tests/test_classic_algorithms.py` (и другие референсы `TestDataset`) — переименование/`__test__ = False`.
- Система Nuitka и uv останутся без изменений; результат сборки (артефакты) не меняется.