## ADDED Requirements

### Requirement: CI build free of known log warnings

Система SHALL завершать CI-сборку Nuitka (ubuntu-latest и windows-latest) без предупреждений, известных на момент реализации: deprecation Node.js 20/24 от Actions, `Nuitka-Scons: not using ccache` (linux), git-hints про `init.defaultBranch` и `PytestCollectionWarning`. Логи сборки SHALL NOT содержать строк `warning:`/`error` от инструментов, контролируемых проектом (workflow-шаги, версии actions, pytest-запуск).

#### Scenario: Node deprecation annotations absent
- **КОГДА** сборка запущена после обновления `actions/checkout` и `actions/upload-artifact` на node24-версии (`v7`)
- **ТОГДА** в логах и в разделе `ANNOTATIONS` run SHALL NOT быть сообщений «Node.js 20 is deprecated» и `[DEP0040]`/`[DEP0169]`

#### Scenario: No ccache warning on linux
- **КОГДА** Linux-job выполняет сборку Nuitka после установки `ccache`
- **ТОГДА** лог SHALL NOT содержать `Nuitka-Scons:WARNING: You are not using ccache`

#### Scenario: No pytest collection warning
- **КОГДА** pytest запускается с маской `not slow and not gui`
- **ТОГДА** в сводке SHALL NOT появляться `PytestCollectionWarning` и строка `1 warning`

#### Scenario: No git default branch hint
- **КОГДА** на runner выполняется шаг перед checkout с `git config --global init.defaultBranch main`
- **ТОГДА** лог checkout SHALL NOT содержать hint «Using 'master' as the name for the initial branch»