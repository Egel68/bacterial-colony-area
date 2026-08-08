# image-labeling Specification (delta for simplify-labeling-session-setup)

## MODIFIED Requirements

### Requirement: Session directory structure

Для режима «исходники» (`source`) система SHALL использовать поддиректорию `source/` сессии как рабочую директорию. Для режима «обрезки» (`cropped`) система SHALL использовать поддиректорию `cropped/`. Если соответствующая поддиректория отсутствует, система SHALL создать её перед использованием. Для масок система SHALL использовать `masks/` (для source) или `cropped_masks/` (для cropped), создавая директорию при необходимости.

#### Scenario: Source dir as working directory

- **КОГДА** пользователь находится в режиме «исходники»
- **ТОГДА** `get_current_dir(session, "source")` SHALL вернуть `source/` (даже если в ней нет файлов)

#### Scenario: Cropped dir as working directory

- **КОГДА** пользователь находится в режиме «обрезки»
- **ТОГДА** `get_current_dir(session, "cropped")` SHALL вернуть `cropped/`

#### Scenario: Missing source dir auto-created

- **КОГДА** в сессии отсутствует `source/`
- **ТОГДА** система SHALL создать `source/` перед загрузкой списка файлов и вернуть её

#### Scenario: Mask directory selection

- **КОГДА** запрашивается маска для source
- **ТОГДА** `get_mask_dir(session, "source")` SHALL вернуть `masks/`