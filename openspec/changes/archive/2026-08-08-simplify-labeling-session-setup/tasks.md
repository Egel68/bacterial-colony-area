## 1. Управляющая логика сессий

- [x] 1.1 Создать `labeling/session_manager.py` с константами `DEFAULT_SESSION_ROOT = Path.home() / "BacteriaLabeling"`, `SESSION_SUBDIRS = ("source", "masks", "cropped", "cropped_masks")` и функцией `ensure_session_structure(session_dir)` (создаёт корень и поддиректории)
- [x] 1.2 Реализовать `sanitize_name(name)`: очистка от недопустимых символов Windows (`<>:"/\|?*`), обрезка до 60 символов, пустое имя → `None`
- [x] 1.3 Реализовать `SessionManager` с инъектируемым путём к JSON-конфигу: `list_recent()`, `add(session_dir, name)`, `remove(path)`, `session_path(name)`, `current_root`, `set_root(path)`
- [x] 1.4 `SessionManager`: дедупликация по пути/имени, сортировка от свежих к старым, лимит 10 записей, сохранение после изменений
- [x] 1.5 Обработка отсутствующего/повреждённого JSON: возвращать пустой список/дефолтный корень; создавать родительскую папку при записи
- [x] 1.6 Сохранять переопределённый `session_root` в конфиге; при отсутствии использовать `DEFAULT_SESSION_ROOT`

## 2. Упрощение контроллера и окна разметки

- [x] 2.1 В `LabelingController.get_current_dir` убрать fallback на корень сессии: всегда возвращать поддиректорию `source/` или `cropped/`, создавая её при отсутствии
- [x] 2.2 В `LabelingController.get_mask_dir` всегда возвращать `masks/` или `cropped_masks/` (создавая при необходимости)
- [x] 2.3 В `LabelingWindow.__init__` вызывать `ensure_session_structure(self.session_dir)` и установить `self.source_dir = self.session_dir / "source"`
- [x] 2.4 Показывать полный путь к корню сессии в заголовке окна и текущей рабочей поддиректории в левой панели; обновлять при `_on_mode_changed`

## 3. Диалог настройки сессии

- [x] 3.1 Создать `ui/labeling_session_dialog.py` с классом `LabelingSessionDialog(QDialog)`: поле «Название сессии», подпись «Сессии сохраняются в: <root>», список «Недавние сессии»
- [x] 3.2 Кнопка «Создать и открыть» доступна только при непустом валидном названии; Enter в поле = создание
- [x] 3.3 Обработка существующего имени: предупреждение с выбором «Открыть существующую» или изменить название
- [x] 3.4 Список недавних сессий: клик/двойной клик открывает сессию, кнопка «Открыть» для выбранной
- [x] 3.5 Кнопка «📂 Открыть существующую папку…» через `QFileDialog.getExistingDirectory`
- [x] 3.6 Кнопка «✏️ Изменить» рядом с подписью хранилища: выбор нового корня сессий и сохранение через `SessionManager.set_root`

## 4. Интеграция в главном окне

- [x] 4.1 В `MainWindow._open_labeling` заменить `QFileDialog.getExistingDirectory` на модальный `LabelingSessionDialog`
- [x] 4.2 По результату диалога: `ensure_session_structure`, добавление в `SessionManager.list_recent` через `add`, открытие `LabelingWindow`

## 5. Тесты

- [x] 5.1 Создать `tests/test_session_manager.py`: санитизация имён, создание структуры, самосоздание поддиректорий
- [x] 5.2 Тесты `SessionManager`: добавление/дедупликация/порядок/лимит 10, чтение повреждённого JSON как пустого, выживание между «запусками» (пересоздание из файла)
- [x] 5.3 Тесты `SessionManager`: переопределение `session_root` сохраняется в конфиг и восстанавливается при новом экземпляре; дефолт — `DEFAULT_SESSION_ROOT`
- [x] 5.4 Тесты `get_current_dir`/`get_mask_dir`: возвращают поддиректории `source/`/`cropped/` даже если они пустые или отсутствуют; корневой fallback удалён

## 6. Сборка и проверка

- [x] 6.1 Проверить, что новые файлы (`labeling/session_manager.py`, `ui/labeling_session_dialog.py`) попадают в Nuitka/PyInstaller-сборку (скрипты `scripts/nuitka_build.py`, `scripts/build_pyinstaller.sh`)
- [x] 6.2 Локально собрать бинарник и проверить поток: создание новой сессии, повторное открытие из недавних, кнопка «Открыть существующую папку»
- [x] 6.3 Обновить раздел «Разметка» в `AGENTS.md`/документации, если изменилось описание workflow