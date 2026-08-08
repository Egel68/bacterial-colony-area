"""
Диалог «Новая сессия разметки»: создание сессии по имени,
смена расположения хранилища, открытие недавних и произвольной папки.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from labeling.session_manager import (
    SessionManager,
    ensure_session_structure,
    sanitize_name,
)


class LabelingSessionDialog(QDialog):
    """Выбор или создание сессии разметки."""

    def __init__(self, manager: SessionManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.session_dir: Optional[Path] = None
        self.setWindowTitle("Новая сессия разметки")
        self.setMinimumWidth(460)
        self._init_ui()
        self._reload_recents()
        self._update_root_label()
        self.name_edit.setFocus()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Новая сессия разметки")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(title)

        hint = QLabel("Введите название — папка сессии создастся автоматически.")
        hint.setStyleSheet("color: #a6adc8; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например: День 1 — Чашка А")
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.name_edit.returnPressed.connect(self._on_create)
        layout.addWidget(self.name_edit)

        root_frame = QFrame()
        root_frame.setStyleSheet(
            "QFrame { background-color: #1e1e2e; border-radius: 8px; }"
        )
        rl = QVBoxLayout(root_frame)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)

        root_caption = QLabel("Сессии сохраняются в:")
        root_caption.setStyleSheet("color: #a6adc8; font-size: 11px;")
        rl.addWidget(root_caption)

        root_row = QHBoxLayout()
        root_row.setSpacing(8)
        self.root_label = QLabel()
        self.root_label.setStyleSheet("font-size: 12px;")
        self.root_label.setWordWrap(True)
        root_row.addWidget(self.root_label, stretch=1)

        btn_change_root = QPushButton("✏️ Изменить")
        btn_change_root.setStyleSheet("background-color: #45475a; padding: 4px 8px;")
        btn_change_root.clicked.connect(self._on_change_root)
        root_row.addWidget(btn_change_root)
        rl.addLayout(root_row)

        layout.addWidget(root_frame)

        recent_caption = QLabel("Недавние сессии:")
        recent_caption.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(recent_caption)

        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._on_open_recent)
        layout.addWidget(self.recent_list, stretch=1)

        recent_row = QHBoxLayout()
        recent_row.setSpacing(8)
        btn_open_recent = QPushButton("Открыть выбранную")
        btn_open_recent.clicked.connect(self._on_open_recent_clicked)
        recent_row.addWidget(btn_open_recent)

        recent_row.addStretch()

        btn_open_folder = QPushButton("📂 Открыть существующую папку…")
        btn_open_folder.setStyleSheet("background-color: #45475a; padding: 6px 10px;")
        btn_open_folder.clicked.connect(self._on_open_existing_folder)
        recent_row.addWidget(btn_open_folder)
        layout.addLayout(recent_row)

        buttons = QHBoxLayout()
        buttons.addStretch()
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_cancel)

        self.create_button = QPushButton("Создать и открыть")
        self.create_button.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "padding: 8px 16px;"
        )
        self.create_button.clicked.connect(self._on_create)
        self.create_button.setEnabled(False)
        buttons.addWidget(self.create_button)
        layout.addLayout(buttons)

    def _on_name_changed(self, text: str):
        self.create_button.setEnabled(sanitize_name(text) is not None)

    def _update_root_label(self):
        self.root_label.setText(str(self.manager.current_root))

    def _reload_recents(self):
        self.recent_list.clear()
        for rec in self.manager.recent_sessions:
            item = QListWidgetItem(rec["name"])
            item.setToolTip(rec["path"])
            item.setData(Qt.ItemDataRole.UserRole, rec["path"])
            self.recent_list.addItem(item)

    def _on_open_recent(self, item: QListWidgetItem):
        self._open_recent_item(item)

    def _on_open_recent_clicked(self):
        item = self.recent_list.currentItem()
        if item is not None:
            self._open_recent_item(item)

    def _open_recent_item(self, item: QListWidgetItem):
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        name = item.text()
        if not path.is_dir():
            QMessageBox.warning(self, "Сессия недоступна", f"Папка не найдена:\n{path}")
            self.manager.remove(path)
            self._reload_recents()
            return
        self.manager.add(path, name)
        self._accept_session(path)

    def _on_open_existing_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку сессии",
            str(self.manager.current_root),
        )
        if not folder:
            return
        path = ensure_session_structure(Path(folder))
        self.manager.add(path, path.name)
        self._accept_session(path)

    def _on_change_root(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку хранения сессий",
            str(self.manager.current_root),
        )
        if folder:
            self.manager.set_root(Path(folder))
            self._update_root_label()

    def _on_create(self):
        name = sanitize_name(self.name_edit.text())
        if name is None:
            return
        path = self.manager.session_path(name)
        if path.exists():
            answer = QMessageBox.question(
                self,
                "Сессия существует",
                f"Сессия «{name}» уже существует.\n\nОткрыть её?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        ensure_session_structure(path)
        self.manager.add(path, name)
        self._accept_session(path)

    def _accept_session(self, path: Path):
        self.session_dir = Path(path).resolve()
        self.accept()
