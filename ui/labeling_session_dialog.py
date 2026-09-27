"""
Диалог «Новая сессия разметки»: создание сессии по имени,
смена расположения хранилища, открытие недавних и произвольной папки.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QBoxLayout,
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
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from labeling.session_manager import (
    SessionManager,
    ensure_session_structure,
    sanitize_name,
)
from .responsive import ResponsiveMetrics, install_application_responsive_sizing


class LabelingSessionDialog(QDialog):
    """Выбор или создание сессии разметки."""

    def __init__(self, manager: SessionManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.session_dir: Optional[Path] = None
        self.setWindowTitle("Новая сессия разметки")
        self.setMinimumSize(0, 0)
        self._init_ui()
        self._responsive_sizer = install_application_responsive_sizing(
            self, reference_size=(640, 480), minimum_scale=0.75
        )
        self._reload_recents()
        self._update_root_label()
        self.name_edit.setFocus()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.dialog_layout = layout
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("sessionContentScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setMinimumSize(0, 0)
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        body = QWidget()
        body.setMinimumWidth(0)
        body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        title = QLabel("Новая сессия разметки")
        title.setProperty("responsiveRole", "title")
        title.setStyleSheet("font-weight: bold;")
        body_layout.addWidget(title)

        hint = QLabel("Введите название — папка сессии создастся автоматически.")
        hint.setProperty("responsiveRole", "subtitle")
        hint.setStyleSheet("color: #a6adc8;")
        hint.setWordWrap(True)
        body_layout.addWidget(hint)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например: День 1 — Чашка А")
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.name_edit.returnPressed.connect(self._on_create)
        self.name_edit.setMinimumWidth(0)
        body_layout.addWidget(self.name_edit)

        root_frame = QFrame()
        root_frame.setStyleSheet(
            "QFrame { background-color: #1e1e2e; border-radius: 8px; }"
        )
        rl = QVBoxLayout(root_frame)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)

        root_caption = QLabel("Сессии сохраняются в:")
        root_caption.setProperty("responsiveRole", "subtitle")
        root_caption.setStyleSheet("color: #a6adc8;")
        rl.addWidget(root_caption)

        self.root_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.root_row.setSpacing(8)
        self.root_label = QLabel()
        self.root_label.setProperty("responsiveRole", "subtitle")
        self.root_label.setWordWrap(True)
        self.root_row.addWidget(self.root_label, stretch=1)

        self.btn_change_root = QPushButton("✏️ Изменить")
        self.btn_change_root.setObjectName("secondary")
        self.btn_change_root.setMinimumWidth(0)
        self.btn_change_root.clicked.connect(self._on_change_root)
        self.root_row.addWidget(self.btn_change_root)
        rl.addLayout(self.root_row)

        body_layout.addWidget(root_frame)

        recent_caption = QLabel("Недавние сессии:")
        recent_caption.setProperty("responsiveRole", "subtitle")
        recent_caption.setStyleSheet("color: #a6adc8;")
        body_layout.addWidget(recent_caption)

        self.recent_list = QListWidget()
        # Не даём layout сжать список до нулевой высоты в диалоге 400×300;
        # общий responsive sizer масштабирует этот базовый минимум.
        self.recent_list.setMinimumSize(0, 72)
        self.recent_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.recent_list.itemDoubleClicked.connect(self._on_open_recent)
        body_layout.addWidget(self.recent_list, stretch=1)

        self.recent_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.recent_row.setSpacing(8)
        self.btn_open_recent = QPushButton("Открыть выбранную")
        self.btn_open_recent.setMinimumWidth(0)
        self.btn_open_recent.clicked.connect(self._on_open_recent_clicked)
        self.recent_row.addWidget(self.btn_open_recent)

        self.btn_open_folder = QPushButton("📂 Открыть существующую папку…")
        self.btn_open_folder.setObjectName("secondary")
        self.btn_open_folder.setMinimumWidth(0)
        self.btn_open_folder.clicked.connect(self._on_open_existing_folder)
        self.recent_row.addWidget(self.btn_open_folder)
        body_layout.addLayout(self.recent_row)

        self.content_scroll.setWidget(body)
        layout.addWidget(self.content_scroll, stretch=1)

        self.action_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setObjectName("secondary")
        self.btn_cancel.setMinimumWidth(0)
        self.btn_cancel.clicked.connect(self.reject)
        self.action_row.addWidget(self.btn_cancel)

        self.create_button = QPushButton("Создать и открыть")
        self.create_button.setObjectName("success")
        self.create_button.setMinimumWidth(0)
        self.create_button.clicked.connect(self._on_create)
        self.create_button.setEnabled(False)
        self.action_row.addWidget(self.create_button)
        layout.addLayout(self.action_row)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.contentsRect().width()
        narrow = width < 620
        compact = width < 360
        direction = (
            QBoxLayout.Direction.TopToBottom
            if narrow
            else QBoxLayout.Direction.LeftToRight
        )
        self.recent_row.setDirection(direction)
        self.root_row.setDirection(direction)
        self.action_row.setDirection(
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        metrics = getattr(self._responsive_sizer, "metrics", None) or ResponsiveMetrics(
            1.0
        )
        margin = metrics.dimension(8 if narrow else 12, minimum=1)
        self.dialog_layout.setContentsMargins(margin, margin, margin, margin)

    def _on_name_changed(self, text: str):
        self.create_button.setEnabled(sanitize_name(text) is not None)

    def _update_root_label(self):
        root = str(self.manager.current_root)
        self.root_label.setText(root)
        self.root_label.setToolTip(root)

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
