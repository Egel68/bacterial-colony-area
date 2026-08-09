"""
Главное окно приложения.
Содержит элементы для выбора файла и запуска анализа.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .analysis_window import AnalysisWindow
from .labeling_session_dialog import LabelingSessionDialog
from utils.image_loader import SUPPORTED_EXTENSIONS


class MainWindow(QMainWindow):
    """Главное окно приложения для выбора изображения."""

    def __init__(self):
        super().__init__()
        self.selected_file_path = None
        self.analysis_window = None
        self._init_ui()

    def _init_ui(self):
        """Инициализация пользовательского интерфейса."""
        self.setWindowTitle("Bacteria Colony Analyzer")
        self.setMinimumSize(900, 500)
        self.resize(700, 450)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)

        # Заголовок
        self._create_header(main_layout)

        # Разделитель
        main_layout.addSpacing(10)

        # Секция выбора файла
        self._create_file_section(main_layout)

        # Информация о форматах
        self._create_format_info(main_layout)

        # Растяжка
        main_layout.addStretch()

        # Кнопка анализа
        self._create_analyze_button(main_layout)

    def _create_header(self, layout: QVBoxLayout):
        """Создание заголовка окна."""
        # Заголовок
        title = QLabel("🔬 Анализатор бактериальных колоний")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Подзаголовок
        subtitle = QLabel(
            "Загрузите изображение чашки Петри для анализа колоний бактерий"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

    def _create_file_section(self, layout: QVBoxLayout):
        """Создание секции выбора файла."""
        # Контейнер для выбора файла
        file_frame = QFrame()
        file_frame.setStyleSheet("""
            QFrame {
                background-color: #313244;
                border-radius: 12px;
                padding: 10px;
            }
        """)
        file_layout = QVBoxLayout(file_frame)
        file_layout.setContentsMargins(20, 20, 20, 20)
        file_layout.setSpacing(15)
        # Метка
        file_label = QLabel("📁 Путь к изображению:")
        file_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        file_layout.addWidget(file_label)

        # Горизонтальный layout для поля ввода и кнопки
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        # Поле ввода пути
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Выберите файл изображения...")
        self.path_input.setReadOnly(True)
        self.path_input.textChanged.connect(self._on_path_changed)
        input_layout.addWidget(self.path_input)

        # Кнопка открытия
        self.open_button = QPushButton("📂 Открыть")
        self.open_button.setObjectName("secondary")
        self.open_button.clicked.connect(self._open_file_dialog)
        self.open_button.setMinimumWidth(130)
        input_layout.addWidget(self.open_button)

        file_layout.addLayout(input_layout)
        layout.addWidget(file_frame)

    def _create_format_info(self, layout: QVBoxLayout):
        """Создание информации о поддерживаемых форматах."""
        formats_text = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        info_label = QLabel(f"ℹ️ Поддерживаемые форматы: {formats_text}")
        info_label.setObjectName("info")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

    def _create_analyze_button(self, layout: QVBoxLayout):
        """Создание кнопки анализа."""
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.analyze_button = QPushButton("🔍 Анализировать")
        self.analyze_button.setObjectName("success")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setMinimumSize(200, 50)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                border-radius: 12px;
            }
        """)
        self.analyze_button.clicked.connect(self._start_analysis)
        button_layout.addWidget(self.analyze_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Кнопка разметки тестовых изображений
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.label_button = QPushButton("✏️ Разметка тестовых изображений")
        self.label_button.setObjectName("secondary")
        self.label_button.setMinimumSize(260, 40)
        self.label_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                border-radius: 10px;
            }
        """)
        self.label_button.clicked.connect(self._open_labeling)
        label_layout.addWidget(self.label_button)
        label_layout.addStretch()
        layout.addLayout(label_layout)

    def _open_labeling(self):
        from labeling import LabelingWindow
        from labeling.session_manager import SessionManager

        manager = SessionManager()
        dialog = LabelingSessionDialog(manager, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.session_dir:
            self.labeling_window = LabelingWindow(dialog.session_dir, self)
            self.labeling_window.show()

    def _open_file_dialog(self):
        """Открытие диалога выбора файла."""
        formats_filter = "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение чашки Петри", str(Path.home()), formats_filter
        )

        if file_path:
            self.path_input.setText(file_path)
            self.selected_file_path = file_path

    def _on_path_changed(self, text: str):
        """Обработка изменения пути к файлу."""
        is_valid = self._validate_file(text)
        self.analyze_button.setEnabled(is_valid)

        if text and not is_valid:
            self.path_input.setStyleSheet("border-color: #f38ba8;")
        else:
            self.path_input.setStyleSheet("")

    def _validate_file(self, file_path: str) -> bool:
        """Проверка валидности файла."""
        if not file_path:
            return False

        path = Path(file_path)

        # Проверяем существование файла
        if not path.exists():
            return False

        # Проверяем расширение
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        return True

    def _start_analysis(self):
        """Запуск анализа изображения."""
        if not self.selected_file_path:
            self._show_error("Файл не выбран", "Пожалуйста, выберите файл изображения.")
            return

        if not self._validate_file(self.selected_file_path):
            formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            self._show_error(
                "Неверный формат файла",
                f"Выбранный файл имеет неподдерживаемый формат.\n\n"
                f"Поддерживаемые форматы:\n{formats}",
            )
            return

        # Открываем окно анализа
        try:
            self.analysis_window = AnalysisWindow(self.selected_file_path, self)
            self.analysis_window.show()
        except Exception as e:
            self._show_error(
                "Ошибка загрузки", f"Не удалось загрузить изображение:\n{str(e)}"
            )

    def _show_error(self, title: str, message: str):
        """Показать сообщение об ошибке."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
