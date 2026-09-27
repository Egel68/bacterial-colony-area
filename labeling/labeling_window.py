import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent

from analysis.geometry import PetriInfo
from labeling.session_manager import ensure_session_structure
from ui.controllers.labeling_controller import LabelingController
from ui.responsive import (
    install_application_responsive_sizing,
    set_responsive_stylesheet,
)
from utils.image_loader import load_image
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PaintLabel(QLabel):
    ZOOM_MIN = 0.1
    ZOOM_MAX = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(0, 0)
        self.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._base_zoom_minimum = QSize(0, 0)

        self._image = None
        self._mask = None
        self.brush_size = 20
        self.is_drawing = True
        self._painting = False
        self._zoom_factor = 1.0
        self._fit_scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._original_pixmap = None
        self.petri_info = None
        self.show_petri_circle = False
        self.view_mode = 0

    def set_image(self, image: np.ndarray, mask: np.ndarray = None):
        self._image = image.copy()
        h, w = image.shape[:2]
        if mask is not None and mask.shape[:2] == (h, w):
            self._mask = mask.copy()
        else:
            self._mask = np.zeros((h, w), dtype=np.uint8)
        self._zoom_factor = 1.0
        self._render()

    def set_petri_info(self, info: Optional[PetriInfo]):
        self.petri_info = info
        self._render()

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def clear_mask(self):
        if self._mask is not None:
            self._mask.fill(0)
            self._render()

    def zoom_reset(self):
        self._zoom_factor = 1.0
        self._update_scaled()

    def zoom_in(self):
        self._zoom_factor = min(self.ZOOM_MAX, self._zoom_factor * self.ZOOM_STEP)
        self._update_scaled()

    def zoom_out(self):
        self._zoom_factor = max(self.ZOOM_MIN, self._zoom_factor / self.ZOOM_STEP)
        self._update_scaled()

    @property
    def zoom_percent(self) -> int:
        return int(self._zoom_factor * 100)

    def _render(self):
        if self._image is None:
            return

        if self.view_mode == 1:
            display = cv2.cvtColor(self._mask, cv2.COLOR_GRAY2BGR)
        else:
            display = self._image.copy()
            green = np.zeros_like(display)
            green[self._mask > 0] = [0, 255, 0]
            display = cv2.addWeighted(display, 1.0, green, 0.35, 0)

            if self.show_petri_circle and self.petri_info is not None:
                center = self.petri_info.center
                radius = self.petri_info.radius
                cv2.circle(display, center, radius, (255, 100, 100), 2)

        h, w = display.shape[:2]
        q_img = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(q_img)
        self._update_scaled()

    def _update_scaled(self):
        if self._original_pixmap is None:
            return

        orig_w = self._original_pixmap.width()
        orig_h = self._original_pixmap.height()

        scroll_area = getattr(self, "scroll_area", None)
        viewport_size = (
            scroll_area.viewport().size() if scroll_area is not None else self.size()
        )

        fit_scale = min(
            viewport_size.width() / orig_w,
            viewport_size.height() / orig_h,
        )

        self._fit_scale = fit_scale
        current_scale = fit_scale * self._zoom_factor

        new_w = max(1, int(orig_w * current_scale))
        new_h = max(1, int(orig_h * current_scale))

        scaled = self._original_pixmap.scaled(
            new_w,
            new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._scale = orig_w / new_w

        # Увеличение картинки должно прокручиваться внутри viewport, не
        # передавая её размер в sizeHint родительского окна.
        self.setMinimumSize(self._base_zoom_minimum)

        self._offset_x = (self.width() - scaled.width()) / 2.0
        self._offset_y = (self.height() - scaled.height()) / 2.0

        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def wheelEvent(self, event):
        if self._original_pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()

    def _widget_to_image(self, pos: QPoint):
        x = int((pos.x() - self._offset_x) * self._scale)
        y = int((pos.y() - self._offset_y) * self._scale)
        return x, y

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._mask is not None:
            self._painting = True
            self._paint_at(self._widget_to_image(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._painting and self._mask is not None:
            self._paint_at(self._widget_to_image(event.pos()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._painting = False

    def _paint_at(self, pos):
        x, y = pos
        h, w = self._mask.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        r = max(1, self.brush_size // 2)
        value = 255 if self.is_drawing else 0
        cv2.circle(self._mask, (x, y), r, value, -1)
        self._render()


class LabelingWindow(QMainWindow):
    def __init__(self, session_dir: Path, parent=None):
        super().__init__(parent)
        self.session_dir = session_dir.resolve()
        ensure_session_structure(self.session_dir)
        self.source_dir = self.session_dir / "source"
        self.masks_dir = self.session_dir / "masks"
        self.cropped_dir = self.session_dir / "cropped"
        self.cropped_masks_dir = self.session_dir / "cropped_masks"

        self.current_stem = None
        self.current_path = None
        self.mode = "source"
        self.petri_info = None
        self.controller = LabelingController()
        self.paint_label = PaintLabel()

        self._init_ui()
        self._responsive_sizer = install_application_responsive_sizing(
            self, minimum_scale=0.75
        )
        self._load_file_list()

    def _init_ui(self):
        self.setWindowTitle(f"Разметка — {self.session_dir}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        self.root_layout = root
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._create_file_panel(root)
        self._create_image_area(root)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_initial_maximized_applied", False):
            self._initial_maximized_applied = True
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._responsive_sizer is not None:
            self._responsive_sizer.update_scale()

    def _create_file_panel(self, root: QHBoxLayout):
        panel = QFrame()
        self.file_panel = panel
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(320)
        panel.setStyleSheet(
            "QFrame { background-color: #181825; border-radius: 10px; }"
        )
        vl = QVBoxLayout(panel)
        self.file_panel_layout = vl
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(8)

        title = QLabel("📁 Тестовые изображения")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #89b4fa;")
        vl.addWidget(title)

        self.path_label = QLabel()
        self.path_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self.path_label.setWordWrap(True)
        vl.addWidget(self.path_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["📁 Исходные изображения", "✂️ Обрезки чашек"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        vl.addWidget(self.mode_combo)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet(
            "QListWidget { background-color: #1e1e2e; border-radius: 6px; }"
            "QListWidget::item:selected { background-color: #89b4fa; color: #1e1e2e; }"
        )
        self.file_list.itemClicked.connect(self._on_file_selected)
        vl.addWidget(self.file_list, stretch=1)

        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.setStyleSheet(
            "background-color: #45475a; font-size: 12px; padding: 6px;"
        )
        self.btn_refresh.clicked.connect(self._load_file_list)
        vl.addWidget(self.btn_refresh)

        self.btn_add_images = QPushButton("📂 Добавить изображения")
        self.btn_add_images.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "font-size: 12px; padding: 6px;"
        )
        self.btn_add_images.clicked.connect(self._on_add_images)
        vl.addWidget(self.btn_add_images)

        self._create_petri_panel(vl)

        root.addWidget(panel)

    def _create_petri_panel(self, parent: QVBoxLayout):
        self.petri_frame = QFrame()
        self.petri_frame.setStyleSheet(
            "QFrame { background-color: #1e1e2e; border-radius: 8px; }"
        )
        pl = QVBoxLayout(self.petri_frame)
        self.petri_layout = pl
        pl.setContentsMargins(8, 8, 8, 8)
        pl.setSpacing(6)

        pl.addWidget(QLabel("🔍 Чашка Петри"))
        pl.addWidget(QLabel("Центр X:"))
        self.spin_cx = QSpinBox()
        self.spin_cx.setRange(0, 10000)
        pl.addWidget(self.spin_cx)

        pl.addWidget(QLabel("Центр Y:"))
        self.spin_cy = QSpinBox()
        self.spin_cy.setRange(0, 10000)
        pl.addWidget(self.spin_cy)

        pl.addWidget(QLabel("Радиус:"))
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(10, 5000)
        pl.addWidget(self.spin_radius)

        self.spin_cx.valueChanged.connect(self._on_spinner_changed)
        self.spin_cy.valueChanged.connect(self._on_spinner_changed)
        self.spin_radius.valueChanged.connect(self._on_spinner_changed)

        self.btn_auto_detect = QPushButton("🔍 Авто-поиск")
        self.btn_auto_detect.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; "
            "padding: 6px; font-size: 12px;"
        )
        self.btn_auto_detect.clicked.connect(self._on_auto_detect)
        pl.addWidget(self.btn_auto_detect)

        self.btn_crop = QPushButton("✂️ Обрезать по чашке")
        self.btn_crop.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "padding: 6px; font-size: 12px;"
        )
        self.btn_crop.clicked.connect(self._on_crop)
        pl.addWidget(self.btn_crop)

        parent.addWidget(self.petri_frame)

    def _create_image_area(self, root: QHBoxLayout):
        wrapper = QWidget()
        wrapper.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        wrapper.setStyleSheet("background-color: #1e1e2e; border-radius: 10px;")
        vl = QVBoxLayout(wrapper)
        self.image_layout = vl
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(8)

        self.status_label = QLabel("Выберите изображение из списка слева")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        vl.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: #181825; width: 10px; }"
            "QScrollBar:horizontal { background: #181825; height: 10px; }"
            "QScrollBar::handle { background: #45475a; border-radius: 5px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
        )
        scroll.setWidget(self.paint_label)
        self.image_scroll_area = scroll
        self.paint_label.scroll_area = scroll
        vl.addWidget(scroll, stretch=1)

        self._create_toolbar(vl)

        root.addWidget(wrapper, stretch=1)

    def _create_toolbar(self, parent: QVBoxLayout):
        bar = QFrame()
        bar.setStyleSheet("QFrame { background-color: #181825; border-radius: 8px; }")
        hl = QVBoxLayout(bar)
        self.toolbar_layout = hl
        hl.setContentsMargins(8, 4, 8, 4)
        hl.setSpacing(4)
        brush_zoom_row = QHBoxLayout()
        brush_zoom_row.setSpacing(4)
        editing_row = QHBoxLayout()
        editing_row.setSpacing(4)
        persistence_row = QHBoxLayout()
        persistence_row.setSpacing(4)

        brush_zoom_row.addWidget(QLabel("Размер кисти:"))

        self.btn_brush_minus = QPushButton("−")
        self.btn_brush_minus.setFixedWidth(44)
        self.btn_brush_minus.setStyleSheet(
            "background-color: #45475a; padding: 2px; font-size: 11px;"
        )
        self.btn_brush_minus.clicked.connect(self._on_brush_decrease)
        brush_zoom_row.addWidget(self.btn_brush_minus)

        self.brush_size_label = QLabel("20 px")
        self.brush_size_label.setFixedWidth(44)
        self.brush_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brush_zoom_row.addWidget(self.brush_size_label)

        self.btn_brush_plus = QPushButton("+")
        self.btn_brush_plus.setFixedWidth(44)
        self.btn_brush_plus.setStyleSheet(
            "background-color: #45475a; padding: 2px; font-size: 11px;"
        )
        self.btn_brush_plus.clicked.connect(self._on_brush_increase)
        brush_zoom_row.addWidget(self.btn_brush_plus)
        brush_zoom_row.addStretch()

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self.zoom_label.setFixedWidth(56)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brush_zoom_row.addWidget(self.zoom_label)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(44)
        self.btn_zoom_in.setStyleSheet(
            "background-color: #45475a; padding: 2px; font-size: 11px;"
        )
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        brush_zoom_row.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedWidth(44)
        self.btn_zoom_out.setStyleSheet(
            "background-color: #45475a; padding: 2px; font-size: 11px;"
        )
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        brush_zoom_row.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("1:1")
        self.btn_zoom_reset.setMinimumWidth(56)
        self.btn_zoom_reset.setStyleSheet(
            "background-color: #45475a; padding: 2px 6px; font-size: 11px;"
        )
        self.btn_zoom_reset.clicked.connect(self._on_zoom_reset)
        brush_zoom_row.addWidget(self.btn_zoom_reset)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["🖼 Изображение + маска", "🏁 Только маска"])
        self.view_combo.setStyleSheet("font-size: 12px; padding: 4px 8px;")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        editing_row.addWidget(self.view_combo, stretch=1)

        self.btn_draw = QPushButton("✏️ Рисовать")
        self.btn_draw.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "padding: 6px 14px;"
        )
        self.btn_draw.clicked.connect(lambda: self._set_mode(True))
        editing_row.addWidget(self.btn_draw)

        self.btn_erase = QPushButton("🧹 Ластик")
        self.btn_erase.setStyleSheet("background-color: #45475a; padding: 6px 14px;")
        self.btn_erase.clicked.connect(lambda: self._set_mode(False))
        editing_row.addWidget(self.btn_erase)

        self.btn_clear = QPushButton("🗑 Очистить")
        self.btn_clear.setStyleSheet(
            "background-color: #f38ba8; color: #1e1e2e; padding: 6px 14px;"
        )
        self.btn_clear.clicked.connect(self._on_clear)
        persistence_row.addWidget(self.btn_clear)

        self.btn_save = QPushButton("💾 Сохранить маску")
        self.btn_save.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; "
            "padding: 6px 14px;"
        )
        self.btn_save.clicked.connect(self._on_save)
        persistence_row.addWidget(self.btn_save)

        self.btn_export = QPushButton("📦 Экспорт в ZIP")
        self.btn_export.setStyleSheet(
            "background-color: #f9e2af; color: #1e1e2e; font-weight: bold; "
            "padding: 6px 14px;"
        )
        self.btn_export.clicked.connect(self._on_export_zip)
        persistence_row.addWidget(self.btn_export)

        for widget in (
            self.view_combo,
            self.btn_draw,
            self.btn_erase,
            self.btn_clear,
            self.btn_save,
            self.btn_export,
        ):
            widget.setMinimumWidth(0)
        hl.addLayout(brush_zoom_row)
        hl.addLayout(editing_row)
        hl.addLayout(persistence_row)

        parent.addWidget(bar)

    def _update_zoom_label(self):
        self.zoom_label.setText(f"{self.paint_label.zoom_percent}%")

    def _on_zoom_in(self):
        self.paint_label.zoom_in()
        self._update_zoom_label()

    def _on_zoom_out(self):
        self.paint_label.zoom_out()
        self._update_zoom_label()

    def _on_zoom_reset(self):
        self.paint_label.zoom_reset()
        self._update_zoom_label()

    def _on_brush_decrease(self):
        new = max(2, self.paint_label.brush_size - 2)
        self.paint_label.brush_size = new
        self.brush_size_label.setText(f"{new} px")

    def _on_brush_increase(self):
        new = min(100, self.paint_label.brush_size + 2)
        self.paint_label.brush_size = new
        self.brush_size_label.setText(f"{new} px")

    def _on_view_changed(self, index: int):
        self.paint_label.view_mode = index
        self.paint_label._render()

    def _set_mode(self, drawing: bool):
        self.paint_label.is_drawing = drawing
        if drawing:
            set_responsive_stylesheet(
                self.btn_draw,
                "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
                "padding: 6px 14px;",
            )
            set_responsive_stylesheet(
                self.btn_erase, "background-color: #45475a; padding: 6px 14px;"
            )
        else:
            set_responsive_stylesheet(
                self.btn_draw,
                "background-color: #45475a; padding: 6px 14px;",
            )
            set_responsive_stylesheet(
                self.btn_erase,
                "background-color: #f38ba8; color: #1e1e2e; font-weight: bold; "
                "padding: 6px 14px;",
            )

    def _on_clear(self):
        if self.paint_label._mask is None:
            return
        answer = QMessageBox.question(
            self,
            "Очистить маску",
            "Вы уверены, что хотите очистить всю разметку?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.paint_label.clear_mask()

    def _get_current_dir(self):
        return self.controller.get_current_dir(self.session_dir, self.mode)

    def _get_mask_dir(self):
        return self.controller.get_mask_dir(self.session_dir, self.mode)

    def _update_path_label(self):
        working = self._get_current_dir()
        self.path_label.setText(f"📁 {working}")

    def _on_add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения для разметки",
            str(Path.home()),
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)",
        )
        if not files:
            return
        copied = 0
        dst_dir = self.session_dir / "source"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in map(Path, files):
            dst = dst_dir / src.name
            shutil.copy2(str(src), str(dst))
            copied += 1
        self.status_label.setText(f"📂 Скопировано изображений: {copied}")
        self._load_file_list()

    def _on_mode_changed(self, index: int):
        self.mode = "source" if index == 0 else "cropped"
        self.petri_frame.setVisible(self.mode == "source")
        self.paint_label.show_petri_circle = False
        self.petri_info = None
        self.paint_label.petri_info = None
        self.current_stem = None
        self.current_path = None
        self._clear_image()
        self._load_file_list()

    def _clear_image(self):
        self.paint_label._image = None
        self.paint_label._mask = None
        self.paint_label._original_pixmap = None
        self.paint_label.clear()
        self.status_label.setText("Выберите изображение из списка слева")

    def _load_file_list(self):
        self.file_list.clear()
        d = self._get_current_dir()
        for f in self.controller.list_image_files(d):
            item = QListWidgetItem(f.name)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            self.file_list.addItem(item)
        self._update_path_label()

    def _on_file_selected(self, item: QListWidgetItem):
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        try:
            image_rgb = self.controller.load_image_rgb(str(path))
        except ValueError:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить {path.name}")
            return

        self.current_stem = path.stem
        self.current_path = path
        self.petri_info = None
        self.paint_label.petri_info = None
        self.paint_label.show_petri_circle = False

        if self.mode == "source":
            self.spin_cx.setValue(0)
            self.spin_cy.setValue(0)
            self.spin_radius.setValue(0)

        mask_dir = self._get_mask_dir()
        mask_path = mask_dir / f"{self.current_stem}_mask.png"
        mask = self.controller.load_mask(mask_path, image_rgb.shape[:2])
        if mask is not None:
            self.status_label.setText(f"📷 {path.name} (маска загружена)")
        else:
            self.status_label.setText(f"📷 {path.name}")

        self.paint_label.set_image(image_rgb, mask)
        self._update_zoom_label()

        if self.mode == "source":
            self._on_auto_detect()

    def _on_spinner_changed(self):
        if self.mode != "source" or self.current_stem is None:
            return
        cx = self.spin_cx.value()
        cy = self.spin_cy.value()
        r = self.spin_radius.value()
        if r <= 0:
            return
        if self.paint_label._image is None:
            return
        h, w = self.paint_label._image.shape[:2]
        self.petri_info = PetriInfo(
            cx=cx,
            cy=cy,
            radius=r,
            image_shape=(h, w),
        )
        self.paint_label.petri_info = self.petri_info
        self.paint_label.show_petri_circle = True
        self.paint_label._render()

    def _on_auto_detect(self):
        if self.current_path is None:
            return
        try:
            image_bgr = load_image(str(self.current_path))
        except ValueError:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {self.current_path.name}"
            )
            return

        self.status_label.setText("Поиск чашки Петри...")
        info = self.controller.detect_petri(image_bgr)
        if info is None:
            self.status_label.setText("❌ Чашка не найдена. Настройте вручную.")
            return

        self.petri_info = info
        self.paint_label.petri_info = info
        self.paint_label.show_petri_circle = True

        self.spin_cx.blockSignals(True)
        self.spin_cy.blockSignals(True)
        self.spin_radius.blockSignals(True)
        self.spin_cx.setValue(info.cx)
        self.spin_cy.setValue(info.cy)
        self.spin_radius.setValue(info.radius)
        self.spin_cx.blockSignals(False)
        self.spin_cy.blockSignals(False)
        self.spin_radius.blockSignals(False)

        self.paint_label._render()
        self.status_label.setText(
            f"✅ Чашка найдена: центр ({info.cx}, {info.cy}), радиус {info.radius} px"
        )

    def _on_crop(self):
        if self.current_path is None:
            QMessageBox.information(self, "Обрезка", "Сначала выберите изображение.")
            return
        if self.petri_info is None or self.petri_info.radius <= 0:
            QMessageBox.warning(
                self, "Обрезка", "Сначала найдите чашку Петри (авто-поиск или вручную)."
            )
            return

        try:
            image_bgr = load_image(str(self.current_path))
        except ValueError:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {self.current_path.name}"
            )
            return

        cropped = self.controller.crop_by_petri(image_bgr, self.petri_info)

        out_name = f"{self.current_stem}_cropped.png"
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cropped_dir / out_name
        self.controller.save_mask(cropped, out_path)

        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.blockSignals(False)

        self.mode = "cropped"
        self.petri_frame.setVisible(False)
        self.paint_label.show_petri_circle = False
        self.petri_info = None
        self.paint_label.petri_info = None

        self._load_file_list()

        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text() == out_name:
                self.file_list.setCurrentItem(item)
                self._on_file_selected(item)
                break

        QMessageBox.information(
            self,
            "Готово",
            f"Обрезок сохранён:\n{out_name}\n\n"
            f"Теперь можно размечать маску на обрезанном изображении.",
        )

    def _on_export_zip(self):
        default_name = f"{self.session_dir.name}.zip"
        zip_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить ZIP архив",
            str(Path.home() / default_name),
            "ZIP архивы (*.zip)",
        )
        if not zip_path_str:
            return
        zip_path = Path(zip_path_str)
        self.controller.export_session_to_zip(self.session_dir, zip_path)
        QMessageBox.information(
            self, "Экспорт завершён", f"Архив сохранён:\n{zip_path}"
        )

    def _on_save(self):
        if self.current_stem is None or self.paint_label._mask is None:
            QMessageBox.information(
                self, "Сохранение", "Нет активного изображения для сохранения."
            )
            return
        mask_dir = self._get_mask_dir()
        mask_path = mask_dir / f"{self.current_stem}_mask.png"
        if mask_path.exists():
            answer = QMessageBox.question(
                self,
                "Подтверждение",
                f"Файл {mask_path.name} уже существует. Перезаписать?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.controller.save_mask(self.paint_label._mask, mask_path)
        QMessageBox.information(
            self, "Сохранено", f"Маска сохранена:\n{mask_path.name}"
        )
        self.status_label.setText(f"✅ Маска сохранена: {mask_path.name}")
