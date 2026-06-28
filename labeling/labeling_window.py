import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtWidgets import (
    QComboBox,
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

from analysis.colony_detector import ColonyDetector

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class PaintLabel(QLabel):
    ZOOM_MIN = 0.1
    ZOOM_MAX = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._image = None
        self._mask = None
        self.brush_size = 20
        self.is_drawing = True
        self._painting = False
        self._zoom_factor = 1.0
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

    def set_petri_info(self, info: Optional[Dict]):
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
                center = self.petri_info["center"]
                radius = self.petri_info["radius"]
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

        parent_widget = self.parentWidget()
        viewport_size = parent_widget.size() if parent_widget else self.size()

        fit_scale = min(
            viewport_size.width() / orig_w,
            viewport_size.height() / orig_h,
        )

        current_scale = fit_scale * self._zoom_factor

        new_w = max(1, int(orig_w * current_scale))
        new_h = max(1, int(orig_h * current_scale))

        scaled = self._original_pixmap.scaled(
            new_w, new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._scale = orig_w / new_w

        if self._zoom_factor != 1.0:
            self.setMinimumSize(new_w, new_h)
        else:
            self.setMinimumSize(0, 0)

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
        x = int(pos.x() * self._scale)
        y = int(pos.y() * self._scale)
        return x, y

    def mousePressEvent(self, event: QMouseEvent):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._mask is not None
        ):
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
    def __init__(self, parent=None):
        super().__init__(parent)
        base = Path(__file__).resolve().parent.parent
        self.source_dir = base / "test_images" / "source"
        self.masks_dir = base / "test_images" / "masks"
        self.cropped_dir = base / "test_images" / "cropped"
        self.cropped_masks_dir = base / "test_images" / "cropped_masks"

        for d in [self.source_dir, self.masks_dir, self.cropped_dir, self.cropped_masks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.current_stem = None
        self.current_path = None
        self.mode = "source"
        self.petri_info = None
        self.detector = ColonyDetector()
        self.paint_label = PaintLabel()

        self._init_ui()
        self._load_file_list()

    def _init_ui(self):
        self.setWindowTitle("Разметка тестовых изображений")
        self.resize(1300, 850)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._create_file_panel(root)
        self._create_image_area(root)

    def _create_file_panel(self, root: QHBoxLayout):
        panel = QFrame()
        panel.setFixedWidth(240)
        panel.setStyleSheet(
            "QFrame { background-color: #181825; border-radius: 10px; }"
        )
        l = QVBoxLayout(panel)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(8)

        title = QLabel("📁 Тестовые изображения")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #89b4fa;")
        l.addWidget(title)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["📁 Исходные изображения", "✂️ Обрезки чашек"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        l.addWidget(self.mode_combo)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet(
            "QListWidget { background-color: #1e1e2e; border-radius: 6px; }"
            "QListWidget::item:selected { background-color: #89b4fa; color: #1e1e2e; }"
        )
        self.file_list.itemClicked.connect(self._on_file_selected)
        l.addWidget(self.file_list, stretch=1)

        btn_refresh = QPushButton("🔄 Обновить")
        btn_refresh.setStyleSheet(
            "background-color: #45475a; font-size: 12px; padding: 6px;"
        )
        btn_refresh.clicked.connect(self._load_file_list)
        l.addWidget(btn_refresh)

        self._create_petri_panel(l)

        root.addWidget(panel)

    def _create_petri_panel(self, parent: QVBoxLayout):
        self.petri_frame = QFrame()
        self.petri_frame.setStyleSheet(
            "QFrame { background-color: #1e1e2e; border-radius: 8px; }"
        )
        pl = QVBoxLayout(self.petri_frame)
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

        btn_auto = QPushButton("🔍 Авто-поиск")
        btn_auto.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; "
            "padding: 6px; font-size: 12px;"
        )
        btn_auto.clicked.connect(self._on_auto_detect)
        pl.addWidget(btn_auto)

        btn_crop = QPushButton("✂️ Обрезать по чашке")
        btn_crop.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "padding: 6px; font-size: 12px;"
        )
        btn_crop.clicked.connect(self._on_crop)
        pl.addWidget(btn_crop)

        parent.addWidget(self.petri_frame)

    def _create_image_area(self, root: QHBoxLayout):
        wrapper = QWidget()
        wrapper.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        wrapper.setStyleSheet("background-color: #1e1e2e; border-radius: 10px;")
        vl = QVBoxLayout(wrapper)
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
        vl.addWidget(scroll, stretch=1)

        self._create_toolbar(vl)

        root.addWidget(wrapper, stretch=1)

    def _create_toolbar(self, parent: QVBoxLayout):
        bar = QFrame()
        bar.setStyleSheet(
            "QFrame { background-color: #181825; border-radius: 8px; }"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(12, 8, 12, 8)
        hl.setSpacing(10)

        hl.addWidget(QLabel("Размер кисти:"))

        btn_brush_minus = QPushButton("−")
        btn_brush_minus.setFixedWidth(30)
        btn_brush_minus.setStyleSheet("background-color: #45475a; padding: 4px; font-size: 14px;")
        btn_brush_minus.clicked.connect(self._on_brush_decrease)
        hl.addWidget(btn_brush_minus)

        self.brush_size_label = QLabel("20 px")
        self.brush_size_label.setFixedWidth(40)
        self.brush_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(self.brush_size_label)

        btn_brush_plus = QPushButton("+")
        btn_brush_plus.setFixedWidth(30)
        btn_brush_plus.setStyleSheet("background-color: #45475a; padding: 4px; font-size: 14px;")
        btn_brush_plus.clicked.connect(self._on_brush_increase)
        hl.addWidget(btn_brush_plus)

        hl.addStretch()

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        self.zoom_label.setFixedWidth(50)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(self.zoom_label)

        btn_zoom_in = QPushButton("🔍+")
        btn_zoom_in.setStyleSheet(
            "background-color: #45475a; padding: 4px 10px; font-size: 12px;"
        )
        btn_zoom_in.clicked.connect(self._on_zoom_in)
        hl.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("🔍−")
        btn_zoom_out.setStyleSheet(
            "background-color: #45475a; padding: 4px 10px; font-size: 12px;"
        )
        btn_zoom_out.clicked.connect(self._on_zoom_out)
        hl.addWidget(btn_zoom_out)

        btn_zoom_reset = QPushButton("⟲ 1:1")
        btn_zoom_reset.setStyleSheet(
            "background-color: #45475a; padding: 4px 10px; font-size: 12px;"
        )
        btn_zoom_reset.clicked.connect(self._on_zoom_reset)
        hl.addWidget(btn_zoom_reset)

        hl.addStretch()

        self.view_combo = QComboBox()
        self.view_combo.addItems(["🖼 Изображение + маска", "🏁 Только маска"])
        self.view_combo.setStyleSheet("font-size: 12px; padding: 4px 8px;")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        hl.addWidget(self.view_combo)

        self.btn_draw = QPushButton("✏️ Рисовать")
        self.btn_draw.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "padding: 6px 14px;"
        )
        self.btn_draw.clicked.connect(lambda: self._set_mode(True))
        hl.addWidget(self.btn_draw)

        self.btn_erase = QPushButton("🧹 Ластик")
        self.btn_erase.setStyleSheet(
            "background-color: #45475a; padding: 6px 14px;"
        )
        self.btn_erase.clicked.connect(lambda: self._set_mode(False))
        hl.addWidget(self.btn_erase)

        btn_clear = QPushButton("🗑 Очистить")
        btn_clear.setStyleSheet(
            "background-color: #f38ba8; color: #1e1e2e; padding: 6px 14px;"
        )
        btn_clear.clicked.connect(self._on_clear)
        hl.addWidget(btn_clear)

        btn_save = QPushButton("💾 Сохранить маску")
        btn_save.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; "
            "padding: 6px 14px;"
        )
        btn_save.clicked.connect(self._on_save)
        hl.addWidget(btn_save)

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
            self.btn_draw.setStyleSheet(
                "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; "
                "padding: 6px 14px;"
            )
            self.btn_erase.setStyleSheet(
                "background-color: #45475a; padding: 6px 14px;"
            )
        else:
            self.btn_draw.setStyleSheet(
                "background-color: #45475a; padding: 6px 14px;"
            )
            self.btn_erase.setStyleSheet(
                "background-color: #f38ba8; color: #1e1e2e; font-weight: bold; "
                "padding: 6px 14px;"
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
        return self.source_dir if self.mode == "source" else self.cropped_dir

    def _get_mask_dir(self):
        return self.masks_dir if self.mode == "source" else self.cropped_masks_dir

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
        self.status_label.setText(
            "Выберите изображение из списка слева"
        )

    def _load_file_list(self):
        self.file_list.clear()
        d = self._get_current_dir()
        if not d.exists():
            return
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                item = QListWidgetItem(f.name)
                item.setData(Qt.ItemDataRole.UserRole, str(f))
                self.file_list.addItem(item)

    def _on_file_selected(self, item: QListWidgetItem):
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {path.name}"
            )
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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
        mask = None
        if mask_path.exists():
            loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if loaded is not None and loaded.shape == image_rgb.shape[:2]:
                mask = loaded
                self.status_label.setText(f"📷 {path.name} (маска загружена)")
            else:
                self.status_label.setText(f"📷 {path.name}")
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
        self.petri_info = {
            "center": (cx, cy),
            "radius": r,
        }
        self.paint_label.petri_info = self.petri_info
        self.paint_label.show_petri_circle = True
        self.paint_label._render()

    def _on_auto_detect(self):
        if self.current_path is None:
            return
        image_bgr = cv2.imread(str(self.current_path))
        if image_bgr is None:
            return

        self.status_label.setText("Поиск чашки Петри...")
        _, info = self.detector.detect_petri_dish(image_bgr)
        if info is None:
            self.status_label.setText("❌ Чашка не найдена. Настройте вручную.")
            return

        self.petri_info = info
        self.paint_label.petri_info = info
        self.paint_label.show_petri_circle = True

        self.spin_cx.blockSignals(True)
        self.spin_cy.blockSignals(True)
        self.spin_radius.blockSignals(True)
        self.spin_cx.setValue(info["center"][0])
        self.spin_cy.setValue(info["center"][1])
        self.spin_radius.setValue(info["radius"])
        self.spin_cx.blockSignals(False)
        self.spin_cy.blockSignals(False)
        self.spin_radius.blockSignals(False)

        self.paint_label._render()
        self.status_label.setText(
            f"✅ Чашка найдена: центр ({info['center'][0]}, {info['center'][1]}), "
            f"радиус {info['radius']} px"
        )

    def _on_crop(self):
        if self.current_path is None:
            QMessageBox.information(
                self, "Обрезка", "Сначала выберите изображение."
            )
            return
        if self.petri_info is None or self.petri_info["radius"] <= 0:
            QMessageBox.warning(
                self, "Обрезка",
                "Сначала найдите чашку Петри (авто-поиск или вручную)."
            )
            return

        cx, cy = self.petri_info["center"]
        r = self.petri_info["radius"]

        image_bgr = cv2.imread(str(self.current_path))
        if image_bgr is None:
            return

        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(image_bgr.shape[1], cx + r)
        y2 = min(image_bgr.shape[0], cy + r)
        cropped = image_bgr[y1:y2, x1:x2]

        circle_center = (cx - x1, cy - y1)
        circle_mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, circle_center, r, 255, -1)
        cropped[circle_mask == 0] = [0, 0, 0]

        out_name = f"{self.current_stem}_cropped.png"
        out_path = self.cropped_dir / out_name
        cv2.imwrite(str(out_path), cropped)

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
            self, "Готово",
            f"Обрезок сохранён:\n{out_name}\n\n"
            f"Теперь можно размечать маску на обрезанном изображении."
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
        cv2.imwrite(str(mask_path), self.paint_label._mask)
        QMessageBox.information(
            self, "Сохранено", f"Маска сохранена:\n{mask_path.name}"
        )
        self.status_label.setText(f"✅ Маска сохранена: {mask_path.name}")
