import logging
import math
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen

from analysis.geometry import PetriInfo
from labeling.session_manager import ensure_session_structure
from ui.background import BackgroundOperation
from ui.controllers.labeling_controller import LabelingController
from ui.responsive import (
    install_application_responsive_sizing,
    set_responsive_stylesheet,
)
from utils.image_loader import load_image, load_image_worker_safe

log = logging.getLogger(__name__)
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
        self._scale = 1.0
        self._scaled_size = QSize(0, 0)
        # Кеш композита для уменьшенных представлений: {масштаб уровня: ndarray}.
        # Хранит только отображение (1/2 и 1/8 кадра), патчится локально кистью.
        self._level_cache: dict[float, np.ndarray] = {}
        self._view_mode = 0
        self.petri_info = None
        self.show_petri_circle = False

    @property
    def view_mode(self) -> int:
        return self._view_mode

    @view_mode.setter
    def view_mode(self, value: int) -> None:
        value = int(value)
        if value != self._view_mode:
            self._view_mode = value
            self._level_cache.clear()
            self.update()

    def set_image(self, image: np.ndarray, mask: np.ndarray = None):
        self._image = image.copy()
        h, w = image.shape[:2]
        if mask is not None and mask.shape[:2] == (h, w):
            self._mask = mask.copy()
        else:
            self._mask = np.zeros((h, w), dtype=np.uint8)
        self._zoom_factor = 1.0
        self._level_cache.clear()
        self._render()

    def clear_image(self):
        """Полный сброс канваса (изображение, маска, кеш отображения)."""
        self._image = None
        self._mask = None
        self._level_cache.clear()
        self._scaled_size = QSize(0, 0)
        self.clear()
        self.update()

    def set_petri_info(self, info: Optional[PetriInfo]):
        self.petri_info = info
        self._render()

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def clear_mask(self):
        if self._mask is not None:
            self._mask.fill(0)
            self._level_cache.clear()
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

    # ------------------------------------------------------------------
    # Рендеринг: только видимая/повреждённая область (viewport-bounded).
    # Полноразмерные `_image`/`_mask` — источник истины; события кисти
    # обновляют локальный dirty rect, а не пересобирают весь overlay.
    # ------------------------------------------------------------------

    # Уровни кеша отображения (масштабы кадра). Для zoom-out используется
    # ближайший уровень, не превышающий масштаб отображения, поэтому
    # масштабирование всегда идёт вниз и остаётся чётким.
    _LEVEL_SCALES = (0.5, 0.125)
    # Начиная с какого масштаба отображения композит строится напрямую
    # из полного разрешения (область ограничена viewport).
    _FULL_RES_THRESHOLD = 0.5

    def _render(self):
        """Перерисовать канвас (геометрия + repaint видимой области)."""
        self._update_scaled()

    def _update_scaled(self):
        if self._image is None:
            return

        orig_h, orig_w = self._image.shape[:2]

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

        self._scaled_size = QSize(new_w, new_h)
        self._scale = orig_w / new_w

        # Увеличение картинки должно прокручиваться внутри viewport, не
        # передавая её размер в sizeHint родительского окна.
        self.setMinimumSize(self._base_zoom_minimum)

        self._offset_x = (self.width() - new_w) / 2.0
        self._offset_y = (self.height() - new_h) / 2.0

        self.updateGeometry()
        self.update()

    def sizeHint(self):
        if self._image is not None and not self._scaled_size.isEmpty():
            return QSize(self._scaled_size)
        return super().sizeHint()

    def minimumSizeHint(self):
        # QScrollArea растит виджет по minimumSizeHint (как делал QLabel
        # с pixmap) — так появляются полосы прокрутки при зуме.
        if self._image is not None and not self._scaled_size.isEmpty():
            return QSize(self._scaled_size)
        return super().minimumSizeHint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def wheelEvent(self, event):
        if self._image is None:
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

    def _image_to_widget_rect(self, x0: int, y0: int, x1: int, y1: int):
        """Прямоугольник изображения -> widget-прямоугольник (с запасом 1px)."""
        s = self._scale
        left = int(math.floor(x0 / s + self._offset_x)) - 1
        top = int(math.floor(y0 / s + self._offset_y)) - 1
        right = int(math.ceil(x1 / s + self._offset_x)) + 1
        bottom = int(math.ceil(y1 / s + self._offset_y)) + 1
        return QRect(QPoint(left, top), QPoint(right, bottom))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._image is None or self._mask is None:
            return
        painter = QPainter(self)
        try:
            self._paint_widget_rect(painter, event.rect())
        finally:
            painter.end()

    def _paint_widget_rect(self, painter: QPainter, rect):
        """Рисует композит только для заданного widget-прямоугольника."""
        h, w = self._image.shape[:2]
        rect = rect.intersected(self.rect())
        if rect.isEmpty():
            return
        s = self._scale  # пикселей изображения на экранный пиксель

        ix0 = max(0, int(math.floor((rect.left() - self._offset_x) * s)) - 1)
        iy0 = max(0, int(math.floor((rect.top() - self._offset_y) * s)) - 1)
        ix1 = min(w, int(math.ceil((rect.right() + 1 - self._offset_x) * s)) + 1)
        iy1 = min(h, int(math.ceil((rect.bottom() + 1 - self._offset_y) * s)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return

        display_scale = 1.0 / s  # экранных пикселей на пиксель изображения
        source, ps, jx0, jy0, jx1, jy1 = self._source_region(
            ix0, iy0, ix1, iy1, display_scale
        )
        # QImage требует смежный буфер: срезы кеша уровней копируются.
        source = np.ascontiguousarray(source)

        qimage = QImage(
            source.data,
            source.shape[1],
            source.shape[0],
            3 * source.shape[1],
            QImage.Format.Format_RGB888,
        )
        target = QRectF(
            jx0 / s + self._offset_x,
            jy0 / s + self._offset_y,
            (jx1 - jx0) / s,
            (jy1 - jy0) / s,
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, qimage)

        # Окружность чашки поверх изображения (как раньше, но в экранных
        # координатах: толщина 2px исходного кадра, пересчитанная по зуму).
        if (
            self.view_mode == 0
            and self.show_petri_circle
            and self.petri_info is not None
        ):
            info = self.petri_info
            image_rect = QRectF(
                self._offset_x,
                self._offset_y,
                w / s,
                h / s,
            )
            painter.save()
            painter.setClipRect(image_rect)
            painter.setPen(QPen(QColor(100, 100, 255), max(1.0, 2.0 / s)))
            painter.drawEllipse(
                QPointF(info.cx / s + self._offset_x, info.cy / s + self._offset_y),
                info.radius / s,
                info.radius / s,
            )
            painter.restore()

    def _select_level(self, display_scale: float) -> float:
        """Выбирает масштаб источника: уровень кеша >= масштаба отображения."""
        if display_scale >= self._FULL_RES_THRESHOLD:
            return 1.0
        best = self._LEVEL_SCALES[-1]
        for level in sorted(self._LEVEL_SCALES, reverse=True):
            if level >= display_scale:
                best = level
                break
        return best

    def _source_region(self, ix0, iy0, ix1, iy1, display_scale):
        """Композит области: (массив BGR, масштаб, границы в пикселях кадра)."""
        ps = self._select_level(display_scale)
        if ps >= 1.0:
            return self._composite_region(ix0, iy0, ix1, iy1), 1.0, ix0, iy0, ix1, iy1

        level = self._get_level(ps)
        lh, lw = level.shape[:2]
        lx0 = max(0, int(math.floor(ix0 * ps)))
        ly0 = max(0, int(math.floor(iy0 * ps)))
        lx1 = min(lw, max(lx0 + 1, int(math.ceil(ix1 * ps))))
        ly1 = min(lh, max(ly0 + 1, int(math.ceil(iy1 * ps))))
        return (
            level[ly0:ly1, lx0:lx1],
            ps,
            lx0 / ps,
            ly0 / ps,
            lx1 / ps,
            ly1 / ps,
        )

    def _composite_region(self, x0, y0, x1, y1) -> np.ndarray:
        """Композит области полного разрешения (формула как у старого _render)."""
        if self.view_mode == 1:
            return cv2.cvtColor(self._mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        region = self._image[y0:y1, x0:x1]
        green = np.zeros_like(region)
        green[:, :, 1] = self._mask[y0:y1, x0:x1]
        return cv2.addWeighted(region, 1.0, green, 0.35, 0)

    def _blend_level(self, image_s: np.ndarray, mask_s: np.ndarray) -> np.ndarray:
        """Композит уменьшенного представления (маска может быть мягкой)."""
        if self.view_mode == 1:
            return cv2.cvtColor(mask_s, cv2.COLOR_GRAY2BGR)
        green = np.zeros_like(image_s)
        green[:, :, 1] = mask_s
        return cv2.addWeighted(image_s, 1.0, green, 0.35, 0)

    def _build_level(self, ps: float) -> np.ndarray:
        """Строит кеш композита уровня (только при инвалидации/первом показе)."""
        h, w = self._image.shape[:2]
        lw = max(1, int(round(w * ps)))
        lh = max(1, int(round(h * ps)))
        image_s = cv2.resize(self._image, (lw, lh), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(self._mask, (lw, lh), interpolation=cv2.INTER_AREA)
        return self._blend_level(image_s, mask_s)

    def _get_level(self, ps: float) -> np.ndarray:
        level = self._level_cache.get(ps)
        if level is None:
            level = self._build_level(ps)
            self._level_cache[ps] = level
        return level

    def _patch_levels(self, x0, y0, x1, y1) -> None:
        """Локально обновляет кеш уровней после изменения маски в bbox."""
        for ps, level in self._level_cache.items():
            lh, lw = level.shape[:2]
            lx0 = max(0, int(math.floor(x0 * ps)))
            ly0 = max(0, int(math.floor(y0 * ps)))
            lx1 = min(lw, max(lx0 + 1, int(math.ceil(x1 * ps))))
            ly1 = min(lh, max(ly0 + 1, int(math.ceil(y1 * ps))))
            sub_img = cv2.resize(
                self._image[y0:y1, x0:x1],
                (lx1 - lx0, ly1 - ly0),
                interpolation=cv2.INTER_AREA,
            )
            sub_mask = cv2.resize(
                self._mask[y0:y1, x0:x1],
                (lx1 - lx0, ly1 - ly0),
                interpolation=cv2.INTER_AREA,
            )
            level[ly0:ly1, lx0:lx1] = self._blend_level(sub_img, sub_mask)

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
        # Локальное обновление отображения: патч кешей + dirty rect.
        x0, y0 = max(0, x - r - 1), max(0, y - r - 1)
        x1, y1 = min(w, x + r + 2), min(h, y + r + 2)
        self._patch_levels(x0, y0, x1, y1)
        self.update(self._image_to_widget_rect(x0, y0, x1, y1))


class LabelingWindow(QMainWindow):
    _IDLE_STATUS = "Выберите изображение из списка слева"
    _LIST_BUSY_STATUS = "⏳ Обновление списка файлов…"

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
        # Поколение выбора: устаревшие результаты фоновых загрузок
        # не должны перезаписывать более новый выбор пользователя.
        self._selection_generation = 0
        self._load_operation: Optional[BackgroundOperation] = None
        # Поколение обновления списка файлов (задача 6.2).
        self._list_generation = 0
        self._list_operation: Optional[BackgroundOperation] = None
        # Фоновые операции поиска чашки и обрезки (задача 6.3).
        self._detect_operation: Optional[BackgroundOperation] = None
        self._crop_operation: Optional[BackgroundOperation] = None
        # Пакетное копирование исходных файлов (задача 6.4).
        self._copy_operation: Optional[BackgroundOperation] = None
        # Сохранение маски и атомарный ZIP-экспорт (задача 6.6).
        self._save_operation: Optional[BackgroundOperation] = None
        self._export_operation: Optional[BackgroundOperation] = None
        # Deferred close (задача 6.7): окно закрывается после safe-точки.
        self._closing = False
        self._pending_closes = 0
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
        self.btn_refresh.clicked.connect(lambda: self._load_file_list())
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
        if self._closing:
            return  # закрытие: новая работа не начинается
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения для разметки",
            str(Path.home()),
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)",
        )
        if not files:
            return
        dst_dir = self.session_dir / "source"
        dst_dir.mkdir(parents=True, exist_ok=True)

        if self._copy_operation is not None and self._copy_operation.is_running():
            return  # предыдущее копирование ещё выполняется

        self.status_label.setText(f"⏳ Копирование изображений: 0 / {len(files)}")
        generation = self._selection_generation
        operation = BackgroundOperation(
            self._copy_work, generation=generation, parent=self
        )
        operation.progress_changed.connect(self._on_copy_progress)
        operation.result_ready.connect(self._on_copy_finished)
        operation.error_raised.connect(self._on_copy_failed)
        operation.cancellation_confirmed.connect(self._on_copy_cancelled)
        self._copy_operation = operation
        operation.start(files=list(files), dst_dir=str(dst_dir))

    def _copy_work(self, ctx, files, dst_dir: str):
        """Worker: пакетное копирование исходных файлов (shutil.copy2)."""
        total = len(files)
        copied = 0
        for index, src in enumerate(map(Path, files), start=1):
            ctx.checkpoint()  # отмена между файлами
            ctx.status(f"копирование {src.name}")
            dst = Path(dst_dir) / src.name
            # Семантика перезаписи при совпадении имён сохраняется (copy2).
            shutil.copy2(str(src), str(dst))
            copied += 1
            ctx.progress(index, total)
        return {"copied": copied, "total": total}

    def _on_copy_progress(self, completed: int, total: int):
        if total > 0:
            self.status_label.setText(
                f"⏳ Копирование изображений: {completed} / {total}"
            )

    def _on_copy_finished(self, result, generation: int):
        copied = result["copied"]
        if generation == self._selection_generation:
            self.status_label.setText(f"📂 Скопировано изображений: {copied}")
        self._load_file_list()

    def _on_copy_failed(self, message: str, generation: int):
        if generation == self._selection_generation:
            self.status_label.setText("❌ Ошибка копирования")
        QMessageBox.warning(
            self,
            "Ошибка копирования",
            f"Копирование прервано:\n{message}\n\n"
            "Уже скопированные файлы сохранены.",
        )
        # Частично скопированные файлы должны быть видны в списке.
        self._load_file_list()

    def _on_copy_cancelled(self, generation: int):
        if generation == self._selection_generation:
            self.status_label.setText("⏹ Копирование отменено")
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

    def _operations(self) -> list:
        return [
            self._load_operation,
            self._list_operation,
            self._detect_operation,
            self._crop_operation,
            self._copy_operation,
            self._save_operation,
            self._export_operation,
        ]

    def closeEvent(self, event):
        """Deferred close: окно закрывается после safe-точки операций (6.7)."""
        running = [
            op for op in self._operations() if op is not None and op.is_running()
        ]
        if not running:
            super().closeEvent(event)
            return
        event.ignore()
        if self._closing:
            return  # уже ждём безопасной точки
        self._closing = True
        self._pending_closes = len(running)
        self.status_label.setText(
            "⏳ Завершение… Окно закроется после завершения операций."
        )
        for operation in running:
            # Кооперативная отмена между safe-точками; непрерываемый
            # native-вызов продолжит работу и не блокирует event loop.
            operation.cancel()
            operation.close_reached.connect(self._on_operation_close_reached)
            operation.request_close()

    def _on_operation_close_reached(self):
        self._pending_closes -= 1
        if self._pending_closes <= 0:
            self._closing = False
            self.close()

    def _clear_image(self):
        # Смена файла/режима: устаревшие результаты фоновых операций
        # должны отбрасываться по номеру поколения.
        self._selection_generation += 1
        self.paint_label.clear_image()
        self.status_label.setText(self._IDLE_STATUS)

    def _load_file_list(self, select_name: str | None = None):
        """Обновляет список файлов сессии вне GUI-потока (задача 6.2).

        `select_name` — имя файла для автовыбора после завершения обновления
        (используется обрезкой для перехода к созданному файлу).
        """
        if self._closing:
            return  # закрытие: новая работа не начинается
        self._list_generation += 1
        generation = self._list_generation
        directory = self._get_current_dir()
        self._update_path_label()

        if self._list_operation is not None and self._list_operation.is_running():
            self._list_operation.cancel()
        if self.current_stem is None and self.status_label.text() in (
            self._IDLE_STATUS,
            self._LIST_BUSY_STATUS,
        ):
            self.status_label.setText(self._LIST_BUSY_STATUS)

        operation = BackgroundOperation(
            self._list_files_work, generation=generation, parent=self
        )
        operation.result_ready.connect(
            lambda result, gen: self._on_file_list_loaded(result, gen, select_name)
        )
        operation.error_raised.connect(self._on_file_list_failed)
        self._list_operation = operation
        operation.start(directory=str(directory))

    def _list_files_work(self, ctx, directory: str):
        """Worker: перечисление и сортировка файлов сессии."""
        ctx.status("обновление списка файлов")
        files = self.controller.list_image_files(Path(directory))
        return [str(f) for f in files]

    def _on_file_list_loaded(self, result, generation: int, select_name=None):
        if generation != self._list_generation:
            return  # устаревшее обновление списка
        self.file_list.clear()
        selected_item = None
        for name in result:
            item = QListWidgetItem(Path(name).name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.file_list.addItem(item)
            if select_name is not None and item.text() == select_name:
                selected_item = item
        if self.current_stem is None and self.status_label.text() == (
            self._LIST_BUSY_STATUS
        ):
            self.status_label.setText(self._IDLE_STATUS)
        if selected_item is not None:
            self.file_list.setCurrentItem(selected_item)
            self._on_file_selected(selected_item)

    def _on_file_list_failed(self, message: str, generation: int):
        if generation != self._list_generation:
            return
        self.status_label.setText("❌ Не удалось обновить список файлов")
        log.warning("Не удалось обновить список файлов: %s", message)

    def _on_file_selected(self, item: QListWidgetItem):
        if self._closing:
            return  # закрытие: новая работа не начинается
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        self._selection_generation += 1
        generation = self._selection_generation

        if self._load_operation is not None and self._load_operation.is_running():
            self._load_operation.cancel()

        mask_path = self._get_mask_dir() / f"{path.stem}_mask.png"
        self.status_label.setText(f"⏳ Загрузка {path.name}…")

        operation = BackgroundOperation(
            self._load_image_work, generation=generation, parent=self
        )
        operation.result_ready.connect(self._on_image_loaded)
        operation.error_raised.connect(self._on_image_load_failed)
        self._load_operation = operation
        operation.start(path=str(path), mask_path=str(mask_path), stem=path.stem)

    def _load_image_work(self, ctx, path: str, mask_path: str, stem: str):
        """Worker: декодирование изображения и маски вне GUI-потока."""
        ctx.status("декодирование изображения")
        image_bgr = load_image_worker_safe(path)
        ctx.checkpoint()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ctx.status("декодирование маски")
        mask = self.controller.load_mask(Path(mask_path), image_rgb.shape[:2])
        return {"image": image_rgb, "mask": mask, "path": path, "stem": stem}

    def _on_image_loaded(self, result, generation: int):
        if generation != self._selection_generation:
            return  # устаревший результат: выбор уже заменён

        image_rgb = result["image"]
        mask = result["mask"]
        path = Path(result["path"])

        self.current_stem = result["stem"]
        self.current_path = path
        self.petri_info = None
        self.paint_label.petri_info = None
        self.paint_label.show_petri_circle = False

        if self.mode == "source":
            self.spin_cx.setValue(0)
            self.spin_cy.setValue(0)
            self.spin_radius.setValue(0)

        if mask is not None:
            self.status_label.setText(f"📷 {path.name} (маска загружена)")
        else:
            self.status_label.setText(f"📷 {path.name}")

        self.paint_label.set_image(image_rgb, mask)
        self._update_zoom_label()

        if self.mode == "source":
            self._on_auto_detect()

    def _on_image_load_failed(self, message: str, generation: int):
        if generation != self._selection_generation:
            return
        self.status_label.setText("❌ Ошибка загрузки")
        QMessageBox.warning(
            self,
            "Ошибка",
            f"Не удалось загрузить изображение:\n{message}",
        )

    def _current_image_bgr(self) -> Optional[np.ndarray]:
        """Декодированный оригинал текущего изображения в BGR.

        Переиспользует уже декодированный кадр канваса (RGB) вместо
        повторного чтения исходного файла; fallback — чтение файла.
        """
        if self.paint_label._image is not None:
            return cv2.cvtColor(self.paint_label._image, cv2.COLOR_RGB2BGR)
        if self.current_path is not None:
            return load_image(str(self.current_path))
        return None

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
        if self._closing:
            return  # закрытие: новая работа не начинается
        if self.current_path is None:
            return
        try:
            image_bgr = self._current_image_bgr()
        except ValueError:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {self.current_path.name}"
            )
            return
        if image_bgr is None:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {self.current_path.name}"
            )
            return

        if self._detect_operation is not None and self._detect_operation.is_running():
            return  # предыдущий поиск ещё выполняется

        self.status_label.setText("🔍 Поиск чашки Петри…")
        generation = self._selection_generation
        operation = BackgroundOperation(
            self._detect_work, generation=generation, parent=self
        )
        operation.result_ready.connect(self._on_detect_finished)
        operation.error_raised.connect(self._on_detect_failed)
        self._detect_operation = operation
        operation.start(image_bgr=image_bgr)

    def _detect_work(self, ctx, image_bgr):
        """Worker: автоматический поиск чашки Петри (вне GUI-потока)."""
        ctx.status("поиск чашки Петри")
        return self.controller.detect_petri(image_bgr)

    def _on_detect_finished(self, info, generation: int):
        if generation != self._selection_generation:
            return  # устаревший результат: выбор уже заменён
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

    def _on_detect_failed(self, message: str, generation: int):
        if generation != self._selection_generation:
            return
        self.status_label.setText("❌ Ошибка поиска чашки")
        QMessageBox.warning(
            self, "Ошибка", f"Не удалось выполнить поиск чашки:\n{message}"
        )

    def _on_crop(self):
        if self._closing:
            return  # закрытие: новая работа не начинается
        if self.current_path is None:
            QMessageBox.information(self, "Обрезка", "Сначала выберите изображение.")
            return
        if self.petri_info is None or self.petri_info.radius <= 0:
            QMessageBox.warning(
                self, "Обрезка", "Сначала найдите чашку Петри (авто-поиск или вручную)."
            )
            return

        try:
            image_bgr = self._current_image_bgr()
        except ValueError:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {self.current_path.name}"
            )
            return
        if image_bgr is None:
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить {self.current_path.name}"
            )
            return

        if self._crop_operation is not None and self._crop_operation.is_running():
            return  # предыдущая обрезка ещё выполняется

        out_name = f"{self.current_stem}_cropped.png"
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cropped_dir / out_name

        self.status_label.setText("✂️ Обрезка по чашке…")
        generation = self._selection_generation
        operation = BackgroundOperation(
            self._crop_work, generation=generation, parent=self
        )
        operation.result_ready.connect(self._on_crop_finished)
        operation.error_raised.connect(self._on_crop_failed)
        self._crop_operation = operation
        operation.start(
            image_bgr=image_bgr,
            petri_info=self.petri_info,
            out_path=str(out_path),
            out_name=out_name,
        )

    def _crop_work(self, ctx, image_bgr, petri_info, out_path: str, out_name: str):
        """Worker: обрезка по чашке и запись файла (вне GUI-потока)."""
        ctx.status("обрезка изображения")
        cropped = self.controller.crop_by_petri(image_bgr, petri_info)
        ctx.checkpoint()  # файл не пишется после отмены
        ctx.status("запись обрезка")
        self.controller.save_mask(cropped, Path(out_path))
        return {"out_name": out_name}

    def _on_crop_finished(self, result, generation: int):
        out_name = result["out_name"]
        if generation != self._selection_generation:
            # Выбор уже заменён: файл создан, но навигацию не навязываем.
            self._load_file_list()
            return

        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.blockSignals(False)

        self.mode = "cropped"
        self.petri_frame.setVisible(False)
        self.paint_label.show_petri_circle = False
        self.petri_info = None
        self.paint_label.petri_info = None

        self._load_file_list(select_name=out_name)

        QMessageBox.information(
            self,
            "Готово",
            f"Обрезок сохранён:\n{out_name}\n\n"
            f"Теперь можно размечать маску на обрезанном изображении.",
        )

    def _on_crop_failed(self, message: str, generation: int):
        self.status_label.setText("❌ Ошибка обрезки")
        QMessageBox.warning(self, "Ошибка", f"Не удалось выполнить обрезку:\n{message}")

    def _on_export_zip(self):
        if self._closing:
            return  # закрытие: новая работа не начинается
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
        if self._export_operation is not None and self._export_operation.is_running():
            return  # предыдущий экспорт ещё выполняется

        self.status_label.setText("📦 Экспорт ZIP…")
        operation = BackgroundOperation(
            self._export_zip_work,
            generation=self._selection_generation,
            parent=self,
        )
        operation.progress_changed.connect(self._on_export_progress)
        operation.result_ready.connect(self._on_export_finished)
        operation.error_raised.connect(self._on_export_failed)
        operation.cancellation_confirmed.connect(self._on_export_cancelled)
        self._export_operation = operation
        operation.start(
            session_dir=str(self.session_dir), final_path=str(zip_path)
        )

    def _export_zip_work(self, ctx, session_dir: str, final_path: str):
        """Worker: ZIP через временный файл, публикация только после успеха.

        Отменённый или ошибочный экспорт не трогает ранее существовавший
        файл по целевому пути и не оставляет частичный архив под именем.
        """
        final = Path(final_path)
        tmp = final.parent / f".{final.name}.partial"
        try:
            self.controller.export_session_to_zip(
                Path(session_dir),
                tmp,
                checkpoint=ctx.checkpoint,
                progress=lambda done: ctx.progress(done),
            )
            ctx.checkpoint()  # отмена до публикации не изменяет целевой файл
            tmp.replace(final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return {"final_path": str(final)}

    def _on_export_progress(self, completed: int, total: int):
        self.status_label.setText(f"📦 Экспорт ZIP: файлов {completed}")

    def _on_export_finished(self, result, generation: int):
        final = result["final_path"]
        QMessageBox.information(
            self, "Экспорт завершён", f"Архив сохранён:\n{final}"
        )
        self.status_label.setText("✅ Экспорт ZIP завершён")

    def _on_export_failed(self, message: str, generation: int):
        self.status_label.setText("❌ Ошибка экспорта")
        QMessageBox.warning(
            self,
            "Ошибка экспорта",
            f"Экспорт не завершён:\n{message}\n\nЦелевой файл не изменён.",
        )

    def _on_export_cancelled(self, generation: int):
        self.status_label.setText("⏹ Экспорт отменён")

    def _on_save(self):
        if self._closing:
            return  # закрытие: новая работа не начинается
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
        if self._save_operation is not None and self._save_operation.is_running():
            return  # предыдущее сохранение ещё выполняется

        self.status_label.setText("💾 Сохранение маски…")
        generation = self._selection_generation
        operation = BackgroundOperation(
            self._save_mask_work, generation=generation, parent=self
        )
        operation.result_ready.connect(self._on_save_finished)
        operation.error_raised.connect(self._on_save_failed)
        self._save_operation = operation
        # Маска передаётся неизменяемым snapshot'ом на момент действия
        # пользователя (см. ui.background.snapshot).
        operation.start(mask=self.paint_label._mask, path=str(mask_path))

    def _save_mask_work(self, ctx, mask, path: str):
        """Worker: запись маски из неизменяемого snapshot'а."""
        ctx.status("запись маски")
        self.controller.save_mask(mask, Path(path))
        return {"name": Path(path).name}

    def _on_save_finished(self, result, generation: int):
        name = result["name"]
        QMessageBox.information(self, "Сохранено", f"Маска сохранена:\n{name}")
        self.status_label.setText(f"✅ Маска сохранена: {name}")

    def _on_save_failed(self, message: str, generation: int):
        self.status_label.setText("❌ Ошибка сохранения маски")
        QMessageBox.warning(
            self, "Ошибка", f"Не удалось сохранить маску:\n{message}"
        )
