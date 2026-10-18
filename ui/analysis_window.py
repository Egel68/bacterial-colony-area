"""
Окно анализа изображения.
Отображает изображение и результаты анализа фиксированной нейросетевой моделью.
"""

import cv2
import numpy as np
from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap

from ui.background import BackgroundOperation
from utils.image_loader import load_image_worker_safe
from PyQt6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.geometry import PetriInfo
from .controllers.analysis_controller import AnalysisController
from .responsive import ResponsiveMetrics, install_application_responsive_sizing


# Порог масштаба отображения, начиная с которого presentation строится
# напрямую из полного разрешения. Для меньших масштабов используется
# быстрый путь 1/2 (интерполяция вниз), данные анализа при этом не
# изменяются — уменьшается только отображаемое представление.
DISPLAY_FULL_RES_THRESHOLD = 0.5
DISPLAY_HALF_RES_SCALE = 0.5
# Неизменяемое значение рабочей границы для фиксированной модели.
MODEL_MARGIN_PERCENT = 8.0
APPROVED_MODEL_NAME = "NN:colony_seg"


def _composite_bgr(
    original,
    preprocessed,
    colony_mask,
    petri_info,
    margin,
    mode,
    show_contour,
    show_overlay,
    scale,
):
    """Композит представления в BGR для масштаба `scale` (1.0 или 0.5).

    Формулы наложения совпадают с прежним синхронным `_update_display`;
    при `scale=0.5` геометрия и толщины пересчитываются пропорционально.
    """

    def scaled_int(value):
        return max(1, int(round(value * scale))) if scale != 1.0 else int(value)

    if scale == 1.0:
        base = original
    else:
        h, w = original.shape[:2]
        base = cv2.resize(
            original,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    if mode == 2:  # Preprocessed
        if preprocessed is not None:
            if scale == 1.0:
                gray = preprocessed
            else:
                h, w = preprocessed.shape[:2]
                gray = cv2.resize(
                    preprocessed,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return base.copy()

    if mode == 3:  # Binary
        if colony_mask is not None or preprocessed is not None:
            mask = colony_mask if colony_mask is not None else preprocessed
            if scale != 1.0:
                h, w = mask.shape[:2]
                mask = cv2.resize(
                    mask,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.zeros_like(base)

    final_img = base.copy()
    if show_contour and petri_info is not None:
        if mode == 1:  # Original
            contour_color = (255, 0, 0)
        else:  # Result
            contour_color = (100, 100, 255)
        cv2.circle(
            final_img,
            (scaled_int(petri_info.cx), scaled_int(petri_info.cy)),
            scaled_int(petri_info.radius),
            contour_color,
            scaled_int(2),
        )
        if mode != 1:
            r_inner = int(petri_info.radius * (100 - margin) / 100)
            cv2.circle(
                final_img,
                (scaled_int(petri_info.cx), scaled_int(petri_info.cy)),
                scaled_int(r_inner),
                (255, 255, 0),
                1,
            )

    if mode != 0:
        return final_img

    # Result: зелёное наложение колоний + контуры.
    if show_overlay and colony_mask is not None:
        mask = colony_mask
        if scale != 1.0:
            h, w = mask.shape[:2]
            mask = cv2.resize(
                mask,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        overlay = final_img.copy()
        overlay[mask > 0] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0, final_img)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_img, cnts, -1, (0, 255, 0), 1)
    return final_img


def render_presentation(
    original,
    preprocessed,
    colony_mask,
    petri_info,
    margin,
    mode,
    show_contour,
    show_overlay,
    display_w,
    display_h,
):
    """Готовит отображаемое RGB-представление размером (display_h, display_w).

    Полноразмерные данные анализа не изменяются: для уменьшенного
    отображения используется быстрый путь 1/2, для масштабов >=
    DISPLAY_FULL_RES_THRESHOLD — прежний путь полного разрешения.
    Работает в worker-потоке, Qt widgets/QPixmap не затрагивает.
    """
    if original is None:
        return None
    orig_h, orig_w = original.shape[:2]
    display_w = max(2, int(display_w))
    display_h = max(2, int(display_h))
    display_scale = min(display_w / orig_w, display_h / orig_h)

    if display_scale >= DISPLAY_FULL_RES_THRESHOLD:
        scale = 1.0
    else:
        scale = DISPLAY_HALF_RES_SCALE

    composite = _composite_bgr(
        original,
        preprocessed,
        colony_mask,
        petri_info,
        margin,
        mode,
        show_contour,
        show_overlay,
        scale,
    )
    rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    ch, cw = rgb.shape[:2]
    fit = min(display_w / cw, display_h / ch)
    out_w = max(1, int(round(cw * fit)))
    out_h = max(1, int(round(ch * fit)))
    if (out_w, out_h) != (cw, ch):
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return rgb


def _presentation_work(
    ctx,
    original,
    preprocessed,
    colony_mask,
    petri_info,
    margin,
    mode,
    show_contour,
    show_overlay,
    display_w,
    display_h,
):
    """Worker-функция подготовки presentation (см. render_presentation)."""
    rgb = render_presentation(
        original,
        preprocessed,
        colony_mask,
        petri_info,
        margin,
        mode,
        show_contour,
        show_overlay,
        display_w,
        display_h,
    )
    return {"rgb": rgb, "mode": mode}


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._original_pixmap = None
        self._roi_info: PetriInfo | None = None
        self._drag_roi = False
        self._roi_changed_callback = None
        self._drag_offset = None
        self.setMouseTracking(True)

    def set_image(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()

    def set_roi(self, petri_info: PetriInfo | None):
        self._roi_info = petri_info
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._roi_info is None or self._original_pixmap is None:
            return
        target = self._displayed_image_rect()
        if target.isEmpty():
            return
        scale = target.width() / self._roi_info.image_shape[1]
        center_x = target.x() + self._roi_info.cx * scale
        center_y = target.y() + self._roi_info.cy * scale
        radius = self._roi_info.radius * scale
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
        painter.drawEllipse(
            round(center_x - radius),
            round(center_y - radius),
            round(radius * 2),
            round(radius * 2),
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawLine(
            round(center_x - 6), round(center_y), round(center_x + 6), round(center_y)
        )
        painter.drawLine(
            round(center_x), round(center_y - 6), round(center_x), round(center_y + 6)
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._roi_contains(
            event.position().toPoint()
        ):
            self._drag_roi = True
            point = event.position().toPoint()
            self._drag_offset = (
                self._roi_info.cx - self._widget_to_image_x(point.x()),
                self._roi_info.cy - self._widget_to_image_y(point.y()),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_roi and self._roi_info is not None:
            self._move_roi_to(event.position().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_roi:
            self._drag_roi = False
            self._drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _displayed_image_rect(self):
        if self._roi_info is None:
            pixmap = self.pixmap()
            if pixmap is None or pixmap.isNull():
                return self.rect()
            return pixmap.rect().translated(
                self.contentsRect().center() - pixmap.rect().center()
            )
        scale = min(
            self.contentsRect().width() / self._roi_info.image_shape[1],
            self.contentsRect().height() / self._roi_info.image_shape[0],
        )
        width = max(1, round(self._roi_info.image_shape[1] * scale))
        height = max(1, round(self._roi_info.image_shape[0] * scale))
        return QRect(
            self.contentsRect().center().x() - width // 2,
            self.contentsRect().center().y() - height // 2,
            width,
            height,
        )

    def _roi_contains(self, point):
        if self._roi_info is None:
            return False
        image_rect = self._displayed_image_rect()
        if not image_rect.contains(point):
            return False
        center_x = (
            image_rect.x()
            + self._roi_info.cx * image_rect.width() / self._roi_info.image_shape[1]
        )
        center_y = (
            image_rect.y()
            + self._roi_info.cy * image_rect.height() / self._roi_info.image_shape[0]
        )
        radius = (
            self._roi_info.radius * image_rect.width() / self._roi_info.image_shape[1]
        )
        return (point.x() - center_x) ** 2 + (point.y() - center_y) ** 2 <= radius**2

    def _move_roi_to(self, point):
        image_rect = self._displayed_image_rect()
        if image_rect.isEmpty():
            return
        info = self._roi_info
        cx = self._widget_to_image_x(point.x())
        cy = self._widget_to_image_y(point.y())
        if self._drag_offset is not None:
            cx += self._drag_offset[0]
            cy += self._drag_offset[1]
        radius = info.radius
        cx = min(max(cx, radius), info.image_shape[1] - radius - 1)
        cy = min(max(cy, radius), info.image_shape[0] - radius - 1)
        updated = PetriInfo(cx, cy, radius, info.image_shape)
        if updated != info:
            self._roi_info = updated
            self.update()
            if self._roi_changed_callback is not None:
                self._roi_changed_callback(updated)

    def _widget_to_image_x(self, x):
        image_rect = self._displayed_image_rect()
        return round(
            (x - image_rect.x()) * self._roi_info.image_shape[1] / image_rect.width()
        )

    def _widget_to_image_y(self, y):
        image_rect = self._displayed_image_rect()
        return round(
            (y - image_rect.y()) * self._roi_info.image_shape[0] / image_rect.height()
        )

    def _update_scaled_pixmap(self):
        if self._original_pixmap:
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.update()


class AnalysisWindow(QMainWindow):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path

        self.controller = AnalysisController()

        self.original_image = None
        self.display_image = None

        self.petri_mask = None
        self.petri_info: PetriInfo | None = None
        self.analysis_results = None
        self._analysis_generation = 0
        self._analysis_worker_running = False
        self._analysis_operation: BackgroundOperation | None = None
        self._init_operation: BackgroundOperation | None = None
        self._save_operation: BackgroundOperation | None = None
        self._display_operation: BackgroundOperation | None = None
        self._analysis_busy = False
        self._analysis_kind = "algorithm"
        self._closing = False
        self._pending_closes = 0
        # Поколение представления: устаревшие результаты фоновых операций
        # не должны перезаписывать более новое состояние окна.
        self._view_generation = 0
        # Поколение отображаемого представления (задача 5.3).
        self._display_generation = 0
        # Debounce перерисовки представления при ресайзе окна.
        self._resize_debounce = QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.setInterval(150)
        self._resize_debounce.timeout.connect(self._update_display)

        self._updating_ui = False
        self.setMinimumSize(0, 0)

        self._init_ui()
        self._responsive_sizer = install_application_responsive_sizing(
            self, minimum_scale=0.75
        )
        # Декодирование, поиск чашки и первый анализ — вне конструктора (5.1).
        self._start_initial_analysis()

    def _start_initial_analysis(self):
        """Фоновая загрузка изображения и авто-поиск чашки (задача 5.1)."""
        if self._closing:
            return
        self._view_generation += 1
        generation = self._view_generation
        self.status_label.setText("Загрузка изображения…")
        self.statusBar().showMessage("Загрузка изображения…")
        operation = BackgroundOperation(
            self._init_work, generation=generation, parent=self
        )
        operation.status_changed.connect(self.status_label.setText)
        operation.result_ready.connect(self._on_init_finished)
        operation.error_raised.connect(self._on_init_failed)
        self._init_operation = operation
        operation.start(image_path=self.image_path)

    def _init_work(self, ctx, image_path: str):
        """Worker: декодирование изображения и поиск чашки Петри."""
        ctx.status("Загрузка изображения…")
        image = load_image_worker_safe(image_path)
        ctx.checkpoint()
        ctx.status("Поиск чашки Петри…")
        petri_mask, petri_info = self.controller.find_petri_dish(image)
        return {"image": image, "petri_mask": petri_mask, "petri_info": petri_info}

    def _on_init_finished(self, result, generation: int):
        if generation != self._view_generation:
            return  # устаревший результат
        self.original_image = result["image"]
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self._apply_petri_result(result["petri_mask"], result["petri_info"])

    def _on_init_failed(self, message: str, generation: int):
        if generation != self._view_generation:
            return
        self.status_label.setText("Ошибка загрузки изображения")
        self.statusBar().showMessage("Ошибка загрузки изображения")
        QMessageBox.critical(self, "Ошибка загрузки", message)

    def _init_ui(self):
        self.setWindowTitle("Анализ бактериальных колоний")
        self.resize(1400, 950)  # Немного увеличим высоту

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, central_widget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        self.controls_scroll_area = QScrollArea()
        scroll = self.controls_scroll_area
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(450)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustIgnored)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll.setStyleSheet("QScrollArea {border: none; background-color: #1e1e2e;}")
        self.controls_widget = QWidget()
        self.controls_widget.setMinimumSize(0, 0)
        self.controls_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setSpacing(15)

        self._create_view_controls(self.controls_layout)
        self._create_analysis_controls(self.controls_layout)
        self._create_results_panel(self.controls_layout)
        self._create_action_buttons(self.controls_layout)
        self.controls_layout.addStretch()

        controls_widget = self.controls_widget
        scroll.setWidget(controls_widget)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._create_image_panel(self.main_layout)
        self.main_layout.addWidget(scroll, stretch=1)
        self._controls_reflowed = False
        self._update_controls_width()
        self.statusBar().showMessage("Готов к работе")

    def _update_controls_width(self):
        if not hasattr(self, "controls_scroll_area"):
            return
        if getattr(self, "_controls_reflowed", False):
            self.controls_scroll_area.setMaximumWidth(16_777_215)
            return

        sizer = getattr(self, "_responsive_sizer", None)
        metrics = getattr(sizer, "metrics", None) or ResponsiveMetrics(1.0)
        available_width = self.contentsRect().width() or self.width()
        minimum_width = metrics.dimension(280, minimum=1)
        maximum_width = metrics.dimension(450, minimum=minimum_width)
        target_width = max(
            minimum_width,
            min(maximum_width, round(available_width * 0.34)),
        )
        self.controls_scroll_area.setMaximumWidth(target_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_controls_width()
        # Перерисовка представления debounce-ится; быстрый ресайз не должен
        # накапливать фоновые подготовки (см. задачу 5.3).
        if self.original_image is not None:
            self._resize_debounce.start()
        compact = self.width() < 920 or self.height() < 680
        if compact != self._controls_reflowed:
            self._controls_reflowed = compact
            self.main_layout.setDirection(
                QBoxLayout.Direction.TopToBottom
                if compact
                else QBoxLayout.Direction.LeftToRight
            )
            sizer = getattr(self, "_responsive_sizer", None)
            metrics = getattr(sizer, "metrics", None) or ResponsiveMetrics(1.0)
            self.controls_scroll_area.setMaximumHeight(
                metrics.dimension(350, minimum=1) if compact else 16_777_215
            )
            self._update_controls_width()

    def _create_image_panel(self, layout: QHBoxLayout):
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: #11111b; border-radius: 15px;")
        image_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        vl = QVBoxLayout(image_frame)

        self.image_header = QLabel("Оригинальное изображение")
        self.image_header.setStyleSheet("color: #89b4fa; font-weight: bold;")
        self.image_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.image_header)

        self.image_label = ImageLabel()
        self.image_label._roi_changed_callback = self._on_roi_changed
        self.image_label.setToolTip(
            "Перетащите круг мышью, чтобы изменить центр области"
        )
        vl.addWidget(self.image_label, stretch=1)

        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("color: #a6adc8;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        vl.addWidget(self.progress_bar)

        layout.addWidget(image_frame, stretch=2)

    def _create_view_controls(self, layout: QVBoxLayout):
        group = QGroupBox("👁️ Режим просмотра")
        vl = QVBoxLayout(group)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            [
                "🔍 Результат (С наложением)",
                "📷 Оригинал",
                "🌗 Предобработка (Контраст)",
                "🏁 Бинарная маска (Ч/Б)",
            ]
        )
        self.view_mode_combo.currentIndexChanged.connect(self._update_display)
        vl.addWidget(self.view_mode_combo)

        self.show_petri_contour = QCheckBox("Показать контур чашки")
        self.show_petri_contour.setChecked(True)
        self.show_petri_contour.stateChanged.connect(self._update_display)
        vl.addWidget(self.show_petri_contour)

        self.show_area_overlay = QCheckBox("Закрасить колонии")
        self.show_area_overlay.setChecked(True)
        self.show_area_overlay.stateChanged.connect(self._update_display)
        vl.addWidget(self.show_area_overlay)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Радиус, px:"))
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 1)
        self.spin_radius.setEnabled(False)
        self.spin_radius.valueChanged.connect(self._on_roi_radius_changed)
        radius_row.addWidget(self.spin_radius)
        radius_row.addStretch()
        vl.addLayout(radius_row)
        layout.addWidget(group)

    def _create_analysis_controls(self, layout: QVBoxLayout):
        group = QGroupBox("🧠 Анализ")
        vl = QVBoxLayout(group)
        self.btn_apply = QPushButton("🔄 Пересчитать")
        self.btn_apply.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; margin-top: 10px;"
        )
        self.btn_apply.clicked.connect(self._run_colony_analysis_only)
        vl.addWidget(self.btn_apply)
        layout.addWidget(group)

    def _create_results_panel(self, layout: QVBoxLayout):
        group = QGroupBox("📊 Результаты")
        vl = QVBoxLayout(group)
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setMinimumHeight(58)
        vl.addWidget(self.text_results)
        layout.addWidget(group)

    def _create_action_buttons(self, layout: QVBoxLayout):
        h = QHBoxLayout()
        self.btn_save = QPushButton("💾 Сохранить")
        self.btn_save.clicked.connect(self._save_result)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.setObjectName("secondary")
        self.btn_close.clicked.connect(self.close)
        h.addWidget(self.btn_save)
        h.addWidget(self.btn_close)
        layout.addLayout(h)

    # --- ЛОГИКА ---

    def _apply_petri_result(self, petri_mask, petri_info):
        self.petri_mask = petri_mask
        self.petri_info = petri_info
        self.image_label.set_roi(None)
        self.spin_radius.setEnabled(False)
        if self.petri_info is None or self.petri_mask is None:
            self.status_label.setText("Чашка Петри не найдена")
            self.statusBar().showMessage("Анализ невозможен: чашка Петри не найдена")
            QMessageBox.warning(
                self,
                "Чашка Петри не найдена",
                "Не удалось автоматически определить чашку Петри. "
                "Проверьте качество и расположение снимка, затем откройте его снова.",
            )
            return
        self._set_roi_info(self.petri_info)
        self._run_colony_analysis_only()

    def _set_roi_info(self, info: PetriInfo):
        h, w = info.image_shape
        max_radius = max(1, min((w - 1) // 2, (h - 1) // 2))
        bounded_radius = min(max(info.radius, 1), max_radius)
        cx = min(max(info.cx, bounded_radius), w - bounded_radius - 1)
        cy = min(max(info.cy, bounded_radius), h - bounded_radius - 1)
        self.petri_info = PetriInfo(cx, cy, bounded_radius, info.image_shape)
        self.petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.petri_mask, self.petri_info.center, bounded_radius, 255, -1)
        self.image_label.set_roi(self.petri_info)
        self.spin_radius.blockSignals(True)
        self.spin_radius.setRange(1, max_radius)
        self.spin_radius.setValue(bounded_radius)
        self.spin_radius.setEnabled(True)
        self.spin_radius.blockSignals(False)
        if self.original_image is not None:
            self._update_display()

    def _on_roi_changed(self, info: PetriInfo):
        self._set_roi_info(info)
        self._invalidate_analysis_for_roi_change()

    def _on_roi_radius_changed(self, radius: int):
        if self.petri_info is None:
            return
        info = PetriInfo(
            self.petri_info.cx,
            self.petri_info.cy,
            radius,
            self.petri_info.image_shape,
        )
        self._set_roi_info(info)
        self._invalidate_analysis_for_roi_change()

    def _invalidate_analysis_for_roi_change(self):
        self._analysis_generation += 1
        self.analysis_results = None
        if self._analysis_worker_running and self._analysis_operation is not None:
            self._analysis_operation.cancel()
        self._analysis_busy = False
        self.btn_apply.setEnabled(not self._analysis_worker_running)
        self.progress_bar.hide()
        self.controller.set_algorithm_mask(
            np.zeros(self.petri_mask.shape, dtype=np.uint8)
        )
        self.text_results.setText("Область изменена. Нажмите «Пересчитать».")
        self.status_label.setText("Область анализа изменена")
        self.statusBar().showMessage("Проверьте область и нажмите «Пересчитать»")
        self._update_display()

    def _run_colony_analysis_only(self):
        """Запуск анализа: классический или выбранный алгоритм — в worker (5.2)."""
        if self._closing:
            return
        if not self.petri_info or self.original_image is None:
            return

        if self._analysis_worker_running or self._analysis_busy:
            return

        self._analysis_kind = "algorithm"
        self._analysis_busy = True
        self._analysis_worker_running = True
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.btn_apply.setEnabled(False)
        self.statusBar().showMessage(f"Запуск {APPROVED_MODEL_NAME}…")
        self.status_label.setText(f"Запуск {APPROVED_MODEL_NAME}…")

        operation = BackgroundOperation(
            self._analysis_work, generation=self._view_generation, parent=self
        )
        operation.progress_changed.connect(self._on_analysis_progress)
        operation.result_ready.connect(self._on_analysis_finished)
        operation.error_raised.connect(self._on_analysis_failed)
        operation.cancellation_confirmed.connect(self._on_analysis_cancelled)
        operation.finished.connect(self._on_analysis_worker_finished)
        self._analysis_operation = operation
        # Снимок фиксирует входы операции; модель и порог выбраны приложением.
        operation.start(
            image=self.original_image,
            petri_mask=self.petri_mask.copy(),
            petri_info=self.petri_info,
            analysis_generation=self._analysis_generation,
        )

    def _analysis_work(self, ctx, image, petri_mask, petri_info, analysis_generation):
        """Worker: инференс единственной утверждённой bundled-модели."""
        from testing.onnx_algorithm import register_bundled_models

        bundled_models = register_bundled_models()
        algorithm = next(
            (model for model in bundled_models if model.name == APPROVED_MODEL_NAME),
            None,
        )
        if algorithm is None:
            raise RuntimeError(
                f"Встроенная утверждённая модель {APPROVED_MODEL_NAME} не найдена."
            )
        ctx.status(f"Инференс: {algorithm.name}")

        def _progress(completed, total):
            ctx.progress(completed, total)
            ctx.checkpoint()  # кооперативная отмена между тайлами

        result, mask = self.controller.calculate_algorithm_result(
            image,
            petri_mask,
            algorithm,
            petri_info=petri_info,
            margin_percent=MODEL_MARGIN_PERCENT,
            progress_callback=_progress,
        )
        return {
            "kind": "algorithm",
            "results": result,
            "mask": mask,
            "analysis_generation": analysis_generation,
        }

    def _on_analysis_worker_finished(self, generation: int):
        self._analysis_worker_running = False
        if not self._analysis_busy and not self._closing:
            self.btn_apply.setEnabled(True)

    def _on_analysis_progress(self, completed, total):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(completed)
        message = f"Обработано тайлов {completed}/{total}"
        self.statusBar().showMessage(message)
        self.status_label.setText(message)

    def _on_analysis_finished(self, payload, generation: int):
        self._cleanup_analysis_ui()
        if generation != self._view_generation:
            return  # устаревший результат
        if payload["analysis_generation"] != self._analysis_generation:
            self.text_results.setText("Область изменена. Нажмите «Пересчитать».")
            return  # геометрия ROI изменилась во время инференса
        self.controller.set_algorithm_mask(payload["mask"])
        result = payload["results"]
        self.analysis_results = result
        self.text_results.setText(
            f"Количество колоний: {result.colony_count}\n"
            f"Покрытие (рабочей зоны): {result.coverage_percent:.2f}%\n"
            f"Площадь колоний: {result.colony_area_px} px"
        )
        self.statusBar().showMessage(f"Готово. Найдено: {result.colony_count}")
        self.status_label.setText(f"Готово. Найдено: {result.colony_count}")
        self._update_display()

    def _on_analysis_failed(self, message: str, generation: int):
        self._cleanup_analysis_ui()
        if generation != self._view_generation:
            return
        self.statusBar().showMessage("Ошибка нейросетевого анализа")
        title = "Ошибка модели"
        self.status_label.setText("Ошибка анализа")
        QMessageBox.critical(self, title, message)

    def _on_analysis_cancelled(self, generation: int):
        self._cleanup_analysis_ui()
        if generation != self._view_generation:
            return
        self.status_label.setText("Анализ отменён")
        self.statusBar().showMessage("Анализ отменён")

    def _cleanup_analysis_ui(self):
        self._analysis_busy = False
        self.btn_apply.setEnabled(not self._analysis_worker_running)
        self.progress_bar.hide()

    def closeEvent(self, event):
        """Deferred close: окно закрывается после safe-точки операций (5.2)."""
        running = [
            op
            for op in (
                self._init_operation,
                self._analysis_operation,
                self._save_operation,
                self._display_operation,
            )
            if op is not None and op.is_running()
        ]
        if not running:
            super().closeEvent(event)
            return
        event.ignore()
        if self._closing:
            return  # уже ждём безопасной точки
        self._closing = True
        self._pending_closes = len(running)
        message = "Завершение… Окно закроется после завершения анализа."
        self.statusBar().showMessage(message)
        self.status_label.setText(message)
        for operation in running:
            operation.cancel()
            operation.close_reached.connect(self._on_operation_close_reached)
            operation.request_close()

    def _on_operation_close_reached(self):
        self._pending_closes -= 1
        if self._pending_closes <= 0:
            self._closing = False
            self.close()

    def _update_display(self):
        """Готовит отображаемое представление в worker (задача 5.3).

        Полноразмерные данные анализа не изменяются; быстрая смена режима /
        ресайз оставляет отображённым только последний результат.
        """
        if self.original_image is None:
            return
        self._display_generation += 1
        generation = self._display_generation
        self.image_header.setText(self._view_title())

        debug = self.controller.debug_images
        preprocessed = debug.get("preprocessed")
        colony_mask = self.controller.colony_mask
        viewport = self.image_label.size()
        margin = MODEL_MARGIN_PERCENT
        mode = self.view_mode_combo.currentIndex()
        show_contour = self.show_petri_contour.isChecked()
        show_overlay = self.show_area_overlay.isChecked()

        operation = BackgroundOperation(
            _presentation_work,
            generation=generation,
            parent=self,
        )
        operation.result_ready.connect(self._on_presentation_ready)
        self._display_operation = operation
        # Массивы не копируются: окно только заменяет их целиком и не
        # мутирует in-place до завершения операции.
        operation.start(
            original=self.original_image,
            preprocessed=preprocessed,
            colony_mask=colony_mask,
            petri_info=self.petri_info,
            margin=margin,
            mode=mode,
            show_contour=show_contour,
            show_overlay=show_overlay,
            display_w=max(2, viewport.width()),
            display_h=max(2, viewport.height()),
            copy_inputs=False,
        )

    def _view_title(self) -> str:
        mode = self.view_mode_combo.currentIndex()
        if mode == 1:
            return "Оригинальное изображение"
        if mode == 2:
            dbg = self.controller.debug_images
            return (
                "Усиленный контраст"
                if "preprocessed" in dbg
                else "Оригинальное изображение"
            )
        if mode == 3:
            return "Бинарная маска"
        return "Результат анализа"

    def _on_presentation_ready(self, payload, generation: int):
        if generation != self._display_generation:
            return  # устаревшее представление: показываем только последнее
        rgb = payload["rgb"]
        if rgb is None:
            return
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.image_label.set_image(QPixmap.fromImage(q_image))
        self.image_label.set_roi(self.petri_info)

    def _save_result(self):
        if self.original_image is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить", "result.png", "Images (*.png *.jpg)"
        )
        if not file_path:
            return
        if self._closing:
            return
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            pixmap = getattr(self.image_label, "_original_pixmap", None)
        if pixmap is None or pixmap.isNull():
            # Представление ещё готовится: сохранять пока нечего.
            self.status_label.setText("Изображение ещё готовится…")
            return
        # Захватить ссылку на неизменяемый GUI-снимок и размеры сейчас; дорогое
        # масштабирование/композитинг/QImage-конструирование выполняет worker.
        snapshot_pixmap = QPixmap(pixmap)
        snapshot_size = snapshot_pixmap.size()
        self.status_label.setText("Сохранение результата…")
        operation = BackgroundOperation(
            self._save_work, generation=self._view_generation, parent=self
        )
        operation.result_ready.connect(self._on_save_finished)
        operation.error_raised.connect(self._on_save_failed)
        self._save_operation = operation
        operation.start(pixmap=snapshot_pixmap, size=snapshot_size, path=file_path)

    def _save_work(self, ctx, pixmap, size, path: str):
        """Worker: сохранение snapshot'а представления (кодирование по расширению)."""
        ctx.status("Сохранение результата…")
        image = QImage(size, QImage.Format.Format_RGB888)
        image.fill(Qt.GlobalColor.black)
        painter = None
        try:
            from PyQt6.QtGui import QPainter

            painter = QPainter(image)
            painter.drawPixmap(0, 0, pixmap)
        finally:
            if painter is not None:
                painter.end()
        if not image.save(path):
            raise ValueError(f"Не удалось сохранить файл: {path}")
        return {"path": path}

    def _on_save_finished(self, payload, generation: int):
        self.status_label.setText("Результат сохранён")
        self.statusBar().showMessage(f"Сохранено: {payload['path']}")

    def _on_save_failed(self, message: str, generation: int):
        self.status_label.setText("Ошибка сохранения")
        QMessageBox.critical(self, "Ошибка сохранения", message)
