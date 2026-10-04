"""
Окно анализа изображения.
Отображает изображение и результаты анализа с расширенными настройками.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from analysis.params import AnalysisParams
from ui.background import BackgroundOperation
from utils.image_loader import load_image_worker_safe
from PyQt6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QSlider,
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

    def set_image(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._original_pixmap:
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


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
        self._analysis_operation: BackgroundOperation | None = None
        self._init_operation: BackgroundOperation | None = None
        self._save_operation: BackgroundOperation | None = None
        self._display_operation: BackgroundOperation | None = None
        self._analysis_busy = False
        self._analysis_kind = "classic"
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
        self._create_geometry_controls(self.controls_layout)
        self._create_algorithm_controls(self.controls_layout)
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

        self.algorithm_combo = QComboBox()
        self._load_analysis_algorithms()
        vl.addWidget(QLabel("Алгоритм распознавания:"))
        vl.addWidget(self.algorithm_combo)
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)

        layout.addWidget(group)

    def _create_geometry_controls(self, layout: QVBoxLayout):
        group = QGroupBox("📏 Геометрия чашки")
        vl = QFormLayout(group)

        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 5000)
        self.spin_x.setSuffix(" px")
        self.spin_x.valueChanged.connect(self._on_geometry_changed)
        vl.addRow("Центр X:", self.spin_x)

        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 5000)
        self.spin_y.setSuffix(" px")
        self.spin_y.valueChanged.connect(self._on_geometry_changed)
        vl.addRow("Центр Y:", self.spin_y)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(10, 3000)
        self.spin_radius.setSuffix(" px")
        self.spin_radius.valueChanged.connect(self._on_geometry_changed)
        vl.addRow("Радиус:", self.spin_radius)

        self.btn_reset_geometry = QPushButton("Сбросить к авто-поиску")
        self.btn_reset_geometry.setStyleSheet("background-color: #45475a;")
        self.btn_reset_geometry.clicked.connect(self._reset_geometry)
        vl.addRow(self.btn_reset_geometry)

        layout.addWidget(group)

    def _create_algorithm_controls(self, layout: QVBoxLayout):
        group = QGroupBox("⚙️ Параметры алгоритма")
        vl = QVBoxLayout(group)

        # Чувствительность
        vl.addWidget(QLabel("Чувствительность:"))
        h_sens = QHBoxLayout()
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(1, 100)
        self.slider_sens.setValue(50)
        self.slider_sens.valueChanged.connect(
            lambda v: self.label_sens.setText(f"{v}%")
        )
        self.label_sens = QLabel("50%")
        h_sens.addWidget(self.slider_sens)
        h_sens.addWidget(self.label_sens)
        vl.addLayout(h_sens)

        # Контраст
        vl.addWidget(QLabel("Усиление контраста:"))
        h_cont = QHBoxLayout()
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(5, 30)
        self.slider_contrast.setValue(10)
        self.slider_contrast.valueChanged.connect(
            lambda v: self.label_contrast.setText(f"{v / 10:.1f}x")
        )
        self.label_contrast = QLabel("1.0x")
        h_cont.addWidget(self.slider_contrast)
        h_cont.addWidget(self.label_contrast)
        vl.addLayout(h_cont)

        # Отступ
        vl.addWidget(QLabel("Отступ от края (%):"))
        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0, 30)
        self.spin_margin.setValue(8.0)
        vl.addWidget(self.spin_margin)

        # Мин размер
        vl.addWidget(QLabel("Мин. размер колонии (px):"))
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 1000)
        self.spin_min_size.setValue(50)
        vl.addWidget(self.spin_min_size)

        # --- Секция сплошных зон (НОВАЯ) ---
        fill_frame = QFrame()
        fill_frame.setStyleSheet("background-color: #313244; border-radius: 6px;")
        fill_layout = QVBoxLayout(fill_frame)

        self.chk_solid_fill = QCheckBox("💧 Заполнять сплошные зоны")
        self.chk_solid_fill.setToolTip(
            "Включите для сплошных мазков бактерий. Выключите для отдельных мелких колоний."
        )
        fill_layout.addWidget(self.chk_solid_fill)

        fill_h = QHBoxLayout()
        fill_h.addWidget(QLabel("Сила заполнения:"))
        self.spin_fill_strength = QSpinBox()
        self.spin_fill_strength.setRange(1, 100)
        self.spin_fill_strength.setValue(15)
        self.spin_fill_strength.setToolTip(
            "Радиус объединения. Увеличьте, если остаются черные дыры внутри пятен."
        )
        fill_h.addWidget(self.spin_fill_strength)
        fill_layout.addLayout(fill_h)

        vl.addWidget(fill_frame)
        # ------------------------------------

        btn_apply = QPushButton("🔄 Пересчитать")
        btn_apply.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; font-weight: bold; margin-top: 10px;"
        )
        btn_apply.clicked.connect(self._run_colony_analysis_only)
        self.btn_apply = btn_apply
        vl.addWidget(btn_apply)

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

    def _run_full_analysis(self):
        """Авто-поиск чашки и анализ выполняются вне GUI-потока (5.1/5.2)."""
        if self._closing or self.original_image is None:
            return
        if self._analysis_busy:
            return
        self._view_generation += 1
        generation = self._view_generation
        self.status_label.setText("Поиск чашки Петри...")
        self.statusBar().showMessage("Поиск чашки Петри...")
        operation = BackgroundOperation(
            self._locate_work, generation=generation, parent=self
        )
        operation.status_changed.connect(self.status_label.setText)
        operation.result_ready.connect(self._on_locate_finished)
        operation.error_raised.connect(self._on_locate_failed)
        self._init_operation = operation
        operation.start(image=self.original_image)

    def _locate_work(self, ctx, image):
        """Worker: повторный поиск чашки Петри."""
        ctx.status("Поиск чашки Петри...")
        return self.controller.find_petri_dish(image)

    def _on_locate_finished(self, result, generation: int):
        if generation != self._view_generation:
            return  # устаревший результат
        self._apply_petri_result(result[0], result[1])

    def _on_locate_failed(self, message: str, generation: int):
        if generation != self._view_generation:
            return
        self.status_label.setText("Ошибка поиска чашки")
        self.statusBar().showMessage("Ошибка поиска чашки")
        QMessageBox.critical(self, "Ошибка поиска чашки", message)

    def _apply_petri_result(self, petri_mask, petri_info):
        self.petri_mask = petri_mask
        self.petri_info = petri_info
        if self.petri_info:
            self._updating_ui = True
            self.spin_x.setValue(self.petri_info.cx)
            self.spin_y.setValue(self.petri_info.cy)
            self.spin_radius.setValue(self.petri_info.radius)
            self._updating_ui = False
            self._run_colony_analysis_only()
        else:
            QMessageBox.warning(
                self, "Ошибка", "Чашка Петри не найдена. Настройте вручную."
            )
            h, w = self.original_image.shape[:2]
            self.petri_info = PetriInfo(
                cx=w // 2,
                cy=h // 2,
                radius=int(min(w, h) * 0.4),
                image_shape=(h, w),
            )
            self._updating_ui = True
            self.spin_x.setValue(w // 2)
            self.spin_y.setValue(h // 2)
            self.spin_radius.setValue(int(min(w, h) * 0.4))
            self._updating_ui = False

    def _on_geometry_changed(self):
        if self._updating_ui:
            return
        cx = self.spin_x.value()
        cy = self.spin_y.value()
        r = self.spin_radius.value()
        h, w = self.original_image.shape[:2]
        self.petri_info = PetriInfo(
            cx=cx,
            cy=cy,
            radius=r,
            image_shape=(h, w),
        )
        self.petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.petri_mask, (cx, cy), r, 255, -1)
        self._update_display()

    def _reset_geometry(self):
        self._run_full_analysis()

    def _run_colony_analysis_only(self):
        """Запуск анализа: классический или выбранный алгоритм — в worker (5.2)."""
        if self._closing:
            return
        if not self.petri_info or self.original_image is None:
            return

        if self._analysis_busy:
            return

        algorithm = self._selected_algorithm()
        self._analysis_kind = "algorithm" if algorithm is not None else "classic"
        algorithm_name = (
            self.algorithm_combo.currentData()
            if algorithm is not None
            else "__classic__"
        )
        params = AnalysisParams(
            sensitivity=self.slider_sens.value() / 100.0,
            contrast=self.slider_contrast.value() / 10.0,
            margin_percent=self.spin_margin.value(),
            min_colony_size=self.spin_min_size.value(),
            solid_fill=self.chk_solid_fill.isChecked(),
            fill_strength=self.spin_fill_strength.value(),
        )
        margin = self.spin_margin.value()

        self._analysis_busy = True
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.btn_apply.setEnabled(False)
        self.algorithm_combo.setEnabled(False)
        for widget in (self.spin_x, self.spin_y, self.spin_radius, self.spin_margin):
            widget.setEnabled(False)
        if algorithm is not None:
            self.statusBar().showMessage(f"Запуск {algorithm.name}…")
            self.status_label.setText(f"Запуск {algorithm.name}…")
        else:
            self.status_label.setText("Анализ колоний...")
            self.statusBar().showMessage("Анализ колоний...")

        operation = BackgroundOperation(
            self._analysis_work, generation=self._view_generation, parent=self
        )
        operation.progress_changed.connect(self._on_analysis_progress)
        operation.result_ready.connect(self._on_analysis_finished)
        operation.error_raised.connect(self._on_analysis_failed)
        operation.cancellation_confirmed.connect(self._on_analysis_cancelled)
        self._analysis_operation = operation
        # Входы фиксируются snapshot'ом: маска/геометрия на момент запуска.
        operation.start(
            image=self.original_image,
            petri_mask=self.petri_mask,
            petri_info=self.petri_info,
            params=params,
            margin=margin,
            algorithm_name=algorithm_name,
        )

    def _analysis_work(
        self, ctx, image, petri_mask, petri_info, params, margin, algorithm_name
    ):
        """Worker: классический анализ или инференс выбранного алгоритма."""
        if algorithm_name == "__classic__":
            ctx.status("Анализ колоний...")
            results = self.controller.analyze(
                image,
                petri_mask,
                params=params,
                petri_info=petri_info,
                blur_size=5,
            )
            return {"kind": "classic", "results": results}

        from testing.registry import get_algorithm

        algorithm = get_algorithm(algorithm_name)
        ctx.status(f"Инференс: {algorithm.name}")

        def _progress(completed, total):
            ctx.progress(completed, total)
            ctx.checkpoint()  # кооперативная отмена между тайлами

        result, mask = self.controller.calculate_algorithm_result(
            image,
            petri_mask,
            algorithm,
            petri_info=petri_info,
            margin_percent=margin,
            progress_callback=_progress,
        )
        return {"kind": "algorithm", "results": result, "mask": mask}

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
        if payload["kind"] == "algorithm":
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
        if self._analysis_kind == "algorithm":
            self.statusBar().showMessage("Ошибка нейросетевого анализа")
            title = "Ошибка модели"
        else:
            self.statusBar().showMessage("Ошибка классического анализа")
            title = "Ошибка алгоритма"
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
        self.btn_apply.setEnabled(True)
        self.algorithm_combo.setEnabled(True)
        for widget in (self.spin_x, self.spin_y, self.spin_radius, self.spin_margin):
            widget.setEnabled(True)
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

    def _load_analysis_algorithms(self):
        from testing.onnx_algorithm import register_bundled_models
        from testing.registry import list_algorithms

        register_bundled_models()
        self.algorithm_combo.addItem("Классический (параметры ниже)", "__classic__")
        for name in list_algorithms():
            self.algorithm_combo.addItem(name, name)

    def _selected_algorithm(self):
        name = self.algorithm_combo.currentData()
        if not name or name == "__classic__":
            return None
        from testing.registry import get_algorithm

        return get_algorithm(name)

    def _on_algorithm_changed(self, _index):
        name = self.algorithm_combo.currentData()
        uses_classic_controls = name == "__classic__"
        for widget in (
            self.slider_sens,
            self.slider_contrast,
            self.spin_min_size,
            self.chk_solid_fill,
            self.spin_fill_strength,
        ):
            widget.setEnabled(uses_classic_controls)

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
        # binary-слой классического пути приоритетнее маски колоний:
        # сохраняем прежнюю семантику отображения бинарного режима.
        binary_layer = debug.get("binary")
        if self.view_mode_combo.currentIndex() == 3 and binary_layer is not None:
            colony_mask = binary_layer

        viewport = self.image_label.size()
        margin = self.spin_margin.value()
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
