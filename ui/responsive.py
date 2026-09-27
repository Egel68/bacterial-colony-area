"""Общие метрики и обработчик адаптивного размера окон."""

from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLayout, QWidget

from .styles import scale_stylesheet


DEFAULT_REFERENCE_SIZE = (1280, 800)
DEFAULT_MINIMUM_SCALE = 0.75
DEFAULT_MAXIMUM_SCALE = 1.0
MINIMUM_FONT_SCALE = 0.9
WIDGET_SIZE_MAX = 16_777_215


def calculate_scale(
    available_width: int,
    available_height: int,
    reference_width: int = DEFAULT_REFERENCE_SIZE[0],
    reference_height: int = DEFAULT_REFERENCE_SIZE[1],
    minimum_scale: float = DEFAULT_MINIMUM_SCALE,
    maximum_scale: float = DEFAULT_MAXIMUM_SCALE,
) -> float:
    """Вычисляет ограниченный масштаб для логических пикселей Qt."""
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("Размеры макета должны быть положительными")
    if minimum_scale <= 0 or maximum_scale < minimum_scale:
        raise ValueError("Некорректные границы масштаба")

    width_scale = max(0, available_width) / reference_width
    height_scale = max(0, available_height) / reference_height
    return max(minimum_scale, min(maximum_scale, width_scale, height_scale))


def scale_dimension(value: int, scale: float, minimum: int = 0) -> int:
    """Масштабирует размер от исходной метрики, не накапливая округление."""
    if value == 0:
        return 0
    return max(minimum, round(value * scale))


@dataclass(frozen=True)
class ResponsiveMetrics:
    """Согласованный набор размеров для одного окна."""

    scale: float
    minimum_font_scale: float = MINIMUM_FONT_SCALE

    @property
    def font_scale(self) -> float:
        return max(self.minimum_font_scale, self.scale)

    def dimension(self, value: int, minimum: int = 0) -> int:
        return scale_dimension(value, self.scale, minimum)

    def font_px(self, value: int, minimum: int = 11) -> int:
        return scale_dimension(value, self.font_scale, minimum)


class ResponsiveSizer(QObject):
    """Масштабирует типографику и layout-метрики при изменении окна."""

    def __init__(
        self,
        window: QWidget,
        reference_size: tuple[int, int] = DEFAULT_REFERENCE_SIZE,
        minimum_scale: float = DEFAULT_MINIMUM_SCALE,
        maximum_scale: float = DEFAULT_MAXIMUM_SCALE,
        style_factory: Callable[[ResponsiveMetrics], str] | None = None,
    ):
        super().__init__(window)
        self.window = window
        self.reference_size = reference_size
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.style_factory = style_factory
        self.metrics: ResponsiveMetrics | None = None
        self._base_font = QFont(window.font())
        self._base_layout_metrics = self._capture_layout_metrics(window)
        self._base_widget_constraints = self._capture_widget_constraints(window)
        self._base_stylesheets = {
            widget: widget.styleSheet()
            for widget in [window, *window.findChildren(QWidget)]
            if widget.styleSheet()
        }
        self._base_widget_fonts = {
            widget: QFont(widget.font())
            for widget in [window, *window.findChildren(QWidget)]
        }
        window.installEventFilter(self)
        window._responsive_sizer = self
        self.update_scale()

    @staticmethod
    def _capture_layout_metrics(window: QWidget):
        layouts = list(window.findChildren(QLayout))
        root_layout = window.layout()
        if root_layout is not None and root_layout not in layouts:
            layouts.append(root_layout)

        values = {}
        for layout in layouts:
            margins = layout.contentsMargins()
            values[layout] = (
                (margins.left(), margins.top(), margins.right(), margins.bottom()),
                layout.spacing(),
            )
        return values

    @staticmethod
    def _capture_widget_constraints(window: QWidget):
        widgets = [window, *window.findChildren(QWidget)]
        constraints = {}
        for widget in widgets:
            minimum = widget.minimumSize()
            maximum = widget.maximumSize()
            constrained_width = minimum.width() > 0 or maximum.width() < WIDGET_SIZE_MAX
            constrained_height = (
                minimum.height() > 0 or maximum.height() < WIDGET_SIZE_MAX
            )
            if constrained_width or constrained_height:
                constraints[widget] = (
                    (minimum.width(), minimum.height()),
                    (maximum.width(), maximum.height()),
                    constrained_width,
                    constrained_height,
                )
        return constraints

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Type.Resize:
            self.update_scale()
        return False

    def update_scale(self) -> ResponsiveMetrics:
        """Применяет метрики от исходных значений при текущем размере окна."""
        width, height = self.reference_size
        client_area = self.window.contentsRect()
        scale = calculate_scale(
            client_area.width(),
            client_area.height(),
            reference_width=width,
            reference_height=height,
            minimum_scale=self.minimum_scale,
            maximum_scale=self.maximum_scale,
        )
        new_metrics = ResponsiveMetrics(scale)
        if self.metrics is not None and self.metrics == new_metrics:
            return self.metrics

        self.metrics = new_metrics

        for widget, base_font in self._base_widget_fonts.items():
            font = QFont(base_font)
            point_size = base_font.pointSizeF()
            if point_size > 0:
                font.setPointSizeF(max(9.0, point_size * new_metrics.font_scale))
            widget.setFont(font)

        for layout, (margins, spacing) in self._base_layout_metrics.items():
            scaled_margins = [
                scale_dimension(value, new_metrics.scale, 1 if value > 0 else 0)
                for value in margins
            ]
            layout.setContentsMargins(*scaled_margins)
            if spacing >= 0:
                layout.setSpacing(
                    scale_dimension(spacing, new_metrics.scale, 1 if spacing > 0 else 0)
                )

        for widget, (
            minimum,
            maximum,
            constrained_width,
            constrained_height,
        ) in self._base_widget_constraints.items():
            if widget is self.window:
                continue
            if constrained_width:
                min_width = scale_dimension(
                    minimum[0], new_metrics.scale, 1 if minimum[0] > 0 else 0
                )
                max_width = (
                    scale_dimension(maximum[0], new_metrics.scale, min_width)
                    if maximum[0] < WIDGET_SIZE_MAX
                    else WIDGET_SIZE_MAX
                )
                widget.setMinimumWidth(min_width)
                widget.setMaximumWidth(max_width)
            if constrained_height:
                min_height = scale_dimension(
                    minimum[1], new_metrics.scale, 1 if minimum[1] > 0 else 0
                )
                max_height = (
                    scale_dimension(maximum[1], new_metrics.scale, min_height)
                    if maximum[1] < WIDGET_SIZE_MAX
                    else WIDGET_SIZE_MAX
                )
                widget.setMinimumHeight(min_height)
                widget.setMaximumHeight(max_height)

        window_style = self._base_stylesheets.get(self.window, "")
        if self.style_factory is not None:
            window_style = self.style_factory(new_metrics)
        self.window.setStyleSheet(
            scale_stylesheet(window_style, new_metrics.scale, new_metrics.font_scale)
        )
        for widget, base_stylesheet in self._base_stylesheets.items():
            if widget is self.window:
                continue
            widget.setStyleSheet(
                scale_stylesheet(
                    base_stylesheet, new_metrics.scale, new_metrics.font_scale
                )
            )
        self.window.setProperty("responsiveScale", new_metrics.scale)
        return new_metrics

    def set_widget_stylesheet(self, widget: QWidget, stylesheet: str) -> None:
        """Сохраняет исходный локальный QSS и применяет его в текущем масштабе."""
        self._base_stylesheets[widget] = stylesheet
        metrics = self.metrics or ResponsiveMetrics(1.0)
        widget.setStyleSheet(
            scale_stylesheet(stylesheet, metrics.scale, metrics.font_scale)
        )


def install_responsive_sizing(
    window: QWidget,
    reference_size: tuple[int, int] = DEFAULT_REFERENCE_SIZE,
    minimum_scale: float = DEFAULT_MINIMUM_SCALE,
    maximum_scale: float = DEFAULT_MAXIMUM_SCALE,
    style_factory: Callable[[ResponsiveMetrics], str] | None = None,
) -> ResponsiveSizer:
    """Подключает адаптивные метрики к уже собранному окну."""
    return ResponsiveSizer(
        window,
        reference_size=reference_size,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        style_factory=style_factory,
    )


def set_responsive_stylesheet(widget: QWidget, stylesheet: str) -> None:
    """Устанавливает QSS с сохранением базовых px для последующих resize."""
    sizer = getattr(widget.window(), "_responsive_sizer", None)
    if sizer is None:
        widget.setStyleSheet(stylesheet)
        return
    sizer.set_widget_stylesheet(widget, stylesheet)


def install_application_responsive_sizing(
    window: QWidget,
    reference_size: tuple[int, int] = DEFAULT_REFERENCE_SIZE,
    minimum_scale: float = DEFAULT_MINIMUM_SCALE,
    maximum_scale: float = DEFAULT_MAXIMUM_SCALE,
) -> ResponsiveSizer:
    """Подключает общий адаптивный QSS приложения к окну."""
    from .styles import get_application_style

    return install_responsive_sizing(
        window,
        reference_size=reference_size,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        style_factory=lambda _metrics: get_application_style(),
    )
