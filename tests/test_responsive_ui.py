"""Тесты общей политики адаптивного размера интерфейса."""

import pytest
import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QDialog,
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from analysis.geometry import PetriInfo
from analysis.results import AnalysisResult
from labeling.labeling_window import LabelingWindow
from labeling.session_manager import SessionManager, ensure_session_structure
from ui.analysis_window import AnalysisWindow
from ui.controllers.analysis_controller import AnalysisController
from ui.labeling_session_dialog import LabelingSessionDialog
from ui.main_window import MainWindow
from ui.responsive import (
    DEFAULT_MINIMUM_SCALE,
    ResponsiveMetrics,
    calculate_scale,
    install_responsive_sizing,
    scale_dimension,
)
from ui.styles import get_application_style
from ui.testing_window import TestingWindow


def test_scale_is_clamped_and_monotonic():
    sizes = [(480, 320), (640, 480), (800, 600), (1280, 800), (1600, 1000)]
    scales = [calculate_scale(width, height) for width, height in sizes]

    assert scales == sorted(scales)
    assert scales[0] == DEFAULT_MINIMUM_SCALE
    assert scales[2] == DEFAULT_MINIMUM_SCALE
    assert scales[3] == 1.0
    assert scales[4] == 1.0


def test_scale_uses_logical_dimensions_and_rejects_invalid_reference():
    assert calculate_scale(800, 800) == DEFAULT_MINIMUM_SCALE
    with pytest.raises(ValueError, match="положительными"):
        calculate_scale(800, 600, reference_width=0)


def test_metric_scaling_uses_stable_baseline_and_readable_font_floor():
    metrics = ResponsiveMetrics(scale=0.85)

    assert metrics.dimension(20) == 17
    assert metrics.font_px(14) == 13
    assert metrics.font_px(10) == 11
    assert scale_dimension(0, 0.5) == 0


def test_application_stylesheet_scales_dimensions_without_shrinking_font_floor():
    normal = get_application_style()
    compact = get_application_style(scale=0.85)

    assert "padding: 12px 24px" in normal
    assert "padding: 10px 20px" in compact
    assert "font-size: 14px" in normal
    assert "font-size: 13px" in compact


@pytest.mark.gui
def test_local_stylesheet_metrics_scale_and_restore_without_parent_override(qtbot):
    window = QWidget()
    layout = QVBoxLayout(window)
    button = QPushButton("Кнопка", window)
    base_style = "QPushButton { font-size: 14px; padding: 10px 20px; }"
    button.setStyleSheet(base_style)
    layout.addWidget(button)
    window.resize(700, 500)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    client_area = window.contentsRect()
    sizer = install_responsive_sizing(
        window,
        reference_size=(client_area.width(), client_area.height()),
    )
    assert button.styleSheet() == base_style

    window.resize(500, 350)
    qtbot.waitUntil(lambda: sizer.metrics.scale == DEFAULT_MINIMUM_SCALE)
    assert "font-size: 13px" in button.styleSheet()
    assert "padding: 8px 15px" in button.styleSheet()

    window.resize(700, 500)
    qtbot.waitUntil(lambda: sizer.metrics.scale == 1.0)
    assert button.styleSheet() == base_style


@pytest.mark.gui
def test_responsive_sizer_restores_layout_metrics_after_resize(qtbot):
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)
    base_font = QFont(window.font())
    base_font.setPointSizeF(12.0)
    window.setFont(base_font)
    window.resize(700, 500)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    client_area = window.contentsRect()
    sizer = install_responsive_sizing(
        window,
        reference_size=(client_area.width(), client_area.height()),
    )
    base_margins = layout.contentsMargins()
    base_spacing = layout.spacing()
    base_font_size = base_font.pointSizeF()

    for _ in range(3):
        window.resize(500, 350)
        qtbot.waitUntil(lambda: sizer.metrics.scale == DEFAULT_MINIMUM_SCALE)
        assert layout.contentsMargins().left() == round(
            base_margins.left() * DEFAULT_MINIMUM_SCALE
        )
        assert layout.spacing() == round(base_spacing * DEFAULT_MINIMUM_SCALE)
        assert window.font().pointSizeF() == pytest.approx(base_font_size * 0.9)

        window.resize(700, 500)
        qtbot.waitUntil(lambda: sizer.metrics.scale == 1.0)
        assert layout.spacing() == base_spacing
        assert window.font().pointSizeF() == pytest.approx(base_font_size)

    restored_margins = layout.contentsMargins()
    assert (
        restored_margins.left(),
        restored_margins.top(),
        restored_margins.right(),
        restored_margins.bottom(),
    ) == (
        base_margins.left(),
        base_margins.top(),
        base_margins.right(),
        base_margins.bottom(),
    )
    assert layout.spacing() == base_spacing


@pytest.mark.gui
def test_responsive_sizer_scales_and_restores_fixed_width_constraints(qtbot):
    window = QWidget()
    layout = QVBoxLayout(window)
    widget = QWidget(window)
    widget.setFixedWidth(240)
    layout.addWidget(widget)
    window.resize(700, 500)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    client_area = window.contentsRect()
    sizer = install_responsive_sizing(
        window, reference_size=(client_area.width(), client_area.height())
    )

    window.resize(500, 350)
    qtbot.waitUntil(lambda: sizer.metrics.scale == DEFAULT_MINIMUM_SCALE)
    assert widget.minimumWidth() == round(240 * DEFAULT_MINIMUM_SCALE)
    assert widget.maximumWidth() == round(240 * DEFAULT_MINIMUM_SCALE)

    window.resize(700, 500)
    qtbot.waitUntil(lambda: sizer.metrics.scale == 1.0)
    assert widget.minimumWidth() == 240
    assert widget.maximumWidth() == 240


def _show_at_size(qtbot, window, size, restore=False):
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    if restore and window.isMaximized():
        window.showNormal()
        qtbot.wait(20)
    window.resize(*size)
    qtbot.waitUntil(
        lambda: (
            abs(window.width() - size[0]) <= 2 and abs(window.height() - size[1]) <= 2
        ),
        timeout=3000,
    )
    # Qt устанавливает геометрию верхнего окна синхронно, но перестройка layout
    # и вложенных viewport завершается следующей итерацией event loop.
    qtbot.wait(20)


def _assert_inside(widget, ancestor):
    point = widget.mapTo(ancestor, QPoint(0, 0))
    assert point.x() >= 0 and point.y() >= 0
    assert point.x() + widget.width() <= ancestor.width()
    assert point.y() + widget.height() <= ancestor.height()


def _assert_no_overlap(widgets, common_parent):
    rects = []
    for widget in widgets:
        point = widget.mapTo(common_parent, QPoint(0, 0))
        rect = widget.rect().translated(point)
        for previous in rects:
            assert not rect.intersects(previous)
        rects.append(rect)


def _assert_no_interactive_overlap(root):
    interactive_types = (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QListWidget,
        QSlider,
        QTableWidget,
    )
    widgets = [
        widget
        for widget in root.findChildren(QWidget)
        if isinstance(widget, interactive_types) and widget.isVisible()
    ]
    for index, widget in enumerate(widgets):
        for other in widgets[index + 1 :]:
            if widget.isAncestorOf(other) or other.isAncestorOf(widget):
                continue
            first = widget.rect().translated(widget.mapTo(root, QPoint(0, 0)))
            second = other.rect().translated(other.mapTo(root, QPoint(0, 0)))
            assert not first.intersects(second), (
                f"Interactive siblings overlap: {widget!r} and {other!r}"
            )


def _scroll_into_view(qtbot, widget, scroll_area):
    viewport = scroll_area.viewport()
    for _ in range(3):
        point = widget.mapTo(viewport, QPoint(0, 0))
        vertical = scroll_area.verticalScrollBar()
        horizontal = scroll_area.horizontalScrollBar()
        if point.y() < 0:
            vertical.setValue(vertical.value() + point.y())
        elif point.y() + widget.height() > viewport.height():
            vertical.setValue(
                vertical.value() + point.y() + widget.height() - viewport.height()
            )
        if point.x() < 0:
            horizontal.setValue(horizontal.value() + point.x())
        elif point.x() + widget.width() > viewport.width():
            horizontal.setValue(
                horizontal.value() + point.x() + widget.width() - viewport.width()
            )
        qtbot.wait(10)
        point = widget.mapTo(viewport, QPoint(0, 0))
        if (
            0 <= point.x()
            and 0 <= point.y()
            and point.x() + widget.width() <= viewport.width()
            and point.y() + widget.height() <= viewport.height()
        ):
            return
    _assert_inside(widget, viewport)


@pytest.fixture
def synthetic_analysis(monkeypatch, tmp_path):
    image_path = tmp_path / "dish.png"
    image = np.full((180, 180, 3), 96, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    info = PetriInfo(90, 90, 78, (180, 180))
    mask = np.zeros((180, 180), dtype=np.uint8)
    cv2.circle(mask, info.center, info.radius, 255, -1)
    result = AnalysisResult(
        colony_count=0,
        colony_area_px=0,
        colony_area_mm2=0.0,
        petri_area_px=0,
        petri_area_mm2=0.0,
        coverage_percent=0.0,
        px_to_mm2=0.0,
    )
    monkeypatch.setattr(
        AnalysisController,
        "find_petri_dish",
        lambda _self, _image: (mask.copy(), info),
    )
    monkeypatch.setattr(
        AnalysisController,
        "analyze",
        lambda _self, *_args, **_kwargs: result,
    )
    return image_path


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1280, 800), (800, 600)])
def test_main_window_compact_and_standard_layout(qtbot, size):
    window = MainWindow()
    _show_at_size(qtbot, window, size)

    assert window.minimumSize().width() <= size[0]
    assert window.minimumSize().height() <= size[1]
    actions = [
        window.open_button,
        window.analyze_button,
        window.label_button,
        window.testing_button,
    ]
    for action in actions:
        assert action.isVisible()
        _assert_inside(action, window)
    _assert_no_overlap(actions, window)
    assert window.path_input.isVisible()
    _assert_inside(window.path_input, window)
    _assert_no_overlap([window.path_input, window.open_button], window)
    _assert_no_interactive_overlap(window)


@pytest.mark.gui
def test_main_window_restores_normal_metrics_after_compact_resize(qtbot):
    window = MainWindow()
    _show_at_size(qtbot, window, (1280, 800))
    assert window.property("responsiveScale") == 1.0
    assert window.main_layout.spacing() == 20
    assert window.main_layout.contentsMargins().left() == 40

    window.resize(800, 600)
    qtbot.waitUntil(lambda: window.size() == QSize(800, 600))
    qtbot.wait(20)
    assert window.property("responsiveScale") == DEFAULT_MINIMUM_SCALE
    assert window.main_layout.spacing() == 8
    assert window.main_layout.contentsMargins().left() == 14

    window.resize(1280, 800)
    qtbot.waitUntil(lambda: window.size() == QSize(1280, 800))
    qtbot.wait(20)
    assert window.property("responsiveScale") == 1.0
    assert window.main_layout.spacing() == 20
    assert window.main_layout.contentsMargins().left() == 40
    assert "padding: 10px 15px" in window.styleSheet()


@pytest.mark.gui
def test_main_window_preserves_startup_size_and_primary_actions_are_operable(
    qtbot, tmp_path, monkeypatch
):
    image_path = tmp_path / "dish.png"
    assert cv2.imwrite(str(image_path), np.full((16, 16, 3), 80, dtype=np.uint8))
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    assert window.size() == QSize(900, 500)
    window.resize(800, 600)
    qtbot.waitUntil(lambda: window.size() == QSize(800, 600))
    qtbot.wait(20)

    opened = []
    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (opened.append(True) or str(image_path), ""),
    )
    window.open_button.click()
    assert opened == [True]
    assert window.selected_file_path == str(image_path)
    assert window.path_input.text() == str(image_path)
    assert window.analyze_button.isEnabled()

    class _FakeAnalysisWindow:
        def __init__(self, path, parent):
            self.path = path
            self.parent = parent
            self.shown = False

        def show(self):
            self.shown = True

    monkeypatch.setattr("ui.main_window.AnalysisWindow", _FakeAnalysisWindow)
    window.analyze_button.click()
    assert window.analysis_window.path == str(image_path)
    assert window.analysis_window.shown

    class _FakeDialog:
        def __init__(self, _manager, _parent):
            self.exec_calls = 0
            dialogs.append(self)

        def exec(self):
            self.exec_calls += 1
            return int(QDialog.DialogCode.Rejected)

    class _FakeTestingWindow:
        def __init__(self, parent):
            self.parent = parent
            self.shown = False

        def show(self):
            self.shown = True

    dialogs = []
    monkeypatch.setattr("labeling.session_manager.SessionManager", lambda: object())
    monkeypatch.setattr("ui.main_window.LabelingSessionDialog", _FakeDialog)
    monkeypatch.setattr("ui.testing_window.TestingWindow", _FakeTestingWindow)
    window.label_button.click()
    window.testing_button.click()

    assert len(dialogs) == 1 and dialogs[0].exec_calls == 1
    assert window.label_button.isVisible()
    assert window.testing_button.isVisible()
    assert window.testing_window.parent is window
    assert window.testing_window.shown


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1280, 800), (800, 600), (640, 480)])
def test_analysis_controls_are_scroll_reachable_at_supported_sizes(
    qtbot, synthetic_analysis, size
):
    window = AnalysisWindow(str(synthetic_analysis))
    _show_at_size(qtbot, window, size)

    assert window.minimumSize().width() <= size[0]
    assert window.minimumSize().height() <= size[1]
    assert window.controls_scroll_area.verticalScrollBar().maximum() > 0
    controls = [
        window.view_mode_combo,
        window.show_petri_contour,
        window.show_area_overlay,
        window.algorithm_combo,
        window.spin_x,
        window.spin_y,
        window.spin_radius,
        window.btn_reset_geometry,
        window.slider_sens,
        window.slider_contrast,
        window.spin_margin,
        window.spin_min_size,
        window.chk_solid_fill,
        window.spin_fill_strength,
        window.btn_apply,
        window.text_results,
        window.btn_save,
        window.btn_close,
    ]
    actions = [window.btn_apply, window.btn_save, window.btn_close]
    _assert_no_overlap(actions, window.controls_widget)
    for control in controls:
        _scroll_into_view(qtbot, control, window.controls_scroll_area)
        _assert_inside(control, window.controls_scroll_area.viewport())
        assert control.isVisible()
    assert window.image_label.width() > 0 and window.image_label.height() > 0
    _assert_no_interactive_overlap(window.controls_widget)


@pytest.mark.gui
def test_analysis_image_expands_when_window_grows(qtbot, synthetic_analysis):
    window = AnalysisWindow(str(synthetic_analysis))
    _show_at_size(qtbot, window, (800, 600))
    compact_image_size = window.image_label.size()

    window.resize(1280, 800)
    qtbot.waitUntil(lambda: window.size() == QSize(1280, 800))
    qtbot.wait(20)

    assert window.image_label.width() > compact_image_size.width()
    assert window.image_label.height() > compact_image_size.height()


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1280, 800), (800, 600)])
def test_testing_window_keeps_controls_and_results_reachable(qtbot, size):
    window = TestingWindow()
    _show_at_size(qtbot, window, size)

    assert window.minimumSize().width() <= size[0]
    assert window.minimumSize().height() <= size[1]
    scroll = window.content_scroll
    assert scroll.verticalScrollBar().maximum() > 0
    window._fill_table(
        [
            {
                "name": "ClassicDefault",
                "metrics": {
                    "iou": 0.7,
                    "dice": 0.8,
                    "f1": 0.8,
                    "precision": 0.9,
                    "recall": 0.75,
                },
            }
        ]
    )
    window._fill_comparison(
        {"IoU": {"sample.png": {"ClassicDefault": "ClassicSolidFill"}}}
    )
    controls = [
        window.dataset_input,
        window.btn_browse_dataset,
        *window._algo_checkboxes.values(),
        window.btn_load_model,
        window.chk_per_snapshot,
        window.batch_size_input,
        window.sample_limit_input,
        window.chk_telemetry,
        window.cbx_compare_a,
        window.cbx_compare_b,
        window.btn_run,
        window.btn_export,
        window.table,
        window.comparison_table,
    ]
    assert window.table.rowCount() == 1
    assert window.comparison_table.isVisible()
    assert window.comparison_table.rowCount() == 1
    _assert_no_overlap([window.table, window.comparison_table], window.content_widget)
    for widget in controls:
        _scroll_into_view(qtbot, widget, scroll)
        _assert_inside(widget, scroll.viewport())
    _assert_no_overlap([window.btn_run, window.btn_export], window.content_widget)
    _assert_no_interactive_overlap(window.content_widget)


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1280, 800), (800, 600)])
def test_labeling_window_toolbar_and_image_fit_compact_sizes(
    qtbot, tmp_path, size, monkeypatch
):
    window = LabelingWindow(tmp_path / "label-session")
    _show_at_size(qtbot, window, size, restore=True)

    assert window.minimumSize().width() <= size[0]
    assert window.minimumSize().height() <= size[1]
    assert window.mode_combo.isVisible() and window.file_list.isVisible()
    assert window.file_panel.isVisible() and window.petri_frame.isVisible()
    for panel_control in (
        window.btn_refresh,
        window.btn_add_images,
        window.spin_cx,
        window.spin_cy,
        window.spin_radius,
        window.btn_auto_detect,
        window.btn_crop,
    ):
        assert panel_control.isVisible()
        _assert_inside(panel_control, window)
    actions = [
        window.btn_brush_minus,
        window.btn_brush_plus,
        window.btn_zoom_in,
        window.btn_zoom_out,
        window.btn_zoom_reset,
        window.btn_draw,
        window.btn_erase,
        window.btn_clear,
        window.btn_save,
        window.btn_export,
    ]
    for action in actions:
        assert action.isVisible()
        _assert_inside(action, window)
    _assert_no_overlap(actions, window)
    _assert_no_interactive_overlap(window)

    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    source_path = window.source_dir / "sample.png"
    assert cv2.imwrite(str(source_path), image)
    window.controller.detect_petri = lambda _image: PetriInfo(
        60, 60, 50, image.shape[:2]
    )
    window._load_file_list()
    item = window.file_list.item(0)
    assert item is not None
    window.file_list.setCurrentItem(item)
    window._on_file_selected(item)
    assert window.current_path == source_path
    assert window.paint_label._image.shape[:2] == image.shape[:2]

    qtbot.wait(20)
    initial_size = window.size()
    for _ in range(3):
        window._on_zoom_in()
    qtbot.wait(20)
    assert window.size() == initial_size
    assert (
        window.image_scroll_area.verticalScrollBar().maximum() > 0
        or window.image_scroll_area.horizontalScrollBar().maximum() > 0
    )

    window.paint_label.zoom_reset()
    qtbot.wait(20)
    center = QPoint(window.paint_label.width() // 2, window.paint_label.height() // 2)
    QTest.mouseClick(window.paint_label, Qt.MouseButton.LeftButton, pos=center)
    assert window.paint_label.mask[60, 60] == 255
    monkeypatch.setattr(
        "labeling.labeling_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    window.btn_erase.click()
    QTest.mouseClick(window.paint_label, Qt.MouseButton.LeftButton, pos=center)
    assert window.paint_label.mask[60, 60] == 0
    window.btn_draw.click()
    QTest.mouseClick(window.paint_label, Qt.MouseButton.LeftButton, pos=center)
    window.btn_clear.click()
    assert not window.paint_label.mask.any()
    window.btn_brush_plus.click()
    assert window.paint_label.brush_size == 22

    window.current_stem = "sample"
    saved_masks = []
    monkeypatch.setattr(
        "labeling.labeling_window.QMessageBox.information",
        lambda *_args, **_kwargs: saved_masks.append(True),
    )
    window.btn_save.click()
    saved_mask = window.masks_dir / "sample_mask.png"
    assert saved_mask.is_file()
    assert saved_masks

    archive_path = tmp_path / "session.zip"
    monkeypatch.setattr(
        "labeling.labeling_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(archive_path), ""),
    )
    window.btn_export.click()
    assert archive_path.is_file()


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1280, 800), (640, 480), (400, 300)])
def test_session_dialog_actions_remain_reachable_with_long_paths(qtbot, tmp_path, size):
    config_path = tmp_path / "config" / "settings.json"
    manager = SessionManager(config_path=config_path)
    long_root = tmp_path / ("long-storage-segment-" * 4) / "sessions"
    manager.set_root(long_root)
    session_dirs = []
    for index in range(6):
        session_dir = ensure_session_structure(tmp_path / f"recent-{index}")
        manager.add(session_dir, session_dir.name)
        session_dirs.append(session_dir)

    dialog = LabelingSessionDialog(manager)
    _show_at_size(qtbot, dialog, size)

    assert dialog.minimumSize().width() <= size[0]
    assert dialog.minimumSize().height() <= size[1]
    assert dialog.name_edit.isVisible()
    for action in (dialog.btn_cancel, dialog.create_button):
        assert action.isVisible()
        _assert_inside(action, dialog)
    _assert_no_overlap([dialog.btn_cancel, dialog.create_button], dialog)
    if size == (400, 300):
        assert dialog.content_scroll.verticalScrollBar().maximum() > 0

    for action in (dialog.btn_open_recent, dialog.btn_open_folder):
        _scroll_into_view(qtbot, action, dialog.content_scroll)
        _assert_inside(action, dialog.content_scroll.viewport())
    assert dialog.recent_list.count() == len(session_dirs)
    assert dialog.recent_list.height() >= 54
    assert dialog.root_label.toolTip() == str(long_root)
    _assert_no_interactive_overlap(dialog)

    dialog.name_edit.setText("compact-session")
    assert dialog.create_button.isEnabled()
    dialog.create_button.click()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.session_dir == long_root / "compact-session"


@pytest.mark.gui
def test_session_dialog_recent_session_can_be_selected_and_opened(qtbot, tmp_path):
    manager = SessionManager(config_path=tmp_path / "config.json")
    session_dir = ensure_session_structure(tmp_path / "recent-session")
    manager.add(session_dir, session_dir.name)
    dialog = LabelingSessionDialog(manager)
    _show_at_size(qtbot, dialog, (400, 300))

    _scroll_into_view(qtbot, dialog.recent_list, dialog.content_scroll)
    item = dialog.recent_list.item(0)
    item_rect = dialog.recent_list.visualItemRect(item)
    assert item_rect.isValid()
    QTest.mouseClick(
        dialog.recent_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=item_rect.center(),
    )
    assert dialog.recent_list.currentItem() is item

    _scroll_into_view(qtbot, dialog.btn_open_recent, dialog.content_scroll)
    dialog.btn_open_recent.click()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.session_dir == session_dir.resolve()
