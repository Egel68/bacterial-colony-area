import pytest

from ui.testing_window import TestingWindow, _RunWorker

pytest.importorskip("pytestqt")


@pytest.fixture
def window(qtbot):
    win = TestingWindow()
    qtbot.addWidget(win)
    win.show()
    return win


def test_window_opens_with_algorithms(qtbot, window):
    assert window.isVisible()
    assert len(window._algo_checkboxes) >= 4
    names = list(window._algo_checkboxes)
    assert "ClassicDefault" in names


def test_results_table_header_stretches(qtbot, window):
    from PyQt6.QtWidgets import QHeaderView

    assert (
        window.table.horizontalHeader().sectionResizeMode(0)
        == QHeaderView.ResizeMode.Stretch
    )
    assert (
        window.comparison_table.horizontalHeader().sectionResizeMode(0)
        == QHeaderView.ResizeMode.Stretch
    )


def test_dataset_hint_mentions_directories(qtbot, window):
    from PyQt6.QtWidgets import QLabel

    hints = []
    for label in window.findChildren(QLabel):
        if "source/" in label.text():
            hints.append(label.text())
    assert hints
    text = hints[0]
    for part in ("source/", "masks/", "cropped/", "cropped_masks/"):
        assert part in text


def test_missing_dataset_dir_shows_warning(qtbot, window, tmp_path, monkeypatch):
    window.dataset_input.setText(str(tmp_path / "nope"))
    warnings = []
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.warning", lambda *a, **k: warnings.append(a)
    )
    window._start_run()
    assert warnings
    assert window._thread is None


def test_worker_without_samples_fails(qtbot, tmp_path):
    worker = _RunWorker(str(tmp_path), ["ClassicDefault"])
    failed = []
    worker.failed.connect(lambda msg: failed.append(msg))
    worker.run()
    assert failed
    assert "No test samples found" in failed[0]
