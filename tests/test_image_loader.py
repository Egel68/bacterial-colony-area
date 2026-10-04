import threading

import cv2
import numpy as np
import pytest

from utils.image_loader import (
    SUPPORTED_EXTENSIONS,
    load_image,
    load_image_grayscale,
    load_image_grayscale_worker_safe,
    load_image_worker_safe,
)


@pytest.fixture
def padded_width_png(tmp_path):
    """PNG с шириной 870px: bytesPerLine=2612 > 3×870=2610 (Qt row-padding)."""
    image = np.random.default_rng(42).integers(0, 256, (40, 870, 3), dtype=np.uint8)
    path = tmp_path / "padded.png"
    cv2.imwrite(str(path), image)
    return path, image


@pytest.fixture
def pattern_bgr():
    """Детерминированный тестовый кадр с разными цветами каналов (BGR-контроль)."""
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)  # синий
    image[:, 16:32] = (0, 255, 0)  # зелёный
    image[:, 32:] = (0, 0, 255)  # красный
    return image


def _forbid_qpixmap(monkeypatch):
    """Любое обращение к QPixmap во время теста — ошибка."""
    class _Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("QPixmap вызван вне GUI-потока")

    monkeypatch.setattr("utils.image_loader.QPixmap", _Forbidden)


@pytest.mark.gui
class TestLoadImage:
    def test_load_color(self, test_source_paths, qtbot):
        if not test_source_paths:
            pytest.skip("no test images available")
        img = load_image(str(test_source_paths[0]))
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_load_grayscale(self, test_source_paths, qtbot):
        if not test_source_paths:
            pytest.skip("no test images available")
        img = load_image_grayscale(str(test_source_paths[0]))
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_bgr_order(self, test_source_paths, qtbot):
        if not test_source_paths:
            pytest.skip("no test images available")
        img = load_image(str(test_source_paths[0]))
        assert img.shape[2] == 3

    def test_nonexistent_path(self):
        with pytest.raises(ValueError, match="загрузить"):
            load_image("/nonexistent/file.png")


@pytest.mark.gui
class TestLoadImageGrayscale:
    def test_grayscale_shape(self, test_source_paths, qtbot):
        if not test_source_paths:
            pytest.skip("no test images available")
        img = load_image_grayscale(str(test_source_paths[0]))
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_nonexistent_path_grayscale(self):
        with pytest.raises(ValueError, match="загрузить"):
            load_image_grayscale("/nonexistent/file.png")


@pytest.mark.gui
class TestLoadImageRowPadding:
    def test_padded_width_matches_opencv(self, padded_width_png, qtbot):
        path, expected = padded_width_png
        img = load_image(str(path))
        assert img.shape == expected.shape
        assert img.dtype == np.uint8
        assert np.array_equal(img, expected)

    def test_padded_width_matches_grayscale(self, padded_width_png, qtbot):
        path, expected = padded_width_png
        img = load_image_grayscale(str(path))
        assert img.shape == expected.shape[:2]
        assert img.dtype == np.uint8
        assert img.min() >= 0 and img.max() <= 255


@pytest.mark.gui
class TestLoadImageErrorMessage:
    def test_message_mentions_both_loaders(self):
        with pytest.raises(ValueError) as excinfo:
            load_image("/nonexistent/file.png")
        msg = str(excinfo.value)
        assert "/nonexistent/file.png" in msg
        assert "QPixmap" in msg and "OpenCV" in msg

    def test_grayscale_message_mentions_both_loaders(self):
        with pytest.raises(ValueError) as excinfo:
            load_image_grayscale("/nonexistent/file.png")
        msg = str(excinfo.value)
        assert "QPixmap" in msg and "OpenCV" in msg


class TestSupportedExtensions:
    def test_contains_common_formats(self):
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS
        assert ".gif" not in SUPPORTED_EXTENSIONS
        assert ".svg" not in SUPPORTED_EXTENSIONS


class TestWorkerSafeDecoder:
    """Задача 4.1: потокобезопасный декодер без QPixmap (QImage -> OpenCV)."""

    @pytest.mark.parametrize("ext", ["png", "jpg", "bmp", "tif", "webp"])
    def test_decodes_supported_formats_without_qpixmap(
        self, tmp_path, pattern_bgr, monkeypatch, ext
    ):
        path = tmp_path / f"img.{ext}"
        assert cv2.imwrite(str(path), pattern_bgr)
        _forbid_qpixmap(monkeypatch)
        img = load_image_worker_safe(str(path))
        ref = cv2.imread(str(path))
        assert img.shape == ref.shape
        assert img.dtype == np.uint8
        if ext in ("png", "bmp", "tif"):
            assert np.array_equal(img, ref), f"{ext}: BGR-контракт нарушен"
        else:
            # lossy-форматы: декодеры могут чуть различаться
            assert np.abs(img.astype(int) - ref.astype(int)).max() <= 12

    @pytest.mark.parametrize("ext", ["png", "jpg", "bmp", "tif", "webp"])
    def test_grayscale_decodes_supported_formats(
        self, tmp_path, pattern_bgr, monkeypatch, ext
    ):
        path = tmp_path / f"gray.{ext}"
        assert cv2.imwrite(str(path), pattern_bgr)
        _forbid_qpixmap(monkeypatch)
        gray = load_image_grayscale_worker_safe(str(path))
        ref = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert gray.shape == ref.shape
        assert gray.dtype == np.uint8

    def test_runs_in_background_thread_without_qpixmap(
        self, tmp_path, pattern_bgr, monkeypatch
    ):
        path = tmp_path / "thread.png"
        assert cv2.imwrite(str(path), pattern_bgr)
        _forbid_qpixmap(monkeypatch)
        results = []
        errors = []

        def worker():
            try:
                results.append(load_image_worker_safe(str(path)))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
        assert results and np.array_equal(results[0], pattern_bgr)

    def test_padded_width_matches_opencv(self, padded_width_png, monkeypatch):
        path, expected = padded_width_png
        _forbid_qpixmap(monkeypatch)
        img = load_image_worker_safe(str(path))
        assert img.shape == expected.shape
        assert np.array_equal(img, expected)

    def test_padded_width_grayscale(self, padded_width_png, monkeypatch):
        path, expected = padded_width_png
        _forbid_qpixmap(monkeypatch)
        gray = load_image_grayscale_worker_safe(str(path))
        assert gray.shape == expected.shape[:2]
        assert gray.dtype == np.uint8

    def test_falls_back_to_opencv_when_qt_decoder_fails(
        self, tmp_path, pattern_bgr, monkeypatch
    ):
        path = tmp_path / "fallback.png"
        assert cv2.imwrite(str(path), pattern_bgr)
        monkeypatch.setattr("utils.image_loader._load_image_qimage", lambda _p: None)
        monkeypatch.setattr(
            "utils.image_loader._load_image_grayscale_qimage", lambda _p: None
        )
        img = load_image_worker_safe(str(path))
        assert np.array_equal(img, pattern_bgr)
        gray = load_image_grayscale_worker_safe(str(path))
        assert gray.ndim == 2 and gray.dtype == np.uint8

    def test_both_paths_fail_raises_descriptive_error(self, monkeypatch):
        monkeypatch.setattr("utils.image_loader._load_image_qimage", lambda _p: None)
        monkeypatch.setattr(
            "utils.image_loader._load_image_grayscale_qimage", lambda _p: None
        )
        with pytest.raises(ValueError) as excinfo:
            load_image_worker_safe("/nonexistent/file.png")
        msg = str(excinfo.value)
        assert "/nonexistent/file.png" in msg
        assert "QImage" in msg and "OpenCV" in msg

        with pytest.raises(ValueError) as excinfo:
            load_image_grayscale_worker_safe("/nonexistent/file.png")
        msg = str(excinfo.value)
        assert "QImage" in msg and "OpenCV" in msg


class TestGuiThreadLoadingBehaviorPreserved:
    """Существующий порядок QPixmap -> OpenCV для GUI-потока сохранён."""

    def test_gui_path_prefers_qpixmap_result(self, tmp_path, pattern_bgr, monkeypatch):
        path = tmp_path / "gui.png"
        assert cv2.imwrite(str(path), pattern_bgr)
        sentinel = np.full((3, 3, 3), 7, dtype=np.uint8)
        monkeypatch.setattr(
            "utils.image_loader._load_image_pixmap", lambda _p: sentinel
        )
        monkeypatch.setattr(
            "utils.image_loader._load_image_opencv",
            lambda _p: (_ for _ in ()).throw(AssertionError("OpenCV не должен вызываться")),
        )
        img = load_image(str(path))
        assert np.array_equal(img, sentinel)

    def test_gui_path_falls_back_to_opencv(self, tmp_path, pattern_bgr, monkeypatch):
        path = tmp_path / "gui_fallback.png"
        assert cv2.imwrite(str(path), pattern_bgr)
        monkeypatch.setattr("utils.image_loader._load_image_pixmap", lambda _p: None)
        img = load_image(str(path))
        assert np.array_equal(img, pattern_bgr)
