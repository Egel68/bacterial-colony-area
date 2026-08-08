import cv2
import numpy as np
import pytest

from utils.image_loader import SUPPORTED_EXTENSIONS, load_image, load_image_grayscale


@pytest.fixture
def padded_width_png(tmp_path):
    """PNG с шириной 870px: bytesPerLine=2612 > 3×870=2610 (Qt row-padding)."""
    image = np.random.default_rng(42).integers(
        0, 256, (40, 870, 3), dtype=np.uint8
    )
    path = tmp_path / "padded.png"
    cv2.imwrite(str(path), image)
    return path, image


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
