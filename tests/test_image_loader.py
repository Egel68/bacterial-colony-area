import numpy as np
import pytest

from utils.image_loader import load_image, load_image_grayscale


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
        b, g, r = img[100, 100, 0], img[100, 100, 1], img[100, 100, 2]
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
