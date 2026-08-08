import numpy as np
import pytest

from analysis.image_processor import ImageProcessor

PROC = ImageProcessor()


class TestPreprocessImage:
    def test_shape_and_dtype(self, blank_image_bgr):
        out = PROC.preprocess_image(blank_image_bgr)
        assert out.shape == (200, 200, 3)
        assert out.dtype == np.uint8


class TestApplyClahe:
    def test_3ch_input_returns_2d(self, blank_image_bgr):
        out = PROC.apply_clahe(blank_image_bgr)
        assert out.ndim == 2
        assert out.shape == (200, 200)

    def test_2d_input_unchanged_shape(self, blank_image_gray):
        out = PROC.apply_clahe(blank_image_gray)
        assert out.ndim == 2
        assert out.shape == (200, 200)

    def test_values_in_range(self, blank_image_gray):
        out = PROC.apply_clahe(blank_image_gray)
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255


class TestToGrayscale:
    def test_3ch_to_2d(self, blank_image_bgr):
        out = PROC.to_grayscale(blank_image_bgr)
        assert out.ndim == 2
        assert out.shape == (200, 200)

    def test_2d_passthrough(self, blank_image_gray):
        out = PROC.to_grayscale(blank_image_gray)
        assert out is blank_image_gray


class TestExtractGreenChannel:
    def test_returns_green_channel(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 1] = 255
        green = PROC.extract_green_channel(bgr)
        assert np.all(green == 255)

    def test_2d_passthrough(self, blank_image_gray):
        out = PROC.extract_green_channel(blank_image_gray)
        assert out is blank_image_gray


class TestResizeImage:
    def test_downscale(self):
        img = np.ones((400, 200, 3), dtype=np.uint8)
        resized, scale = PROC.resize_image(img, max_dimension=200)
        assert resized.shape[:2] == (200, 100)
        assert scale == pytest.approx(0.5)

    def test_unchanged_when_smaller(self):
        img = np.ones((100, 200, 3), dtype=np.uint8)
        resized, scale = PROC.resize_image(img, max_dimension=1024)
        assert resized.shape[:2] == (100, 200)
        assert scale == pytest.approx(1.0)
