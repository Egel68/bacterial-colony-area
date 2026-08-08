import cv2
import numpy as np
import pytest

from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams
from analysis.results import AnalysisResult
from ui.controllers.analysis_controller import AnalysisController

CONTROLLER = AnalysisController()


class TestAnalysisController:
    def test_find_petri_dish_returns_info(self, test_source_paths):
        if not test_source_paths:
            pytest.skip("no test images available")
        image = cv2.imread(str(test_source_paths[0]))
        if image is None:
            pytest.skip("could not load test image")
        mask, info = CONTROLLER.find_petri_dish(image)
        assert info is not None
        assert mask is not None
        assert mask.shape == image.shape[:2]

    def test_find_petri_dish_blank(self, blank_image_bgr):
        mask, info = CONTROLLER.find_petri_dish(blank_image_bgr)
        assert mask is None
        assert info is None

    def test_analyze_returns_typed_result(self, synthetic_colony_image):
        h, w = synthetic_colony_image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        petri_info = PetriInfo(
            cx=center[0], cy=center[1], radius=radius, image_shape=(h, w)
        )
        petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(petri_mask, center, radius, 255, -1)
        params = AnalysisParams(
            sensitivity=0.3, contrast=1.0, margin_percent=5, min_colony_size=10
        )
        result = CONTROLLER.analyze(
            synthetic_colony_image, petri_mask, params=params, petri_info=petri_info
        )
        assert isinstance(result, AnalysisResult)
        assert result.colony_count > 0
        assert result.colony_area_px > 0
        assert result.coverage_percent > 0
        assert CONTROLLER.colony_mask is not None
        assert "binary" in CONTROLLER.debug_images
