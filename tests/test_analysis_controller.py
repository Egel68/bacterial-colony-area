import cv2
import numpy as np
import pytest

from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams
from analysis.results import AnalysisResult
from analysis.colony_detector import ColonyDetector
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

    def test_analyze_with_selected_algorithm_uses_model_mask_for_areas(
        self, blank_image_bgr
    ):
        class _FixedAlgorithm:
            name = "fake"
            description = "test"

            def detect(self, image, is_cropped=False):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (w // 2 + 8, h // 2), 12, 255, -1)
                return mask

        h, w = blank_image_bgr.shape[:2]
        petri_info = PetriInfo(cx=w // 2, cy=h // 2, radius=40, image_shape=(h, w))
        petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(petri_mask, petri_info.center, petri_info.radius, 255, -1)
        controller = AnalysisController(detector=ColonyDetector())

        result = controller.analyze_with_algorithm(
            blank_image_bgr,
            petri_mask,
            _FixedAlgorithm(),
            petri_info=petri_info,
            margin_percent=0,
        )

        assert controller.colony_mask is not None
        assert result.colony_area_px == int(np.count_nonzero(controller.colony_mask))
        assert result.colony_count == 1
        assert controller.debug_images == {"binary": controller.colony_mask}

    def test_algorithm_mask_is_clipped_to_petri_working_area_without_state_mutation(
        self, blank_image_bgr
    ):
        class _TwoColonies:
            name = "fake-two-colonies"
            description = "test"

            def detect(self, image, is_cropped=False):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (100, 100), 8, 255, -1)
                cv2.circle(mask, (50, 50), 8, 255, -1)
                return mask

        h, w = blank_image_bgr.shape[:2]
        petri_info = PetriInfo(cx=100, cy=100, radius=70, image_shape=(h, w))
        petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(petri_mask, petri_info.center, petri_info.radius, 255, -1)
        controller = AnalysisController()

        result, mask = controller.calculate_algorithm_result(
            blank_image_bgr,
            petri_mask,
            _TwoColonies(),
            petri_info=petri_info,
            margin_percent=25,
        )

        assert controller.colony_mask is None
        assert result.colony_count == 1
        assert mask[50, 50] == 0
        assert mask[100, 100] == 255
        assert np.count_nonzero(mask[petri_mask == 0]) == 0

    def test_calculating_algorithm_result_does_not_publish_partial_mask(
        self, blank_image_bgr
    ):
        controller = AnalysisController()
        original_mask = np.full(blank_image_bgr.shape[:2], 255, dtype=np.uint8)
        controller.set_algorithm_mask(original_mask)

        class _FailingAlgorithm:
            name = "failure"
            description = "test"

            def detect(self, image, is_cropped=False):
                raise RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            controller.calculate_algorithm_result(
                blank_image_bgr,
                np.full(blank_image_bgr.shape[:2], 255, dtype=np.uint8),
                _FailingAlgorithm(),
            )

        assert np.array_equal(controller.colony_mask, original_mask)

    def test_algorithm_margin_uses_classic_minimum_one_percent(self, blank_image_bgr):
        class _CenterPixel:
            name = "center"
            description = "test"

            def detect(self, image, is_cropped=False):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[100, 170] = 255
                return mask

        petri_info = PetriInfo(100, 100, 70, blank_image_bgr.shape[:2])
        petri_mask = np.zeros(blank_image_bgr.shape[:2], dtype=np.uint8)
        cv2.circle(petri_mask, petri_info.center, petri_info.radius, 255, -1)
        controller = AnalysisController()

        result = controller.analyze_with_algorithm(
            blank_image_bgr,
            petri_mask,
            _CenterPixel(),
            petri_info=petri_info,
            margin_percent=0,
        )

        assert result.colony_area_px == 0

    def test_classic_analyze_path_does_not_invoke_selected_algorithm(
        self, blank_image_bgr, monkeypatch
    ):
        controller = AnalysisController()
        h, w = blank_image_bgr.shape[:2]
        petri_info = PetriInfo(cx=w // 2, cy=h // 2, radius=40, image_shape=(h, w))
        petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(petri_mask, petri_info.center, petri_info.radius, 255, -1)
        calls = []

        class _Algorithm:
            def detect(self, *args, **kwargs):
                calls.append(True)
                return np.zeros((h, w), dtype=np.uint8)

        controller.analyze(
            blank_image_bgr,
            petri_mask,
            AnalysisParams(),
            petri_info=petri_info,
        )
        assert calls == []
