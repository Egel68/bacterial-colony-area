import cv2
import numpy as np
import pytest

from analysis.colony_detector import ColonyDetector
from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams

DETECTOR = ColonyDetector()


class TestFilterComponents:
    def test_removes_small_components(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (7, 7), 255, -1)
        cv2.rectangle(mask, (50, 50), (69, 69), 255, -1)
        out = DETECTOR._filter_components(mask, min_size=10)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        assert num_labels - 1 == 1
        area = stats[1, cv2.CC_STAT_AREA]
        assert area == 400

    def test_keeps_all_when_min_size_one(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (9, 9), 255, -1)
        cv2.rectangle(mask, (50, 50), (69, 69), 255, -1)
        out = DETECTOR._filter_components(mask, min_size=1)
        assert np.array_equal(out, mask)


class TestCountColonies:
    def test_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert DETECTOR.count_colonies(mask) == 0

    def test_two_colonies(self, binary_mask_two_colonies):
        assert DETECTOR.count_colonies(binary_mask_two_colonies) == 2

    def test_one_colony(self, binary_mask_circle):
        assert DETECTOR.count_colonies(binary_mask_circle) == 1


class TestCreateInnerMask:
    def test_margin_reduces_radius(self, petri_info):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 80, 255, -1)
        inner = DETECTOR.create_inner_mask(mask, petri_info, margin_percent=10)
        contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        assert int(radius) == pytest.approx(72, abs=1)

    def test_zero_margin_clamped_to_one(self, petri_info):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 80, 255, -1)
        inner = DETECTOR.create_inner_mask(mask, petri_info, margin_percent=0)
        contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        assert int(radius) == pytest.approx(79, abs=1)


class TestDetectPetriDish:
    def test_returns_info_for_real_image(self, test_source_paths):
        if not test_source_paths:
            pytest.skip("no test images available")
        image = cv2.imread(str(test_source_paths[0]))
        if image is None:
            pytest.skip("could not load test image")
        petri_mask, info = DETECTOR.detect_petri_dish(image)
        assert info is not None
        assert info.radius > 50


class TestDetectColonies:
    def test_detects_colony_on_synthetic(self, synthetic_colony_image):
        h, w = synthetic_colony_image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        petri_info = PetriInfo(cx=center[0], cy=center[1], radius=radius, image_shape=(h, w))
        petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(petri_mask, center, radius, 255, -1)

        params = AnalysisParams(
            sensitivity=0.3, min_colony_size=10, margin_percent=5, contrast=1.0,
        )
        colony_mask, _ = DETECTOR.detect_colonies(
            synthetic_colony_image, petri_mask, params=params, petri_info=petri_info,
        )
        assert colony_mask.sum() > 0
        assert DETECTOR.count_colonies(colony_mask) >= 1
