import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.calculations import AreaCalculator


def make_calculator(diameter=90.0):
    return AreaCalculator(petri_diameter_mm=diameter)


class TestCalculateBasic:
    def test_colony_count_one(self, binary_mask_circle, petri_mask, petri_info):
        calc = make_calculator()
        res = calc.calculate_areas(petri_mask, binary_mask_circle, petri_info)
        assert res["colony_count"] == 1
        assert res["colony_area_px"] > 0
        assert res["coverage_percent"] > 0

    def test_empty_colonies(self, petri_mask, petri_info):
        calc = make_calculator()
        empty = np.zeros_like(petri_mask)
        res = calc.calculate_areas(petri_mask, empty, petri_info)
        assert res["colony_count"] == 0
        assert res["colony_area_px"] == 0
        assert res["coverage_percent"] == 0.0

    def test_two_colonies(self, binary_mask_two_colonies, petri_mask, petri_info):
        calc = make_calculator()
        res = calc.calculate_areas(petri_mask, binary_mask_two_colonies, petri_info)
        assert res["colony_count"] == 2

    def test_colony_area_px_exact(self, petri_mask, petri_info):
        calc = make_calculator()
        square = np.zeros((200, 200), dtype=np.uint8)
        square[50:60, 50:60] = 255
        res = calc.calculate_areas(petri_mask, square, petri_info)
        assert res["colony_area_px"] == 100


class TestMargin:
    def test_margin_reduces_working_area(self, binary_mask_circle, petri_mask, petri_info):
        calc = make_calculator()
        res_no_margin = calc.calculate_areas(petri_mask, binary_mask_circle, petri_info, margin_percent=0)
        res_with_margin = calc.calculate_areas(petri_mask, binary_mask_circle, petri_info, margin_percent=10)
        assert res_with_margin["petri_area_px"] < res_no_margin["petri_area_px"]
        assert res_with_margin["coverage_percent"] > res_no_margin["coverage_percent"]

    def test_margin_zero_same_as_default(self, binary_mask_circle, petri_mask, petri_info):
        calc = make_calculator()
        res_default = calc.calculate_areas(petri_mask, binary_mask_circle, petri_info)
        res_explicit = calc.calculate_areas(petri_mask, binary_mask_circle, petri_info, margin_percent=0)
        assert res_default["petri_area_px"] == res_explicit["petri_area_px"]


class TestPetriAreaMm2:
    def test_with_standard_diameter(self, petri_mask, petri_info):
        calc = make_calculator(diameter=90.0)
        res = calc.calculate_areas(petri_mask, petri_mask, petri_info)
        expected = np.pi * (45.0 ** 2)
        assert res["petri_area_mm2"] == pytest.approx(expected, rel=1e-3)

    def test_with_custom_diameter(self, petri_mask, petri_info):
        calc = make_calculator(diameter=100.0)
        res = calc.calculate_areas(petri_mask, petri_mask, petri_info)
        expected = np.pi * (50.0 ** 2)
        assert res["petri_area_mm2"] == pytest.approx(expected, rel=1e-3)


class TestFallbackNoPetriInfo:
    def test_without_petri_info(self, petri_mask, binary_mask_circle):
        calc = make_calculator()
        res = calc.calculate_areas(petri_mask, binary_mask_circle, petri_info=None)
        assert res["colony_area_px"] > 0
        assert res["colony_count"] >= 0


@given(
    st.integers(1, 80),
    st.integers(1, 80),
)
@settings(max_examples=30)
def test_coverage_bounds_property(col_w, col_h):
    calc = make_calculator()
    petri = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(petri, (100, 100), 80, 255, -1)
    colony = np.zeros_like(petri)
    x1 = 100 - col_w // 2
    y1 = 100 - col_h // 2
    colony[max(0, y1):min(200, y1 + col_h), max(0, x1):min(200, x1 + col_w)] = 255
    colony_in_petri = cv2.bitwise_and(colony, petri)
    info = {"center": (100, 100), "radius": 80, "area_px": int(np.pi * 6400)}
    res = calc.calculate_areas(petri, colony_in_petri, info)
    assert 0 <= res["coverage_percent"] <= 100
    assert res["colony_area_px"] >= 0
    assert res["petri_area_px"] >= 0



