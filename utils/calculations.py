"""
Модуль для вычисления площадей и статистик.
"""

from typing import Optional

import cv2
import numpy as np

from analysis.geometry import PetriInfo
from analysis.results import AnalysisResult


class AreaCalculator:
    """
    Класс для вычисления площадей чашки Петри и колоний.
    """

    def __init__(self, petri_diameter_mm: float = 90.0):
        self.petri_diameter_mm = petri_diameter_mm

    def calculate_areas(
        self,
        petri_mask: np.ndarray,
        colony_mask: np.ndarray,
        petri_info: Optional[PetriInfo] = None,
        margin_percent: float = 0.0,
    ) -> AnalysisResult:

        colony_area_px = np.count_nonzero(colony_mask)

        if petri_info is not None:
            radius_full_px = petri_info.radius

            if margin_percent > 0:
                radius_inner_px = radius_full_px * (100 - margin_percent) / 100
                petri_area_px = np.pi * (radius_inner_px**2)
            else:
                petri_area_px = np.pi * (radius_full_px**2)

            real_radius_mm = self.petri_diameter_mm / 2
            petri_area_mm2 = np.pi * real_radius_mm**2
            px_to_mm2 = petri_area_mm2 / (np.pi * radius_full_px**2)

        else:
            petri_area_px = np.count_nonzero(petri_mask)
            petri_area_mm2 = 0
            px_to_mm2 = 0

        colony_area_mm2 = colony_area_px * px_to_mm2

        coverage_percent = (
            (colony_area_px / petri_area_px * 100) if petri_area_px > 0 else 0
        )

        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        colony_count = max(0, num_labels - 1)

        return AnalysisResult(
            colony_count=colony_count,
            colony_area_px=int(colony_area_px),
            colony_area_mm2=colony_area_mm2,
            petri_area_px=int(petri_area_px),
            petri_area_mm2=petri_area_mm2,
            coverage_percent=coverage_percent,
            px_to_mm2=px_to_mm2,
        )
