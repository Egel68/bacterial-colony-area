"""
Модуль для вычисления площадей и статистик.
"""

from typing import Dict, Optional

import cv2
import numpy as np


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
        petri_info: Optional[Dict] = None,
        margin_percent: float = 0.0,  # <--- НОВЫЙ АРГУМЕНТ
    ) -> Dict:
        """
        Вычисление площадей.
        Теперь учитывает отступ при расчете базовой площади.
        """

        # 1. Считаем площадь колоний (белые пиксели)
        colony_area_px = np.count_nonzero(colony_mask)

        # 2. Считаем правильную площадь чашки (Знаменатель)
        if petri_info and "radius" in petri_info:
            radius_full_px = petri_info["radius"]

            # Если задан отступ, считаем площадь только внутренней "рабочей" зоны
            if margin_percent > 0:
                # Радиус рабочей зоны (где реально искали бактерии)
                radius_inner_px = radius_full_px * (100 - margin_percent) / 100
                petri_area_px = np.pi * (radius_inner_px**2)
            else:
                petri_area_px = np.pi * (radius_full_px**2)

            # Коэффициент перевода в мм
            real_radius_mm = self.petri_diameter_mm / 2
            petri_area_mm2 = np.pi * real_radius_mm**2
            px_to_mm2 = petri_area_mm2 / (np.pi * radius_full_px**2)

        else:
            # Fallback, если нет инфо о радиусе (маловероятно)
            petri_area_px = np.count_nonzero(petri_mask)
            petri_area_mm2 = 0
            px_to_mm2 = 0

        # 3. Основные расчеты
        colony_area_mm2 = colony_area_px * px_to_mm2

        # Процент покрытия относительно РАБОЧЕЙ зоны
        coverage_percent = (
            (colony_area_px / petri_area_px * 100) if petri_area_px > 0 else 0
        )

        # Количество колоний
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        colony_count = max(0, num_labels - 1)

        return {
            "petri_area_px": int(petri_area_px),
            "petri_area_mm2": petri_area_mm2,
            "colony_area_px": colony_area_px,
            "colony_area_mm2": colony_area_mm2,
            "coverage_percent": coverage_percent,
            "colony_count": colony_count,
            "px_to_mm2": px_to_mm2,
        }
