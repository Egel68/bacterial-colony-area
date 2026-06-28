import cv2
import numpy as np

from analysis.colony_detector import ColonyDetector

from .interface import BaseDetectionAlgorithm
from .registry import register_algorithm


class _ClassicBase(BaseDetectionAlgorithm):
    name = ""
    description = ""
    params: dict = {}

    def __init__(self):
        self.detector = ColonyDetector()

    def detect(self, image: np.ndarray, is_cropped: bool = False) -> np.ndarray:
        if is_cropped:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            petri_info = {"center": center, "radius": radius, "area_px": int(3.14159 * radius * radius)}
            petri_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(petri_mask, center, radius, 255, -1)
        else:
            petri_mask, petri_info = self.detector.detect_petri_dish(image)
            if petri_info is None:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                radius = min(w, h) * 0.4
                petri_info = {"center": center, "radius": int(radius), "area_px": int(3.14159 * radius * radius)}
                petri_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(petri_mask, center, int(radius), 255, -1)

        colony_mask, _ = self.detector.detect_colonies(
            image,
            petri_mask,
            petri_info=petri_info,
            **self.params,
        )
        return colony_mask


@register_algorithm
class ClassicDefault(_ClassicBase):
    name = "Classic (default)"
    description = "Стандартные параметры: sensitivity=0.5, margin=10%, min_size=50, contrast=1.0"
    params = {
        "sensitivity": 0.5,
        "min_colony_size": 50,
        "edge_margin_percent": 10,
        "contrast_level": 1.0,
        "blur_size": 5,
        "use_solid_fill": False,
        "fill_strength": 15,
    }


@register_algorithm
class ClassicHighSensitivity(_ClassicBase):
    name = "Classic (high sensitivity)"
    description = "Повышенная чувствительность: sensitivity=0.7, margin=5%, min_size=30"
    params = {
        "sensitivity": 0.7,
        "min_colony_size": 30,
        "edge_margin_percent": 5,
        "contrast_level": 1.0,
        "blur_size": 5,
        "use_solid_fill": False,
        "fill_strength": 15,
    }


@register_algorithm
class ClassicSolidFill(_ClassicBase):
    name = "Classic (solid fill)"
    description = "Сплошная заливка для сливного роста: sensitivity=0.3, margin=15%, fill_strength=15"
    params = {
        "sensitivity": 0.3,
        "min_colony_size": 50,
        "edge_margin_percent": 15,
        "contrast_level": 1.0,
        "blur_size": 5,
        "use_solid_fill": True,
        "fill_strength": 15,
    }


@register_algorithm
class ClassicLowSensitivity(_ClassicBase):
    name = "Classic (low sensitivity)"
    description = "Пониженная чувствительность, крупные колонии: sensitivity=0.3, margin=10%, min_size=100"
    params = {
        "sensitivity": 0.3,
        "min_colony_size": 100,
        "edge_margin_percent": 10,
        "contrast_level": 1.0,
        "blur_size": 5,
        "use_solid_fill": False,
        "fill_strength": 15,
    }
