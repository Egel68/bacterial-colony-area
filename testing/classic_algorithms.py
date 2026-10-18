import cv2
import numpy as np

from analysis.colony_detector import ColonyDetector
from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams

from .interface import BaseDetectionAlgorithm
from .registry import register_algorithm


class _ClassicBase(BaseDetectionAlgorithm):
    name = ""
    description = ""
    params: AnalysisParams | None = None

    def __init__(self):
        self.detector = ColonyDetector()

    def detect(
        self,
        image: np.ndarray,
        is_cropped: bool = False,
        context=None,
    ) -> np.ndarray:
        h, w = image.shape[:2]

        # Геометрия чашки инвариантна для одного снимка/варианта: при
        # совместимом контексте переиспользуется (задача 3.3).
        geometry = context.get_geometry() if context is not None else None
        if geometry is not None:
            petri_mask, petri_info = geometry
        elif is_cropped:
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            petri_info = PetriInfo(
                cx=center[0],
                cy=center[1],
                radius=radius,
                image_shape=(h, w),
            )
            petri_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(petri_mask, center, radius, 255, -1)
            if context is not None:
                context.set_geometry(petri_mask, petri_info)
        else:
            petri_mask, petri_info = self.detector.detect_petri_dish(image)
            if petri_info is None:
                center = (w // 2, h // 2)
                radius = int(min(w, h) * 0.4)
                petri_info = PetriInfo(
                    cx=center[0],
                    cy=center[1],
                    radius=radius,
                    image_shape=(h, w),
                )
                petri_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(petri_mask, center, radius, 255, -1)
            if context is not None:
                context.set_geometry(petri_mask, petri_info)

        colony_mask, _ = self.detector.detect_colonies(
            image,
            petri_mask,
            params=self.params,
            petri_info=petri_info,
            context=context,
        )
        return colony_mask


@register_algorithm
class ClassicDefault(_ClassicBase):
    name = "Classic (default)"
    description = (
        "Стандартные параметры: sensitivity=0.5, margin=10%, min_size=50, contrast=1.0"
    )
    params = AnalysisParams(
        sensitivity=0.5,
        min_colony_size=50,
        margin_percent=10,
        contrast=1.0,
        solid_fill=False,
        fill_strength=15,
    )


@register_algorithm
class ClassicHighSensitivity(_ClassicBase):
    name = "Classic (high sensitivity)"
    description = "Повышенная чувствительность: sensitivity=0.7, margin=5%, min_size=30"
    params = AnalysisParams(
        sensitivity=0.7,
        min_colony_size=30,
        margin_percent=5,
        contrast=1.0,
        solid_fill=False,
        fill_strength=15,
    )


@register_algorithm
class ClassicSolidFill(_ClassicBase):
    name = "Classic (solid fill)"
    description = "Сплошная заливка для сливного роста: sensitivity=0.3, margin=15%, fill_strength=15"
    params = AnalysisParams(
        sensitivity=0.3,
        min_colony_size=50,
        margin_percent=15,
        contrast=1.0,
        solid_fill=True,
        fill_strength=15,
    )


@register_algorithm
class ClassicLowSensitivity(_ClassicBase):
    name = "Classic (low sensitivity)"
    description = "Пониженная чувствительность, крупные колонии: sensitivity=0.3, margin=10%, min_size=100"
    params = AnalysisParams(
        sensitivity=0.3,
        min_colony_size=100,
        margin_percent=10,
        contrast=1.0,
        solid_fill=False,
        fill_strength=15,
    )
