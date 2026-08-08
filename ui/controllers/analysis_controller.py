from typing import Dict, Optional

import numpy as np

from analysis.colony_detector import ColonyDetector
from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams
from analysis.results import AnalysisResult
from utils.calculations import AreaCalculator


class AnalysisController:
    """Бизнес-логика анализа без зависимостей от Qt.

    Принимает/возвращает только ndarray и dataclass'ы.
    """

    def __init__(
        self,
        detector: Optional[ColonyDetector] = None,
        calculator: Optional[AreaCalculator] = None,
    ):
        self._detector = detector or ColonyDetector()
        self._calculator = calculator or AreaCalculator()
        self._colony_mask: Optional[np.ndarray] = None
        self._debug_images: Dict = {}

    def find_petri_dish(
        self,
        image: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[PetriInfo]]:
        return self._detector.detect_petri_dish(image)

    def analyze(
        self,
        image: np.ndarray,
        petri_mask: np.ndarray,
        params: AnalysisParams,
        petri_info: Optional[PetriInfo] = None,
        blur_size: int = 5,
    ) -> AnalysisResult:
        self._colony_mask, self._debug_images = self._detector.detect_colonies(
            image,
            petri_mask,
            params=params,
            petri_info=petri_info,
            blur_size=blur_size,
        )
        return self._calculator.calculate_areas(
            petri_mask,
            self._colony_mask,
            petri_info,
            margin_percent=params.margin_percent,
        )

    @property
    def colony_mask(self) -> Optional[np.ndarray]:
        return self._colony_mask

    @property
    def debug_images(self) -> Dict:
        return self._debug_images
