from typing import Dict, Optional

import cv2
import numpy as np

from analysis.colony_detector import ColonyDetector
from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams
from analysis.results import AnalysisResult
from utils.calculations import AreaCalculator
from testing.interface import BaseDetectionAlgorithm


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

    def analyze_with_algorithm(
        self,
        image: np.ndarray,
        petri_mask: np.ndarray,
        algorithm: BaseDetectionAlgorithm,
        petri_info: Optional[PetriInfo] = None,
        margin_percent: float = 0.0,
        is_cropped: bool = False,
        progress_callback=None,
    ) -> AnalysisResult:
        """Выполнить алгоритм и опубликовать маску в состоянии контроллера."""
        result, mask = self.calculate_algorithm_result(
            image,
            petri_mask,
            algorithm,
            petri_info=petri_info,
            margin_percent=margin_percent,
            is_cropped=is_cropped,
            progress_callback=progress_callback,
        )
        self.set_algorithm_mask(mask)
        return result

    def calculate_algorithm_result(
        self,
        image: np.ndarray,
        petri_mask: np.ndarray,
        algorithm: BaseDetectionAlgorithm,
        petri_info: Optional[PetriInfo] = None,
        margin_percent: float = 0.0,
        is_cropped: bool = False,
        progress_callback=None,
    ) -> tuple[AnalysisResult, np.ndarray]:
        """Вычислить результат без записи в состояние контроллера.

        Чистый метод подходит для фонового worker: GUI публикует готовую маску
        только после получения сигнала завершения в главном потоке.
        """
        detect_with_progress = getattr(algorithm, "detect_with_progress", None)
        if detect_with_progress is not None and progress_callback is not None:
            mask = detect_with_progress(
                image,
                is_cropped=is_cropped,
                progress_callback=progress_callback,
            )
        else:
            mask = algorithm.detect(image, is_cropped=is_cropped)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Algorithm '{algorithm.name}' returned mask shape {mask.shape}, "
                f"expected {image.shape[:2]}"
            )
        if petri_mask.shape != image.shape[:2]:
            raise ValueError(
                f"Petri mask shape {petri_mask.shape} does not match image "
                f"shape {image.shape[:2]}"
            )
        colony_mask = (mask > 0).astype(np.uint8) * 255
        working_mask = petri_mask > 0
        effective_margin = margin_percent
        if not 0 <= margin_percent < 100:
            raise ValueError("margin_percent must be between 0 and 100")
        if petri_info is not None:
            h, w = image.shape[:2]
            inner_dish_mask = np.zeros((h, w), dtype=np.uint8)
            effective_margin = max(margin_percent, 1.0)
            inner_radius = int(petri_info.radius * (100 - effective_margin) / 100)
            cv2.circle(
                inner_dish_mask,
                petri_info.center,
                inner_radius,
                255,
                -1,
            )
            working_mask &= inner_dish_mask > 0
        colony_mask[~working_mask] = 0
        result = self._calculator.calculate_areas(
            petri_mask,
            colony_mask,
            petri_info,
            margin_percent=effective_margin,
        )
        return result, colony_mask

    def set_algorithm_mask(self, mask: np.ndarray) -> None:
        """Сохранить завершённую маску и обновить отладочное представление."""
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D colony mask, got shape {mask.shape}")
        self._colony_mask = (mask > 0).astype(np.uint8) * 255
        self._debug_images = {"binary": self._colony_mask}

    @property
    def colony_mask(self) -> Optional[np.ndarray]:
        return self._colony_mask

    @property
    def debug_images(self) -> Dict:
        return self._debug_images
