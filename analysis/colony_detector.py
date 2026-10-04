"""
Модуль для обнаружения чашки Петри и бактериальных колоний.
Оптимизирован для темного поля (Dark Field) и изображений с бликами.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import PetriInfo
from .image_processor import ImageProcessor
from .params import AnalysisParams
from .profiling import measure_stage
from .sample_context import SampleContext, SampleContextCache


class ColonyDetector:
    """
    Класс для обнаружения чашки Петри и бактериальных колоний.
    """

    def __init__(self):
        self.processor = ImageProcessor()

    def _find_contours(self, mask: np.ndarray) -> List:
        """Обёртка для cv2.findContours."""
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]
        return contours

    def detect_petri_dish(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[PetriInfo]]:
        """
        Обнаружение чашки Петри.
        Использует тот факт, что края чашки на этих фото - самые яркие объекты (блики).
        """
        with measure_stage("dish_search"):
            return self._detect_petri_dish_impl(image)

    def _detect_petri_dish_impl(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[PetriInfo]]:
        gray = self.processor.to_grayscale(image)
        h, w = gray.shape

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        _, bright_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(bright_mask) < (h * w * 0.01):
            bright_mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                101,
                -10,
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        closed_mask = cv2.morphologyEx(
            bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        contours = self._find_contours(closed_mask)

        best_circle = None
        max_area = 0
        center_image = (w // 2, h // 2)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (h * w * 0.1):
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            dist_from_center = np.sqrt(
                (x - center_image[0]) ** 2 + (y - center_image[1]) ** 2
            )
            if dist_from_center > min(h, w) * 0.3:
                continue

            if area > max_area:
                max_area = area
                best_circle = (center, radius)

        if best_circle:
            center, radius = best_circle
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            return mask, PetriInfo(
                cx=center[0],
                cy=center[1],
                radius=radius,
                image_shape=(h, w),
            )

        return self._detect_petri_with_hough(image)

    def _detect_petri_with_hough(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[PetriInfo]]:
        gray = self.processor.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        min_dim = min(image.shape[:2])
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim / 2,
            param1=100,
            param2=30,
            minRadius=int(min_dim * 0.3),
            maxRadius=int(min_dim * 0.48),
        )
        if circles is not None:
            circle = circles[0][0]
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            h, w = image.shape[:2]
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            return mask, PetriInfo(
                cx=center[0],
                cy=center[1],
                radius=radius,
                image_shape=(h, w),
            )
        return None, None

    def create_inner_mask(
        self, petri_mask: np.ndarray, petri_info: PetriInfo, margin_percent: float = 5
    ) -> np.ndarray:
        real_margin = max(margin_percent, 1.0)
        inner_radius = int(petri_info.radius * (100 - real_margin) / 100)
        inner_mask = np.zeros_like(petri_mask)
        cv2.circle(inner_mask, petri_info.center, inner_radius, 255, -1)
        return inner_mask

    def detect_colonies(
        self,
        image: np.ndarray,
        petri_mask: np.ndarray,
        params: AnalysisParams,
        petri_info: Optional[PetriInfo] = None,
        blur_size: int = 5,
        context: Optional[SampleContext] = None,
    ) -> tuple[np.ndarray, Dict]:

        if petri_info:
            roi_mask = self.create_inner_mask(
                petri_mask, petri_info, params.margin_percent
            )
        else:
            roi_mask = petri_mask

        channel = self.processor.extract_green_channel(image)

        # Контекст: переиспользуем инвариантную предобработку (diff) для одного
        # снимка при совпадающем contrast/blur_size (задача 3.3).
        cached_diff = (
            context.get_preprocessed(params.contrast, blur_size)
            if context is not None
            else None
        )
        if cached_diff is not None:
            diff = cached_diff
        else:
            with measure_stage("classic_preprocess"):
                clip_limit = 2.0 * params.contrast
                enhanced = self.processor.apply_clahe(
                    channel, clip_limit=clip_limit, grid_size=8
                )

                k_size = blur_size if blur_size % 2 == 1 else blur_size + 1
                k_size = max(3, k_size)
                denoised = cv2.medianBlur(enhanced, k_size)

                bg = cv2.GaussianBlur(denoised, (51, 51), 0)
                diff = cv2.addWeighted(denoised, 1.5, bg, -0.5, 0)
            if context is not None:
                context.set_preprocessed(params.contrast, blur_size, diff)
        masked_diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

        valid_pixels = masked_diff[roi_mask > 0]
        if len(valid_pixels) == 0:
            return np.zeros_like(channel), {}

        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

        k = 3.0 - (params.sensitivity * 2.5)
        thresh_val = mean_val + k * std_val
        thresh_val = max(mean_val + 5, min(thresh_val, 254))

        with measure_stage("threshold_morphology"):
            _, binary = cv2.threshold(
                masked_diff, int(thresh_val), 255, cv2.THRESH_BINARY
            )

            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean_binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, kernel_morph, iterations=1
            )

            if params.solid_fill:
                fill_k_size = max(3, params.fill_strength)
                fill_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (fill_k_size, fill_k_size)
                )
                clean_binary = cv2.morphologyEx(
                    clean_binary, cv2.MORPH_CLOSE, fill_kernel
                )

                contours = self._find_contours(clean_binary)
                cv2.drawContours(
                    clean_binary, contours, -1, 255, thickness=cv2.FILLED
                )
            else:
                clean_binary = cv2.morphologyEx(
                    clean_binary, cv2.MORPH_CLOSE, kernel_morph, iterations=2
                )

        final_mask = self._filter_components(
            clean_binary, min_size=params.min_colony_size
        )

        debug_images: Dict = {"preprocessed": masked_diff, "binary": clean_binary}

        return final_mask, debug_images

    def _filter_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Отбрасывает компоненты меньше min_size.

        Векторизованный lookup вместо повторного прохода по меткам
        (задача 3.2): каждая метка проверяется по таблице площадей
        за один проход. Результат побитово эквивалентен прежнему циклу
        (порядок нумерации компонентов детерминирован и сохранён).
        """
        with measure_stage("component_filtering"):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            if num_labels <= 1:
                return np.zeros_like(mask)
            # lookup[метка] = 255, если площадь >= min_size (фон 0 всегда отброшен).
            keep = stats[1:, cv2.CC_STAT_AREA] >= min_size
            lookup = np.zeros(num_labels, dtype=np.uint8)
            lookup[1:] = keep.astype(np.uint8) * 255
            filtered_mask = lookup[labels]
        return filtered_mask

    def count_colonies(self, colony_mask: np.ndarray) -> int:
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        return max(0, num_labels - 1)
