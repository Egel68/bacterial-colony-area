"""
Модуль для обнаружения чашки Петри и бактериальных колоний.
Оптимизирован для темного поля (Dark Field) и изображений с бликами.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_processor import ImageProcessor


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
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Обнаружение чашки Петри."""
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
            petri_info = {
                "center": center,
                "radius": radius,
                "area_px": int(np.pi * radius**2),
            }
            return mask, petri_info

        return self._detect_petri_with_hough(image)

    def _detect_petri_with_hough(self, image: np.ndarray) -> Tuple:
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
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            return mask, {
                "center": center,
                "radius": radius,
                "area_px": int(np.pi * radius**2),
            }
        return None, None

    def create_inner_mask(
        self, petri_mask: np.ndarray, petri_info: Dict, margin_percent: float = 5
    ) -> np.ndarray:
        center = petri_info["center"]
        radius = petri_info["radius"]
        real_margin = max(margin_percent, 1.0)
        inner_radius = int(radius * (100 - real_margin) / 100)
        inner_mask = np.zeros_like(petri_mask)
        cv2.circle(inner_mask, center, inner_radius, 255, -1)
        return inner_mask

    def detect_colonies(
        self,
        image: np.ndarray,
        petri_mask: np.ndarray,
        petri_info: Dict = None,
        sensitivity: float = 0.5,
        min_colony_size: int = 50,
        edge_margin_percent: float = 10,
        contrast_level: float = 1.0,
        blur_size: int = 5,
        use_solid_fill: bool = False,
        fill_strength: int = 15,
        algorithm_mode: int = 2,  # 0-Оригинальный, 1-Улучшенный, 2-Продвинутый (с фильтрацией)
    ) -> Tuple[np.ndarray, Dict]:

        # 1. Формируем маску рабочей зоны (ROI)
        if petri_info:
            roi_mask = self.create_inner_mask(
                petri_mask, petri_info, edge_margin_percent
            )
        else:
            roi_mask = petri_mask

        # 2. Выделяем нужный канал и улучшаем контраст
        channel = self.processor.extract_green_channel(image)
        clip_limit = 2.0 * contrast_level
        enhanced = self.processor.apply_clahe(
            channel, clip_limit=clip_limit, grid_size=8
        )

        k_size = max(3, blur_size if blur_size % 2 == 1 else blur_size + 1)
        denoised = cv2.medianBlur(enhanced, k_size)

        # 3. МОРФОЛОГИЧЕСКИЙ TOP-HAT (удаляет неравномерный фон)
        tophat_kernel_size = 51
        bg_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tophat_kernel_size, tophat_kernel_size)
        )
        tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, bg_kernel)

        masked_tophat = cv2.bitwise_and(tophat, tophat, mask=roi_mask)

        # 4. Адаптивная бинаризация
        valid_pixels = masked_tophat[roi_mask > 0]
        if len(valid_pixels) == 0:
            return np.zeros_like(channel), {}

        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

        if algorithm_mode == 0:
            # Оригинальный подход: Слишком резкая зависимость
            k = 4.0 - (sensitivity * 3.5)
            thresh_val = mean_val + k * std_val
        else:
            # Улучшенный подход (ColTapp, OpenCFU): Комбинация Otsu и статистики
            otsu_thresh = cv2.threshold(masked_tophat, 0, 255, cv2.THRESH_OTSU)[0]
            median_val = np.median(valid_pixels)
            stat_thresh = median_val + (3.0 - sensitivity * 2.5) * std_val
            # Взвешиваем пороги для максимальной стабильности
            thresh_val = 0.7 * otsu_thresh + 0.3 * stat_thresh

        thresh_val = np.clip(thresh_val, 5, 250)

        _, binary = cv2.threshold(
            masked_tophat, int(thresh_val), 255, cv2.THRESH_BINARY
        )

        # Очистка базового шума
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_morph, iterations=1)

        # 5. ФОРМИРОВАНИЕ ГРАНИЦ И РАЗДЕЛЕНИЕ СЛИПШИХСЯ КОЛОНИЙ
        if use_solid_fill:
            fill_k_size = max(3, fill_strength)
            fill_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (fill_k_size, fill_k_size)
            )
            clean_binary = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, fill_kernel)
            contours = self._find_contours(clean_binary)
            cv2.drawContours(clean_binary, contours, -1, 255, thickness=cv2.FILLED)
        else:
            # WATERSHED (Водораздел)
            sure_bg = cv2.dilate(opening, kernel_morph, iterations=2)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

            if algorithm_mode == 0:
                # Оригинальный метод
                dt_multiplier = 0.6 - (sensitivity * 0.4)
            else:
                # Метод Vincent & Soille (1991): Жесткие границы диапазона
                dt_multiplier = 0.4 - (sensitivity * 0.2)
                dt_multiplier = max(0.15, min(0.45, dt_multiplier))

            _, sure_fg = cv2.threshold(
                dist_transform, dt_multiplier * dist_transform.max(), 255, 0
            )
            sure_fg = np.uint8(sure_fg)

            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            img_for_watershed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_for_watershed, markers)

            clean_binary = np.zeros_like(opening)
            clean_binary[markers > 1] = 255

        clean_binary = cv2.bitwise_and(clean_binary, clean_binary, mask=roi_mask)

        # 6. Фильтрация
        if algorithm_mode < 2:
            # Обычная фильтрация только по минимальному размеру
            final_mask = self._filter_components(clean_binary, min_size=min_colony_size)
        else:
            # Продвинутая фильтрация (с учетом геометрической формы - extent & circularity)
            final_mask = self._filter_components_advanced(
                clean_binary, min_size=min_colony_size
            )

        debug_images = {
            "preprocessed": masked_tophat,
            "binary": final_mask,
        }

        return final_mask, debug_images

    def _filter_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                filtered_mask[labels == i] = 255
        return filtered_mask

    def _filter_components_advanced(
        self, mask: np.ndarray, min_size: int
    ) -> np.ndarray:
        """
        Геометрическая фильтрация (Brugger et al. / OpenCFU).
        Удаляет вытянутые царапины, артефакты краев чашки и нетипичный мусор.
        """
        contours = self._find_contours(mask)
        filtered_mask = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size:
                continue

            # 1. Extent (площадь / площадь ограничивающего прямоугольника)
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_box_area = w * h
            extent = area / float(bounding_box_area) if bounding_box_area > 0 else 0

            # 2. Circularity (Круглость)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

            # Колонии могут быть слипшимися (что снижает круглость),
            # но они почти никогда не имеют extent < 0.25 и circularity < 0.2
            if extent >= 0.3 and circularity >= 0.25:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return filtered_mask

    def count_colonies(self, colony_mask: np.ndarray) -> int:
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        return max(0, num_labels - 1)
