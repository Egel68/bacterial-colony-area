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

        # 3. МОРФОЛОГИЧЕСКИЙ TOP-HAT (Идеально удаляет неравномерный фон и оставляет только колонии)
        tophat_kernel_size = 51  # Ядро больше, чем самая крупная одиночная колония
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

        # Настраиваем порог отталкиваясь от чувствительности (плавный контроль)
        # При sensitivity 1.0 -> k ~ 0.5 (очень чувствительно)
        # При sensitivity 0.0 -> k ~ 4.0 (только самые яркие)
        k = 4.0 - (sensitivity * 3.5)
        thresh_val = mean_val + k * std_val
        thresh_val = np.clip(thresh_val, 5, 250)

        _, binary = cv2.threshold(
            masked_tophat, int(thresh_val), 255, cv2.THRESH_BINARY
        )

        # Очистка базового шума
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_morph, iterations=1)

        # 5. ФОРМИРОВАНИЕ ГРАНИЦ И РАЗДЕЛЕНИЕ СЛИПШИХСЯ КОЛОНИЙ
        if use_solid_fill:
            # Сплошная заливка мазков (оставляем вашу логику)
            fill_k_size = max(3, fill_strength)
            fill_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (fill_k_size, fill_k_size)
            )
            clean_binary = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, fill_kernel)
            contours = self._find_contours(clean_binary)
            cv2.drawContours(clean_binary, contours, -1, 255, thickness=cv2.FILLED)
        else:
            # ALGORITHM WATERSHED (Водораздел) для разделения одиночных колоний

            # Точный фон (где точно нет колоний)
            sure_bg = cv2.dilate(opening, kernel_morph, iterations=2)

            # Вычисление дистанции от краев к центру колоний (Distance Transform)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

            # Поиск точных центров колоний. Множитель зависит от чувствительности.
            # Если колонии сильно слиплись, порог должен быть выше
            dt_multiplier = 0.6 - (sensitivity * 0.4)
            _, sure_fg = cv2.threshold(
                dist_transform, dt_multiplier * dist_transform.max(), 255, 0
            )
            sure_fg = np.uint8(sure_fg)

            # Неизвестная зона (где колонии соприкасаются)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Маркировка центров для алгоритма
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            # Применяем Watershed на исходном отфильтрованном изображении (переведенном в BGR)
            img_for_watershed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_for_watershed, markers)

            # Восстанавливаем маску колоний (все что больше 1 - это колонии, 1 - это фон, -1 - границы)
            clean_binary = np.zeros_like(opening)
            clean_binary[markers > 1] = 255

        # Обрезаем маску строго по рабочей зоне (ROI), чтобы края чашки не давали артефактов
        clean_binary = cv2.bitwise_and(clean_binary, clean_binary, mask=roi_mask)

        # 6. Фильтрация по минимальному размеру
        final_mask = self._filter_components(clean_binary, min_size=min_colony_size)

        # Для UI возвращаем то, что помогает диагностировать работу пайплайна
        debug_images = {
            "preprocessed": masked_tophat,  # Показываем, как отработал алгоритм выравнивания освещения
            "binary": final_mask,
        }

        return final_mask, debug_images

    def _filter_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered_mask = np.zeros_like(mask)
        # i=0 - это фон, пропускаем его
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                filtered_mask[labels == i] = 255
        return filtered_mask

    def count_colonies(self, colony_mask: np.ndarray) -> int:
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        return max(0, num_labels - 1)
