"""
Модуль для обнаружения чашки Петри и бактериальных колоний.
Содержит алгоритмы компьютерного зрения для сегментации.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_processor import ImageProcessor


class ColonyDetector:
    """
    Класс для обнаружения чашки Петри и бактериальных колоний.
    """

    EDGE_MARGIN_PERCENT = 5

    def __init__(self):
        self.processor = ImageProcessor()

    def _find_contours(self, mask: np.ndarray) -> List:
        """
        Обёртка для cv2.findContours, совместимая с разными версиями OpenCV.

        Args:
            mask: Бинарная маска

        Returns:
            Список контуров
        """
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 4.x возвращает 2 значения, OpenCV 3.x - 3 значения
        contours = result[0] if len(result) == 2 else result[1]
        return contours

    def detect_petri_dish(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Обнаружение чашки Петри на изображении.

        Args:
            image: Входное BGR изображение

        Returns:
            Tuple из (маска чашки Петри, информация о чашке)
        """
        gray = self.processor.to_grayscale(image)

        # Метод 1: HoughCircles
        result = self._detect_petri_with_hough(image)
        if result[0] is not None:
            return result

        # Метод 2: Контурный анализ
        result = self._detect_petri_with_contours(image)
        if result[0] is not None:
            return result

        return None, None

    def _detect_petri_with_hough(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Обнаружение чашки Петри с помощью преобразования Хафа."""
        gray = self.processor.to_grayscale(image)
        blurred = self.processor.apply_gaussian_blur(gray, kernel_size=9)

        min_dim = min(image.shape[:2])
        min_radius = int(min_dim * 0.15)
        max_radius = int(min_dim * 0.49)

        for param2 in [25, 35, 50, 20, 15]:
            for dp in [1.0, 1.2, 1.5]:
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=dp,
                    minDist=min_dim // 2,
                    param1=50,
                    param2=param2,
                    minRadius=min_radius,
                    maxRadius=max_radius,
                )

                if circles is not None:
                    circle = circles[0][0]
                    center = (int(circle[0]), int(circle[1]))
                    radius = int(circle[2])

                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, center, radius, 255, -1)

                    petri_info = {
                        "center": center,
                        "radius": radius,
                        "area_px": int(np.pi * radius**2),
                        "circularity": 1.0,
                    }

                    return mask, petri_info

        return None, None

    def _detect_petri_with_contours(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Обнаружение чашки Петри через контурный анализ."""
        gray = self.processor.to_grayscale(image)
        blurred = self.processor.apply_gaussian_blur(gray, kernel_size=7)

        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours = self._find_contours(thresh)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        image_area = image.shape[0] * image.shape[1]
        if area < image_area * 0.05:
            return None, None

        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        petri_info = {
            "center": center,
            "radius": radius,
            "area_px": int(np.pi * radius**2),
            "circularity": 1.0,
        }

        return mask, petri_info

    def create_inner_mask(
        self, petri_mask: np.ndarray, petri_info: Dict, margin_percent: float = 5
    ) -> np.ndarray:
        """
        Создание внутренней маски с отступом от краёв чашки Петри.
        """
        center = petri_info["center"]
        radius = petri_info["radius"]

        inner_radius = int(radius * (100 - margin_percent) / 100)

        inner_mask = np.zeros_like(petri_mask)
        cv2.circle(inner_mask, center, inner_radius, 255, -1)

        return inner_mask

    def detect_colonies(
        self,
        image: np.ndarray,
        petri_mask: np.ndarray,
        petri_info: Dict = None,
        sensitivity: float = 0.5,
        min_colony_size: int = 100,
        edge_margin_percent: float = 5,
    ) -> np.ndarray:
        """
        Обнаружение бактериальных колоний.

        Args:
            image: Входное BGR изображение
            petri_mask: Маска чашки Петри
            petri_info: Информация о чашке Петри
            sensitivity: Чувствительность обнаружения (0-1)
            min_colony_size: Минимальный размер колонии в пикселях
            edge_margin_percent: Отступ от края чашки в процентах

        Returns:
            Бинарная маска колоний
        """
        # Создаём внутреннюю маску
        if petri_info is not None:
            inner_mask = self.create_inner_mask(
                petri_mask, petri_info, edge_margin_percent
            )
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            inner_mask = cv2.erode(petri_mask, kernel, iterations=2)

        # Применяем внутреннюю маску к изображению
        masked_image = cv2.bitwise_and(image, image, mask=inner_mask)

        # Преобразуем в grayscale
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Преобразуем в HSV
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Получаем статистику внутри чашки
        inner_pixels = gray[inner_mask > 0]

        if len(inner_pixels) == 0:
            return np.zeros_like(petri_mask)

        mean_brightness = np.mean(inner_pixels)
        std_brightness = np.std(inner_pixels)
        median_brightness = np.median(inner_pixels)

        # Список масок для комбинирования
        colony_masks = []

        # Метод 1: Светлые колонии (светлее фона)
        bright_threshold = median_brightness + std_brightness * (1.5 - sensitivity)
        bright_threshold = max(bright_threshold, median_brightness + 10)
        _, bright_mask = cv2.threshold(
            gray, int(bright_threshold), 255, cv2.THRESH_BINARY
        )
        bright_mask = cv2.bitwise_and(bright_mask, inner_mask)
        colony_masks.append(bright_mask)

        # Метод 2: Тёмные колонии (темнее фона)
        dark_threshold = median_brightness - std_brightness * (1.5 - sensitivity)
        dark_threshold = min(dark_threshold, median_brightness - 10)
        if dark_threshold > 5:
            _, dark_mask = cv2.threshold(
                gray, int(dark_threshold), 255, cv2.THRESH_BINARY_INV
            )
            dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
            # Исключаем чёрные области
            _, non_black = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            dark_mask = cv2.bitwise_and(dark_mask, non_black)
            colony_masks.append(dark_mask)

        # Метод 3: Адаптивная пороговая обработка
        block_size = max(11, int(151 * (1 - sensitivity * 0.7)))
        if block_size % 2 == 0:
            block_size += 1

        c_value = max(3, int(25 * (1 - sensitivity)))

        adaptive_bright = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -c_value,
        )
        adaptive_bright = cv2.bitwise_and(adaptive_bright, inner_mask)
        colony_masks.append(adaptive_bright)

        # Метод 4: Цветные колонии (высокая насыщенность)
        s_values = s[inner_mask > 0]
        if len(s_values) > 0:
            s_mean = np.mean(s_values)
            s_std = np.std(s_values)
            s_threshold = s_mean + s_std * (1 - sensitivity * 0.5)
            _, color_mask = cv2.threshold(s, int(s_threshold), 255, cv2.THRESH_BINARY)
            color_mask = cv2.bitwise_and(color_mask, inner_mask)
            colony_masks.append(color_mask)

        # Комбинируем результаты
        combined = np.zeros_like(petri_mask)

        for mask in colony_masks:
            cleaned = self._clean_mask(mask)
            combined = cv2.bitwise_or(combined, cleaned)

        # Финальная очистка
        combined = self._clean_mask(combined)

        # Удаляем объекты на границе
        combined = self._remove_border_objects(combined, inner_mask)

        # Фильтрация по размеру
        filtered = self._filter_by_size(combined, min_colony_size)

        return filtered

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Очистка маски морфологическими операциями."""
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)

        return cleaned

    def _remove_border_objects(
        self, mask: np.ndarray, region_mask: np.ndarray
    ) -> np.ndarray:
        """Удаление объектов, касающихся границы области."""
        contours = self._find_contours(region_mask)

        if not contours:
            return mask

        border_mask = np.zeros_like(region_mask)
        cv2.drawContours(border_mask, contours, -1, 255, thickness=10)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        result = np.zeros_like(mask)

        for i in range(1, num_labels):
            component_mask = (labels == i).astype(np.uint8) * 255
            overlap = cv2.bitwise_and(component_mask, border_mask)

            if np.count_nonzero(overlap) == 0:
                result = cv2.bitwise_or(result, component_mask)

        return result

    def _filter_by_size(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Фильтрация объектов по размеру."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        filtered = np.zeros_like(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                filtered[labels == i] = 255

        return filtered

    def count_colonies(self, colony_mask: np.ndarray) -> int:
        """Подсчёт количества отдельных колоний."""
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        return num_labels - 1
