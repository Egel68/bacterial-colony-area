"""
Модуль для обнаружения чашки Петри и бактериальных колоний.
Содержит алгоритмы компьютерного зрения для сегментации.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .image_processor import ImageProcessor


class ColonyDetector:
    """
    Класс для обнаружения чашки Петри и бактериальных колоний.

    Алгоритм работы:
    1. Обнаружение чашки Петри:
       - Преобразование в grayscale
       - Размытие для уменьшения шума
       - Определение порога для отделения от тёмного фона
       - Поиск контуров и определение наибольшего круглого объекта
       - Использование HoughCircles для точного определения круга

    2. Обнаружение колоний:
       - Создание внутренней маски (с отступом от краёв чашки)
       - Преобразование в цветовое пространство HSV
       - Адаптивная пороговая обработка
       - Выделение областей с характерной яркостью/цветом колоний
       - Морфологическая очистка
       - Фильтрация по размеру
    """

    # Отступ от края чашки Петри в процентах от радиуса
    EDGE_MARGIN_PERCENT = 5

    def __init__(self):
        self.processor = ImageProcessor()

    def detect_petri_dish(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Обнаружение чашки Петри на изображении.

        Алгоритм:
        1. Преобразуем в grayscale и применяем размытие
        2. Используем пороговую обработку для отделения от тёмного фона
        3. Находим контуры
        4. Ищем наибольший контур, похожий на круг
        5. Вписываем минимальную окружность

        Args:
            image: Входное BGR изображение

        Returns:
            Tuple из (маска чашки Петри, информация о чашке)
        """
        # Предобработка
        gray = self.processor.to_grayscale(image)
        blurred = self.processor.apply_gaussian_blur(gray, kernel_size=7)

        # Пробуем сначала метод HoughCircles - он более точный для кругов
        result = self._detect_petri_with_hough(image)
        if result[0] is not None:
            return result

        # Если HoughCircles не сработал, пробуем контурный метод
        # Метод: Пороговая обработка (для тёмного фона)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        # Морфологические операции для очистки
        thresh = self.processor.apply_morphology(thresh, "close", kernel_size=15)
        thresh = self.processor.apply_morphology(thresh, "open", kernel_size=5)

        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None

        # Находим наибольший контур
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Проверяем, достаточно ли большой контур
        image_area = image.shape[0] * image.shape[1]
        if area < image_area * 0.1:  # Менее 10% изображения
            return None, None

        # Вписываем окружность
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Проверяем "круглость" контура
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        if circularity < 0.5:  # Контур недостаточно круглый
            return None, None

        # Создаём маску
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        petri_info = {
            "center": center,
            "radius": radius,
            "area": np.pi * radius**2,
            "circularity": circularity,
        }

        return mask, petri_info

    def _detect_petri_with_hough(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Обнаружение чашки Петри с помощью преобразования Хафа.

        Args:
            image: Входное BGR изображение

        Returns:
            Tuple из (маска чашки Петри, информация о чашке)
        """
        gray = self.processor.to_grayscale(image)
        blurred = self.processor.apply_gaussian_blur(gray, kernel_size=9)

        # Определяем минимальный и максимальный радиус
        min_dim = min(image.shape[:2])
        min_radius = int(min_dim * 0.15)
        max_radius = int(min_dim * 0.48)

        # Поиск кругов с разными параметрами
        for param2 in [30, 50, 70, 20]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dim // 2,
                param1=50,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )

            if circles is not None:
                break

        if circles is None:
            return None, None

        # Берём первый (наиболее вероятный) круг
        circle = circles[0][0]
        center = (int(circle[0]), int(circle[1]))
        radius = int(circle[2])

        # Создаём маску
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        petri_info = {
            "center": center,
            "radius": radius,
            "area": np.pi * radius**2,
            "circularity": 1.0,  # Идеальный круг
        }

        return mask, petri_info

    def create_inner_mask(
        self, petri_mask: np.ndarray, petri_info: Dict, margin_percent: float = None
    ) -> np.ndarray:
        """
        Создание внутренней маски с отступом от краёв чашки Петри.
        Это исключает края чашки из анализа колоний.

        Args:
            petri_mask: Маска чашки Петри
            petri_info: Информация о чашке Петри
            margin_percent: Отступ от края в процентах от радиуса

        Returns:
            Внутренняя маска
        """
        if margin_percent is None:
            margin_percent = self.EDGE_MARGIN_PERCENT

        center = petri_info["center"]
        radius = petri_info["radius"]

        # Вычисляем внутренний радиус
        inner_radius = int(radius * (100 - margin_percent) / 100)

        # Создаём внутреннюю маску
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

        Алгоритм:
        1. Создаём внутреннюю маску (исключаем края чашки)
        2. Применяем маску к изображению
        3. Преобразуем в HSV для лучшего выделения колоний
        4. Колонии обычно светлее агара (повышенная яркость)
        5. Применяем адаптивную пороговую обработку
        6. Очищаем морфологическими операциями
        7. Фильтруем мелкие объекты

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
        # Создаём внутреннюю маску (исключаем края чашки Петри)
        if petri_info is not None:
            inner_mask = self.create_inner_mask(
                petri_mask, petri_info, edge_margin_percent
            )
        else:
            # Если нет информации, используем эрозию
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            inner_mask = cv2.erode(petri_mask, kernel, iterations=2)

        # Применяем внутреннюю маску (исключаем края)
        masked = cv2.bitwise_and(image, image, mask=inner_mask)

        # Улучшаем контраст
        enhanced = self.processor.enhance_contrast(masked)

        # Преобразуем в HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Вычисляем средние значения яркости в области чашки Петри (без краёв)
        inner_values = v[inner_mask > 0]

        if len(inner_values) == 0:
            return np.zeros_like(petri_mask)

        mean_v = np.mean(inner_values)
        std_v = np.std(inner_values)
        median_v = np.median(inner_values)

        # Метод 1: Порог на основе статистики яркости
        # Колонии светлее среднего агара
        threshold_offset = std_v * (1.5 - sensitivity)
        threshold_value = median_v + threshold_offset

        # Ограничиваем порог разумными значениями
        threshold_value = max(min(threshold_value, 250), mean_v + 5)

        _, brightness_thresh = cv2.threshold(
            v, int(threshold_value), 255, cv2.THRESH_BINARY
        )

        # Метод 2: Адаптивная пороговая обработка
        block_size = max(11, int(101 * (1 - sensitivity * 0.7)))
        if block_size % 2 == 0:
            block_size += 1

        c_value = max(3, int(20 * (1 - sensitivity)))

        adaptive_thresh = cv2.adaptiveThreshold(
            v,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -c_value,
        )

        # Метод 3: Обнаружение цветных колоний (высокая насыщенность)
        s_mean = np.mean(s[inner_mask > 0])
        s_std = np.std(s[inner_mask > 0])
        s_threshold = s_mean + s_std * 0.5

        _, saturation_thresh = cv2.threshold(
            s, int(s_threshold), 255, cv2.THRESH_BINARY
        )

        # Комбинируем методы яркости
        combined = cv2.bitwise_and(brightness_thresh, adaptive_thresh)

        # Добавляем цветные колонии
        colored_colonies = cv2.bitwise_and(saturation_thresh, brightness_thresh)
        combined = cv2.bitwise_or(combined, colored_colonies)

        # Применяем ВНУТРЕННЮЮ маску (без краёв чашки!)
        combined = cv2.bitwise_and(combined, inner_mask)

        # Морфологическая очистка
        combined = self.processor.apply_morphology(combined, "open", kernel_size=3)
        combined = self.processor.apply_morphology(combined, "close", kernel_size=3)

        # Удаляем объекты, которые касаются границы внутренней маски
        combined = self._remove_border_objects(combined, inner_mask)

        # Фильтрация по размеру
        filtered = self._filter_by_size(combined, min_colony_size)

        return filtered

    def _remove_border_objects(
        self, mask: np.ndarray, region_mask: np.ndarray
    ) -> np.ndarray:
        """
        Удаление объектов, касающихся границы области.

        Args:
            mask: Бинарная маска объектов
            region_mask: Маска области

        Returns:
            Очищенная маска
        """
        # Создаём контур области
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return mask

        # Создаём маску границы (кольцо)
        border_mask = np.zeros_like(region_mask)
        cv2.drawContours(border_mask, contours, -1, 255, thickness=15)

        # Находим компоненты, которые касаются границы
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        result = np.zeros_like(mask)

        for i in range(1, num_labels):
            component_mask = (labels == i).astype(np.uint8) * 255

            # Проверяем, касается ли компонент границы
            overlap = cv2.bitwise_and(component_mask, border_mask)

            if np.count_nonzero(overlap) == 0:
                # Компонент не касается границы - оставляем
                result = cv2.bitwise_or(result, component_mask)

        return result

    def _filter_by_size(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """
        Фильтрация объектов по размеру.

        Args:
            mask: Бинарная маска
            min_size: Минимальный размер объекта

        Returns:
            Отфильтрованная маска
        """
        # Находим все связные компоненты
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Создаём новую маску
        filtered = np.zeros_like(mask)

        for i in range(1, num_labels):  # Пропускаем фон (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                filtered[labels == i] = 255

        return filtered

    def count_colonies(self, colony_mask: np.ndarray) -> int:
        """
        Подсчёт количества отдельных колоний.

        Args:
            colony_mask: Бинарная маска колоний

        Returns:
            Количество колоний
        """
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        return num_labels - 1  # Вычитаем фон
