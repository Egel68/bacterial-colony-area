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
        """
        Обнаружение чашки Петри.
        Использует тот факт, что края чашки на этих фото - самые яркие объекты (блики).
        """
        gray = self.processor.to_grayscale(image)
        h, w = gray.shape

        # 1. Грубая бинаризация для поиска ярких бликов по кругу
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # Берем только очень яркие пиксели (блики)
        _, bright_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Если бликов мало (тусклое фото), используем адаптивный порог
        if cv2.countNonZero(bright_mask) < (h * w * 0.01):
            bright_mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                101,
                -10,
            )

        # 2. Объединяем блики в один контур (замыкание)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        closed_mask = cv2.morphologyEx(
            bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # 3. Ищем самый большой внешний контур
        contours = self._find_contours(closed_mask)

        best_circle = None
        max_area = 0

        center_image = (w // 2, h // 2)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (h * w * 0.1):  # Игнорируем мелкий мусор
                continue

            # Описываем круг
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Проверяем, что круг примерно по центру (с допуском)
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

            # Создаем маску
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            petri_info = {
                "center": center,
                "radius": radius,
                "area_px": int(np.pi * radius**2),
            }
            return mask, petri_info

        # Fallback: Если не нашли по бликам, пробуем HoughCircles
        return self._detect_petri_with_hough(image)

    def _detect_petri_with_hough(self, image: np.ndarray) -> Tuple:
        """Резервный метод поиска через Хафа."""
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
        """
        Создание внутренней маски.
        Критически важно отступить от края, чтобы убрать блики кольцевой лампы.
        """
        center = petri_info["center"]
        radius = petri_info["radius"]

        # Увеличиваем отступ, так как блики обычно широкие
        real_margin = max(margin_percent, 8.0)

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
    ) -> np.ndarray:
        """
        Обнаружение колоний.

        Стратегия:
        1. Работаем в зеленом канале (лучший контраст для ч/б камер и большинства агаров).
        2. Применяем CLAHE для выравнивания освещения.
        3. Используем пороговое значение + морфологию.
        """
        # 1. Подготовка маски ROI (области интереса)
        if petri_info:
            roi_mask = self.create_inner_mask(
                petri_mask, petri_info, edge_margin_percent
            )
        else:
            roi_mask = petri_mask

        # 2. Работаем с изображением
        # Используем зеленый канал, он обычно менее шумный чем синий и контрастнее красного для белого на черном
        channel = self.processor.extract_green_channel(image)

        # 3. Улучшение контраста (CLAHE)
        # ClipLimit влияет на чувствительность (больше = больше деталей, но и шума)
        # Sensitivity 0.0 -> Clip 1.0, Sensitivity 1.0 -> Clip 6.0
        clip_limit = 1.0 + (sensitivity * 5.0)
        enhanced = self.processor.apply_clahe(
            channel, clip_limit=clip_limit, grid_size=8
        )

        # 4. Подавление фона (Morphological Top-Hat не подходит для крупных мазков,
        # поэтому используем вычитание размытого фона для удаления градиента)
        bg = cv2.GaussianBlur(enhanced, (51, 51), 0)
        # Добавляем 128, чтобы не уйти в минус при вычитании, потом нормализуем
        diff = cv2.addWeighted(enhanced, 1.5, bg, -0.5, 0)

        # Применяем маску ROI сразу, чтобы блики не влияли на гистограмму
        masked_diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

        # 5. Бинаризация
        # Берем только пиксели внутри маски для расчета порога
        valid_pixels = masked_diff[roi_mask > 0]

        if len(valid_pixels) == 0:
            return np.zeros_like(channel)

        # Вычисляем порог на основе статистики пикселей внутри чашки
        # Колонии - это самые яркие пиксели
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

        # Порог: среднее + k * стд. отклонение.
        # Чем выше чувствительность, тем ниже порог (k меньше)
        # Sens 1.0 -> k = 0.5, Sens 0.0 -> k = 3.0
        k = 3.0 - (sensitivity * 2.5)
        thresh_val = mean_val + k * std_val

        # Ограничиваем порог разумными рамками (не ниже фона, не выше максимума)
        thresh_val = max(mean_val + 5, min(thresh_val, 254))

        _, binary = cv2.threshold(masked_diff, int(thresh_val), 255, cv2.THRESH_BINARY)

        # 6. Очистка шума
        # Morph Open удаляет мелкие точки (шум)
        kernel_size = 3
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Morph Close "заливает" дырки внутри колоний
        clean_binary = cv2.morphologyEx(
            clean_binary, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # 7. Фильтрация по размеру
        final_mask = self._filter_components(clean_binary, min_size=min_colony_size)

        return final_mask

    def _filter_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Фильтрация связных компонентов по размеру."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        filtered_mask = np.zeros_like(mask)

        # stats: [left, top, width, height, area]
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                filtered_mask[labels == i] = 255

        return filtered_mask

    def count_colonies(self, colony_mask: np.ndarray) -> int:
        """Подсчёт количества отдельных колоний."""
        # Для точного подсчета слипшихся колоний (Watershed) нужен более сложный алгоритм,
        # но для базовой задачи достаточно ConnectedComponents
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        return max(0, num_labels - 1)
