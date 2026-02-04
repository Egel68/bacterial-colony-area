"""
Модуль для базовой обработки изображений.
Содержит функции предобработки и фильтрации.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class ImageProcessor:
    """Класс для предобработки изображений."""

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для анализа.

        Args:
            image: Входное BGR изображение

        Returns:
            Предобработанное изображение
        """
        # Уменьшение шума с сохранением краёв
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Улучшение контраста изображения.

        Args:
            image: Входное BGR изображение

        Returns:
            Изображение с улучшенным контрастом
        """
        # Конвертируем в LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Применяем CLAHE к каналу яркости
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Собираем обратно
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Конвертация в оттенки серого.

        Args:
            image: Входное BGR изображение

        Returns:
            Grayscale изображение
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Применение размытия по Гауссу.

        Args:
            image: Входное изображение
            kernel_size: Размер ядра (должен быть нечётным)

        Returns:
            Размытое изображение
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_morphology(
        mask: np.ndarray,
        operation: str = "close",
        kernel_size: int = 5,
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Применение морфологических операций.

        Args:
            mask: Бинарная маска
            operation: Тип операции ('open', 'close', 'erode', 'dilate')
            kernel_size: Размер ядра
            iterations: Количество итераций

        Returns:
            Обработанная маска
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        operations = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
        }

        if operation in operations:
            return cv2.morphologyEx(
                mask, operations[operation], kernel, iterations=iterations
            )

        return mask

    @staticmethod
    def resize_image(
        image: np.ndarray, max_dimension: int = 1024
    ) -> Tuple[np.ndarray, float]:
        """
        Изменение размера изображения с сохранением пропорций.

        Args:
            image: Входное изображение
            max_dimension: Максимальный размер по любой стороне

        Returns:
            Tuple из (изменённое изображение, коэффициент масштабирования)
        """
        h, w = image.shape[:2]

        if max(h, w) <= max_dimension:
            return image, 1.0

        if h > w:
            scale = max_dimension / h
        else:
            scale = max_dimension / w

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized, scale
