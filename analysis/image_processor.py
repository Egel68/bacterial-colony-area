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
        """
        # Медианный фильтр хорошо убирает соль-перец шум, сохраняя края
        denoised = cv2.medianBlur(image, 3)
        return denoised

    @staticmethod
    def apply_clahe(
        image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8
    ) -> np.ndarray:
        """
        Применение CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Отлично подходит для выделения деталей на темном фоне.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
        )
        return clahe.apply(gray)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def extract_green_channel(image: np.ndarray) -> np.ndarray:
        """
        Извлечение зеленого канала (часто наиболее контрастен для био-изображений).
        """
        if len(image.shape) == 3:
            return image[:, :, 1]
        return image

    @staticmethod
    def apply_tophat(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        Применение преобразования Top-Hat.
        Выделяет светлые объекты на темном фоне, игнорируя градиенты освещения.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # TopHat = src - open(src)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    @staticmethod
    def resize_image(
        image: np.ndarray, max_dimension: int = 1024
    ) -> Tuple[np.ndarray, float]:
        """Изменение размера изображения с сохранением пропорций."""
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
