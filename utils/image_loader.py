"""Загрузка изображений: GUI-путь (QPixmap → OpenCV) и worker-safe путь.

Публичные функции:

- `load_image` / `load_image_grayscale` — порядок QPixmap → OpenCV для вызовов
  из GUI-потока (историческое поведение сохранено);
- `load_image_worker_safe` / `load_image_grayscale_worker_safe` — потокобезопасный
  декодер на QImage (НЕ QPixmap — он GUI-ресурс) с fallback на OpenCV:
  используется для фоновой загрузки полноразмерных изображений.

Оба пути сохраняют контракты: цветное изображение — BGR `uint8` (h, w, 3),
grayscale — одноканальный `uint8` (h, w); поддерживаемые форматы — PNG, JPEG,
BMP, TIFF, WebP.
"""

import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage

# Расширения файлов изображений, допустимые в интерфейсе приложения
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _rgb888_to_bgr(qimage: QImage) -> np.ndarray | None:
    """Конвертирует QImage RGB888 в BGR uint8 ndarray (учитывая row padding)."""
    try:
        if qimage.isNull():
            return None
        converted = qimage.convertToFormat(QImage.Format.Format_RGB888)
        if converted.isNull():
            return None
        w = converted.width()
        h = converted.height()
        bpl = converted.bytesPerLine()
        ptr = converted.bits()
        ptr.setsize(converted.sizeInBytes())
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        rgb = raw[:, : 3 * w].reshape((h, w, 3))
        return rgb[:, :, ::-1].copy()
    except Exception:
        return None


def _gray8_to_array(qimage: QImage) -> np.ndarray | None:
    """Конвертирует QImage в одноканальный uint8 ndarray (учитывая row padding)."""
    try:
        if qimage.isNull():
            return None
        converted = qimage.convertToFormat(QImage.Format.Format_Grayscale8)
        if converted.isNull():
            return None
        w = converted.width()
        h = converted.height()
        bpl = converted.bytesPerLine()
        ptr = converted.bits()
        ptr.setsize(converted.sizeInBytes())
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        return raw[:, :w].reshape((h, w)).copy()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GUI-путь: исторический порядок QPixmap -> OpenCV
# ---------------------------------------------------------------------------


def _load_image_pixmap(path: str) -> np.ndarray | None:
    try:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        return _rgb888_to_bgr(pixmap.toImage())
    except Exception:
        return None


def _load_image_grayscale_pixmap(path: str) -> np.ndarray | None:
    try:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        return _gray8_to_array(pixmap.toImage())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Worker-safe путь: QImage (без QPixmap) -> OpenCV fallback
# ---------------------------------------------------------------------------


def _load_image_qimage(path: str) -> np.ndarray | None:
    try:
        return _rgb888_to_bgr(QImage(path))
    except Exception:
        return None


def _load_image_grayscale_qimage(path: str) -> np.ndarray | None:
    try:
        return _gray8_to_array(QImage(path))
    except Exception:
        return None


def _load_image_opencv(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img if img is not None else None


def _load_image_grayscale_opencv(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img if img is not None else None


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    """Загрузка из GUI-потока: QPixmap -> OpenCV."""
    bgr = _load_image_pixmap(path)
    if bgr is not None:
        return bgr
    bgr = _load_image_opencv(path)
    if bgr is not None:
        return bgr
    raise ValueError(
        f"Не удалось загрузить изображение: {path} "
        f"(QPixmap и OpenCV не смогли декодировать файл)"
    )


def load_image_grayscale(path: str) -> np.ndarray:
    """Загрузка grayscale из GUI-потока: QPixmap -> OpenCV."""
    gray = _load_image_grayscale_pixmap(path)
    if gray is not None:
        return gray
    gray = _load_image_grayscale_opencv(path)
    if gray is not None:
        return gray
    raise ValueError(
        f"Не удалось загрузить изображение (grayscale): {path} "
        f"(QPixmap и OpenCV не смогли декодировать файл)"
    )


def load_image_worker_safe(path: str) -> np.ndarray:
    """Загрузка из фонового потока: QImage (без QPixmap) -> OpenCV.

    Возвращает BGR uint8 (h, w, 3); годится для вызова из worker-потока.
    """
    bgr = _load_image_qimage(path)
    if bgr is not None:
        return bgr
    bgr = _load_image_opencv(path)
    if bgr is not None:
        return bgr
    raise ValueError(
        f"Не удалось загрузить изображение: {path} "
        f"(QImage и OpenCV не смогли декодировать файл)"
    )


def load_image_grayscale_worker_safe(path: str) -> np.ndarray:
    """Загрузка grayscale из фонового потока: QImage (без QPixmap) -> OpenCV.

    Возвращает одноканальный uint8 (h, w); годится для вызова из worker-потока.
    """
    gray = _load_image_grayscale_qimage(path)
    if gray is not None:
        return gray
    gray = _load_image_grayscale_opencv(path)
    if gray is not None:
        return gray
    raise ValueError(
        f"Не удалось загрузить изображение (grayscale): {path} "
        f"(QImage и OpenCV не смогли декодировать файл)"
    )
