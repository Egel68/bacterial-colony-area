import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage

# Расширения файлов изображений, допустимые в интерфейсе приложения
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _load_image_pixmap(path: str) -> np.ndarray | None:
    try:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        if qimage.isNull():
            return None
        w = qimage.width()
        h = qimage.height()
        bpl = qimage.bytesPerLine()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        rgb = raw[:, :3 * w].reshape((h, w, 3))
        return rgb[:, :, ::-1].copy()
    except Exception:
        return None


def _load_image_opencv(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img if img is not None else None


def load_image(path: str) -> np.ndarray:
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


def _load_image_grayscale_pixmap(path: str) -> np.ndarray | None:
    try:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
        if qimage.isNull():
            return None
        w = qimage.width()
        h = qimage.height()
        bpl = qimage.bytesPerLine()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        return raw[:, :w].reshape((h, w)).copy()
    except Exception:
        return None


def _load_image_grayscale_opencv(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img if img is not None else None


def load_image_grayscale(path: str) -> np.ndarray:
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
