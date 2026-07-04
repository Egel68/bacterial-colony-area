from PyQt6.QtGui import QPixmap, QImage
import numpy as np


def load_image(path: str) -> np.ndarray:
    pixmap = QPixmap(path)
    if pixmap.isNull():
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    if qimage.isNull():
        raise ValueError(f"Не удалось конвертировать изображение: {path}")
    w = qimage.width()
    h = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.sizeInBytes())
    rgb = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
    return rgb[:, :, ::-1].copy()


def load_image_grayscale(path: str) -> np.ndarray:
    pixmap = QPixmap(path)
    if pixmap.isNull():
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    if qimage.isNull():
        raise ValueError(f"Не удалось конвертировать изображение: {path}")
    w = qimage.width()
    h = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.sizeInBytes())
    return np.frombuffer(ptr, dtype=np.uint8).reshape((h, w)).copy()
