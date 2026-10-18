"""ONNX-адаптер с полноразмерной tiled сегментацией для компактной модели."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .interface import BaseDetectionAlgorithm

ProgressCallback = Callable[[int, int], None]


def tile_positions(length: int, tile_size: int = 512, stride: int = 384) -> list[int]:
    if length <= 0:
        raise ValueError("Image dimensions must be positive")
    if tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("tile_size and stride must be positive; stride <= tile_size")
    if length <= tile_size:
        return [0]
    return list(range(0, length, stride))


class TiledOnnxModelAlgorithm(BaseDetectionAlgorithm):
    """CPU-only ONNX inference returning full-resolution binary colony mask."""

    def __init__(
        self,
        model_path: str,
        name: str = "",
        description: str = "",
        img_size: int = 512,
        stride: int = 384,
        threshold: float = 0.5,
        output_is_logits: bool = True,
    ):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model_path = str(model_path)
        self.img_size = img_size
        self.stride = stride
        self.threshold = threshold
        self.output_is_logits = output_is_logits
        self._name = name or Path(model_path).stem
        self._description = (
            description or f"Tiled CPU ONNX-модель: {Path(model_path).name}"
        )
        self._session = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime
            except ImportError as error:
                raise RuntimeError(
                    "onnxruntime CPU не установлен. Установите runtime приложения."
                ) from error
            available = onnxruntime.get_available_providers()
            if "CPUExecutionProvider" not in available:
                raise RuntimeError("onnxruntime CPUExecutionProvider недоступен")
            # Лимиты потоков ONNX Runtime — из общей ресурсной политики (задача 2.6).
            from .resource_policy import onnx_session_options

            options = onnx_session_options()
            if options is None:
                options = onnxruntime.SessionOptions()
            self._session = onnxruntime.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def detect(
        self,
        image: np.ndarray,
        is_cropped: bool = False,
        context=None,
    ) -> np.ndarray:
        return self.detect_with_progress(image, is_cropped=is_cropped)

    def detect_with_progress(
        self,
        image: np.ndarray,
        is_cropped: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        del is_cropped  # Tiled model covers the full supplied frame in both cases.
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        height, width = image.shape[:2]
        ys = tile_positions(height, self.img_size, self.stride)
        xs = tile_positions(width, self.img_size, self.stride)
        total = len(ys) * len(xs)
        probability_sum = np.zeros((height, width), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.uint16)

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        completed = 0
        for y in ys:
            for x in xs:
                y2, x2 = min(height, y + self.img_size), min(width, x + self.img_size)
                tile = image[y:y2, x:x2]
                bottom = self.img_size - tile.shape[0]
                right = self.img_size - tile.shape[1]
                if bottom or right:
                    border_type = (
                        cv2.BORDER_REFLECT_101
                        if tile.shape[0] > 1 and tile.shape[1] > 1
                        else cv2.BORDER_REPLICATE
                    )
                    tile = cv2.copyMakeBorder(tile, 0, bottom, 0, right, border_type)

                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                normalized = (rgb - mean) / std
                tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
                raw = session.run(None, {input_name: tensor})[0]
                raw = np.asarray(raw)[0, 0]
                probability = self._probabilities(
                    raw, output_is_logits=self.output_is_logits
                )
                patch_h, patch_w = y2 - y, x2 - x
                probability_sum[y:y2, x:x2] += probability[:patch_h, :patch_w]
                coverage[y:y2, x:x2] += 1
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        if np.any(coverage == 0):
            raise RuntimeError("Tiled inference left pixels uncovered")
        probability_sum /= coverage
        return (probability_sum >= self.threshold).astype(np.uint8) * 255

    @staticmethod
    def _probabilities(raw: np.ndarray, *, output_is_logits: bool) -> np.ndarray:
        raw = raw.astype(np.float32, copy=False)
        if output_is_logits:
            return 1.0 / (1.0 + np.exp(-np.clip(raw, -80, 80)))
        return np.clip(raw, 0.0, 1.0)
