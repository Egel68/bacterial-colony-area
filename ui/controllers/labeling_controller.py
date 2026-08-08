import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from analysis.colony_detector import ColonyDetector
from analysis.geometry import PetriInfo
from utils.image_loader import SUPPORTED_EXTENSIONS, load_image, load_image_grayscale


class LabelingController:
    """Бизнес-логика разметки без зависимостей от Qt."""

    def __init__(self, detector: Optional[ColonyDetector] = None):
        self._detector = detector or ColonyDetector()

    def load_image_rgb(self, path: str) -> np.ndarray:
        image_bgr = load_image(path)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def load_mask(
        self, path: Path, expected_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        try:
            mask = load_image_grayscale(str(path))
            if mask.shape == expected_shape:
                return mask
        except ValueError:
            pass
        return None

    def detect_petri(self, image_bgr: np.ndarray) -> Optional[PetriInfo]:
        _, info = self._detector.detect_petri_dish(image_bgr)
        return info

    def crop_by_petri(
        self,
        image_bgr: np.ndarray,
        petri_info: PetriInfo,
    ) -> np.ndarray:
        cx, cy = petri_info.cx, petri_info.cy
        r = petri_info.radius
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(image_bgr.shape[1], cx + r)
        y2 = min(image_bgr.shape[0], cy + r)
        cropped = image_bgr[y1:y2, x1:x2]
        circle_center = (cx - x1, cy - y1)
        circle_mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, circle_center, r, 255, -1)
        cropped[circle_mask == 0] = [0, 0, 0]
        return cropped

    def save_mask(self, mask: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask)

    def export_session_to_zip(self, session_dir: Path, output_path: Path) -> None:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_dir, dirs, files in os.walk(str(session_dir)):
                for file in files:
                    full = Path(root_dir) / file
                    arcname = full.relative_to(session_dir.parent)
                    zf.write(str(full), str(arcname))

    def get_current_dir(self, session_dir: Path, mode: str) -> Path:
        if mode == "source":
            sub = session_dir / "source"
            if sub.is_dir() and any(
                f.suffix.lower() in SUPPORTED_EXTENSIONS for f in sub.iterdir()
            ):
                return sub
            return session_dir
        sub = session_dir / "cropped"
        if sub.is_dir() and any(
            f.suffix.lower() in SUPPORTED_EXTENSIONS for f in sub.iterdir()
        ):
            return sub
        return session_dir

    def get_mask_dir(self, session_dir: Path, mode: str) -> Path:
        if mode == "source":
            sub = session_dir / "masks"
            return sub if sub.is_dir() else session_dir / "masks"
        sub = session_dir / "cropped_masks"
        return sub if sub.is_dir() else session_dir / "cropped_masks"

    def list_image_files(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            f
            for f in directory.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
