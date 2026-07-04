from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import cv2
import numpy as np


class TestSample(NamedTuple):
    name: str
    source_image: np.ndarray
    source_mask: np.ndarray
    cropped_image: Optional[np.ndarray]
    cropped_mask: Optional[np.ndarray]
    source_path: str
    cropped_path: Optional[str]


class TestDataset:
    def __init__(self, root: str = "test_images"):
        self.root = Path(root)
        self.source_dir = self.root / "source"
        self.masks_dir = self.root / "masks"
        self.cropped_dir = self.root / "cropped"
        self.cropped_masks_dir = self.root / "cropped_masks"

        self.samples: list[TestSample] = []
        self._load()

    def _find_pair(self, stem: str, ext: str, directory: Path) -> Optional[str]:
        """Ищет файл {stem}_mask.{ext} или {stem}{ext} в указанной директории."""
        for suffix in ["_mask", "_cropped", "_cropped_mask", ""]:
            candidate = directory / f"{stem}{suffix}{ext}"
            if candidate.exists():
                return str(candidate)

        alt = directory / f"{stem}{ext}"
        if alt.exists():
            return str(alt)
        return None

    def _load(self):
        if not self.source_dir.exists():
            return

        for source_path in sorted(self.source_dir.iterdir()):
            if not source_path.is_file():
                continue

            stem = source_path.stem
            ext = source_path.suffix

            source_img = cv2.imread(str(source_path))
            if source_img is None:
                continue

            mask_path = self.masks_dir / f"{stem}_mask{ext}"
            if not mask_path.exists():
                continue
            source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if source_mask is None:
                continue

            cropped_img = None
            cropped_mask = None
            cropped_path = None

            cropped_path_candidate = self.cropped_dir / f"{stem}_cropped{ext}"
            cropped_mask_path = self.cropped_masks_dir / f"{stem}_cropped_mask{ext}"

            if cropped_path_candidate.exists() and cropped_mask_path.exists():
                cropped_img = cv2.imread(str(cropped_path_candidate))
                cropped_mask = cv2.imread(str(cropped_mask_path), cv2.IMREAD_GRAYSCALE)
                cropped_path = str(cropped_path_candidate)

            self.samples.append(
                TestSample(
                    name=stem,
                    source_image=source_img,
                    source_mask=source_mask,
                    cropped_image=cropped_img,
                    cropped_mask=cropped_mask,
                    source_path=str(source_path),
                    cropped_path=cropped_path,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TestSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[TestSample]:
        return iter(self.samples)
