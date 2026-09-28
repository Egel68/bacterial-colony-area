from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import cv2
import numpy as np


class TestSample(NamedTuple):
    name: str
    source_image: Optional[np.ndarray]
    source_mask: Optional[np.ndarray]
    cropped_image: Optional[np.ndarray]
    cropped_mask: Optional[np.ndarray]
    source_path: str
    source_mask_path: str
    cropped_path: Optional[str]
    cropped_mask_path: Optional[str]


class TestDataset:
    __test__ = False

    def __init__(
        self,
        root: str = "test_images",
        sample_limit: int | None = None,
        load_images: bool = True,
    ):
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        self.root = Path(root)
        self.sample_limit = sample_limit
        self.load_images = load_images
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

        loaded_sources = 0
        for source_path in sorted(self.source_dir.iterdir()):
            if not source_path.is_file():
                continue

            stem = source_path.stem
            ext = source_path.suffix

            mask_path = self.masks_dir / f"{stem}_mask{ext}"
            if not mask_path.exists():
                continue
            source_img = cv2.imread(str(source_path)) if self.load_images else None
            source_mask = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if self.load_images
                else None
            )
            if self.load_images and (source_img is None or source_mask is None):
                continue

            loaded_sources += 1

            cropped_img = None
            cropped_mask = None
            cropped_path = None
            cropped_mask_path = None

            cropped_path_candidate = self.cropped_dir / f"{stem}_cropped{ext}"
            cropped_mask_path_candidate = (
                self.cropped_masks_dir / f"{stem}_cropped_mask{ext}"
            )

            if cropped_path_candidate.exists() and cropped_mask_path_candidate.exists():
                if self.load_images:
                    cropped_img = cv2.imread(str(cropped_path_candidate))
                    cropped_mask = cv2.imread(
                        str(cropped_mask_path_candidate), cv2.IMREAD_GRAYSCALE
                    )
                cropped_path = str(cropped_path_candidate)
                cropped_mask_path = str(cropped_mask_path_candidate)

            self.samples.append(
                TestSample(
                    name=stem,
                    source_image=source_img,
                    source_mask=source_mask,
                    cropped_image=cropped_img,
                    cropped_mask=cropped_mask,
                    source_path=str(source_path),
                    source_mask_path=str(mask_path),
                    cropped_path=cropped_path,
                    cropped_mask_path=cropped_mask_path,
                )
            )
            if self.sample_limit is not None and loaded_sources >= self.sample_limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TestSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[TestSample]:
        return iter(self.samples)
