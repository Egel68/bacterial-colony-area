from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_adapters import load_manifest
from .dataset_manifest import DatasetManifest, SampleRecord


class ColonyDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        data_root: Path,
        img_size: int = 512,
        augment: bool = False,
    ):
        self.records = records
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        image = cv2.imread(str(self.data_root / record.image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.data_root / record.mask), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = cv2.resize(
            image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )
        mask = mask[None, :, :]

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask

    def _augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.random() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        angle = np.random.uniform(-180, 180)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        mask = cv2.warpAffine(
            mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return image, mask


def _subsets_from_manifest(
    manifest: DatasetManifest, val_split: float, seed: int
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Разделение записей на train/val по манифесту или random split.

    Если записи уже содержат `subset`, разделение следует манифесту
    (например, для внешних датасетов). Иначе применяется воспроизводимое
    перемешивание от `seed` по `val_split`.
    """
    fixed_train = [s for s in manifest.samples if s.subset == "train"]
    fixed_val = [s for s in manifest.samples if s.subset == "val"]

    if fixed_train or fixed_val:
        if not fixed_train and not fixed_val:
            raise ValueError("Manifest has only test subset, no train/val")
        remaining = [
            s for s in manifest.samples if s.subset not in ("train", "val")
        ]
        if remaining:
            full = list(range(len(remaining)))
            rng = np.random.default_rng(seed)
            rng.shuffle(full)
            n_val = max(1, int(len(full) * val_split))
            for i, rec in enumerate(remaining):
                if i in full[:n_val]:
                    fixed_val.append(rec)
                else:
                    fixed_train.append(rec)
        return fixed_train, fixed_val

    full = list(range(len(manifest.samples)))
    rng = np.random.default_rng(seed)
    rng.shuffle(full)

    n_val = max(1, int(len(full) * val_split))
    val_idx = set(full[:n_val])
    train_records = [manifest.samples[i] for i in full[n_val:]]
    val_records = [manifest.samples[i] for i in val_idx]
    return train_records, val_records


def make_datasets(
    data_root: Path,
    img_size: int = 512,
    val_split: float = 0.2,
    seed: int = 42,
    augment: bool = True,
) -> tuple[Dataset, Dataset]:
    manifest = load_manifest(data_root)
    train_records, val_records = _subsets_from_manifest(manifest, val_split, seed)

    train_ds = ColonyDataset(train_records, data_root, img_size, augment=augment)
    val_ds = ColonyDataset(val_records, data_root, img_size, augment=False)
    return train_ds, val_ds