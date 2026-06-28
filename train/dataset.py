from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class ColonyDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        img_size: int = 512,
        augment: bool = False,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask[None, :, :]

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.random() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        angle = np.random.uniform(-180, 180)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, mask


def make_datasets(
    data_root: Path,
    img_size: int = 512,
    val_split: float = 0.2,
    seed: int = 42,
    augment: bool = True,
) -> tuple[Dataset, Dataset]:
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    image_paths = sorted(images_dir.glob("*.png"))
    mask_paths = [masks_dir / p.name for p in image_paths]
    valid = [(img, mask) for img, mask in zip(image_paths, mask_paths) if mask.exists()]
    if not valid:
        raise FileNotFoundError(f"No image-mask pairs found in {data_root}")
    image_paths, mask_paths = zip(*valid)
    image_paths = list(image_paths)
    mask_paths = list(mask_paths)

    full = list(range(len(image_paths)))
    rng = np.random.default_rng(seed)
    rng.shuffle(full)

    n_val = max(1, int(len(full) * val_split))
    val_idx = set(full[:n_val])
    train_idx = set(full[n_val:])

    train_img = [image_paths[i] for i in train_idx]
    train_mask = [mask_paths[i] for i in train_idx]
    val_img = [image_paths[i] for i in val_idx]
    val_mask = [mask_paths[i] for i in val_idx]

    train_ds = ColonyDataset(train_img, train_mask, img_size, augment=augment)
    val_ds = ColonyDataset(val_img, val_mask, img_size, augment=False)
    return train_ds, val_ds
