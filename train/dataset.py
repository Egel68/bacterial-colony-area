import zlib
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

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


class ColonyPatchDataset(Dataset):
    """Патчи нативного разрешения с seed-controlled sampling по снимку."""

    def __init__(
        self,
        records: list[SampleRecord],
        data_root: Path,
        patch_size: int = 512,
        patches_per_image: int = 4,
        seed: int = 42,
        augment: bool = False,
        training: bool = True,
    ):
        if patch_size <= 0 or patches_per_image <= 0:
            raise ValueError("patch_size and patches_per_image must be positive")
        self.records = list(records)
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.seed = seed
        self.augment = augment
        self.training = training
        self._epoch_value = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._cached_record_id: str | None = None
        self._cached_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._cached_foreground: tuple[np.ndarray, np.ndarray] | None = None
        self._cached_lowres_integral: tuple[np.ndarray, float] | None = None
        self.image_dimensions: dict[str, tuple[int, int]] = {}
        if len({record.id for record in self.records}) != len(self.records):
            raise ValueError("Patch dataset sample IDs must be unique")
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def set_epoch(self, epoch: int) -> None:
        self._epoch_value[0] = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_value[0].item())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index, patch_index = divmod(index, self.patches_per_image)
        record = self.records[record_index]
        image, mask = self._load_pair(record)
        y, x = self._crop_coordinates(record, patch_index)
        image_patch = image[y : y + self.patch_size, x : x + self.patch_size]
        mask_patch = mask[y : y + self.patch_size, x : x + self.patch_size]

        rng = self._rng(record, patch_index, purpose="augment")
        if self.augment:
            image_patch, mask_patch = self._augment(image_patch, mask_patch, rng)

        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        image_float = (image_float - _IMAGENET_MEAN) / _IMAGENET_STD
        image_tensor = torch.from_numpy(image_float.copy()).permute(2, 0, 1)
        mask_tensor = torch.from_numpy((mask_patch > 127).astype(np.float32))[None]
        return image_tensor, mask_tensor

    def _load_pair(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
        if record.id != self._cached_record_id:
            image_path = self.data_root / record.image
            mask_path = self.data_root / record.mask
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise ValueError(f"Could not read image/mask pair for '{record.id}'")
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Image/mask dimensions differ for '{record.id}'")
            self.image_dimensions[record.id] = (image.shape[1], image.shape[0])
            if min(image.shape[:2]) < self.patch_size:
                bottom = max(0, self.patch_size - image.shape[0])
                right = max(0, self.patch_size - image.shape[1])
                image = cv2.copyMakeBorder(
                    image, 0, bottom, 0, right, cv2.BORDER_REFLECT_101
                )
                mask = cv2.copyMakeBorder(
                    mask, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0
                )
            self._cached_record_id = record.id
            self._cached_pair = (image, mask)
            self._cached_foreground = None
            self._cached_lowres_integral = None
        assert self._cached_pair is not None
        return self._cached_pair

    def _crop_coordinates(
        self, record: SampleRecord, patch_index: int
    ) -> tuple[int, int]:
        image, mask = self._load_pair(record)
        max_y = image.shape[0] - self.patch_size
        max_x = image.shape[1] - self.patch_size
        if not self.training:
            if np.any(mask) and patch_index % 2 == 0:
                return self._foreground_coordinates(
                    record, patch_index, mask, max_y, max_x
                )
            return self._validation_background_coordinates(
                record, patch_index, mask, max_y, max_x
            )

        rng = self._rng(record, patch_index, purpose="crop")
        has_foreground = bool(np.any(mask))
        sample_foreground = has_foreground and (patch_index % 2 == 0)
        if sample_foreground:
            return self._foreground_coordinates(record, patch_index, mask, max_y, max_x)
        return int(rng.integers(max_y + 1)), int(rng.integers(max_x + 1))

    def _validation_background_coordinates(
        self,
        record: SampleRecord,
        patch_index: int,
        mask: np.ndarray,
        max_y: int,
        max_x: int,
    ) -> tuple[int, int]:
        """Детерминированно выбрать patch с минимальной площадью разметки."""
        if self._cached_lowres_integral is None:
            scale = min(1.0, 512 / max(mask.shape))
            if scale < 1.0:
                small_mask = cv2.resize(
                    mask,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small_mask = mask
            binary = (small_mask > 0).astype(np.uint8)
            integral = cv2.integral(binary, sdepth=cv2.CV_32S)
            self._cached_lowres_integral = (integral, scale)

        integral, scale = self._cached_lowres_integral
        rng = self._rng(record, patch_index, purpose="validation_background")
        best = (0, 0)
        best_coverage = None
        for _ in range(24):
            y = int(rng.integers(max_y + 1))
            x = int(rng.integers(max_x + 1))
            y1, x1 = int(y * scale), int(x * scale)
            y2 = min(
                integral.shape[0] - 1,
                int(np.ceil((y + self.patch_size) * scale)),
            )
            x2 = min(
                integral.shape[1] - 1,
                int(np.ceil((x + self.patch_size) * scale)),
            )
            coverage = int(
                integral[y2, x2]
                - integral[y1, x2]
                - integral[y2, x1]
                + integral[y1, x1]
            )
            if best_coverage is None or coverage < best_coverage:
                best, best_coverage = (y, x), coverage
                if coverage == 0:
                    break
        return best

    def _foreground_coordinates(
        self,
        record: SampleRecord,
        patch_index: int,
        mask: np.ndarray,
        max_y: int,
        max_x: int,
    ) -> tuple[int, int]:
        if self._cached_foreground is None:
            scale = min(1.0, 512 / max(mask.shape))
            if scale < 1.0:
                small = cv2.resize(
                    mask,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_NEAREST,
                )
                ys, xs = np.nonzero(small > 127)
                ys = np.minimum((ys / scale).astype(np.int32), mask.shape[0] - 1)
                xs = np.minimum((xs / scale).astype(np.int32), mask.shape[1] - 1)
            else:
                ys, xs = np.nonzero(mask > 127)
            self._cached_foreground = (ys, xs)
        ys, xs = self._cached_foreground
        if len(xs) == 0:
            rng = self._rng(record, patch_index, purpose="crop")
            return int(rng.integers(max_y + 1)), int(rng.integers(max_x + 1))
        if self.training:
            rng = self._rng(record, patch_index, purpose="crop")
            target = int(rng.integers(0, len(xs)))
        else:
            target = (zlib.crc32(record.id.encode("utf-8")) + patch_index) % len(xs)
        center_y, center_x = int(ys[target]), int(xs[target])
        y = int(np.clip(center_y - self.patch_size // 2, 0, max_y))
        x = int(np.clip(center_x - self.patch_size // 2, 0, max_x))
        return y, x

    def _rng(self, record: SampleRecord, patch_index: int, purpose: str):
        stable_id = zlib.crc32(record.id.encode("utf-8"))
        epoch = self.epoch if self.training else 0
        purpose_id = zlib.crc32(purpose.encode("utf-8"))
        seed = (
            self.seed + stable_id + epoch * 1_000_003 + patch_index * 9_176 + purpose_id
        ) % (2**32)
        return np.random.default_rng(seed)

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray, rng):
        if rng.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if rng.random() < 0.5:
            image, mask = np.flipud(image), np.flipud(mask)
        turns = int(rng.integers(0, 4))
        if turns:
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if rng.random() < 0.5:
            alpha = float(rng.uniform(0.85, 1.15))
            beta = float(rng.uniform(-16.0, 16.0))
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class GroupedPatchBatchSampler(Sampler[list[int]]):
    """Пакетирует все патчи одного снимка вместе, сохраняя перемешивание групп."""

    def __init__(self, dataset: ColonyPatchDataset, batch_size: int, seed: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.dataset)
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        if not self.dataset.training:
            indices = list(range(len(self.dataset)))
        else:
            rng = np.random.default_rng(self.seed + self.epoch)
            record_order = rng.permutation(len(self.dataset.records))
            patch_order = rng.permutation(self.dataset.patches_per_image)
            indices = [
                int(record_index) * self.dataset.patches_per_image + int(patch_index)
                for record_index in record_order
                for patch_index in patch_order
            ]
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]


def _subsets_from_manifest(
    manifest: DatasetManifest, val_split: float, seed: int
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Разделение записей на train/val по манифесту или random split.

    Если записи уже содержат `subset`, разделение следует манифесту
    (например, для внешних датасетов). Иначе применяется воспроизводимое
    перемешивание от `seed` по `val_split`.
    """
    explicit_subsets = {sample.subset for sample in manifest.samples if sample.subset}
    if explicit_subsets:
        fixed_train = [s for s in manifest.samples if s.subset == "train"]
        fixed_val = [s for s in manifest.samples if s.subset == "val"]
        if not fixed_train or not fixed_val:
            if explicit_subsets == {"test"}:
                raise ValueError("Manifest has only test subset, no train/val")
            raise ValueError("Manifest with explicit subsets needs both train and val")
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


def make_test_dataset(
    data_root: Path,
    img_size: int = 512,
    *,
    records: list[SampleRecord] | None = None,
) -> ColonyDataset:
    """Создать отдельный dataset только из явного subset=test."""
    if records is None:
        manifest = load_manifest(data_root)
        records = manifest.samples
    test_records = [record for record in records if record.subset == "test"]
    return ColonyDataset(test_records, data_root, img_size, augment=False)
