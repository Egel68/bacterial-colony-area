"""Проверки patch dataset для обучения на полном разрешении снимков."""

import pytest

pytest.importorskip("torch")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train.dataset import ColonyPatchDataset, GroupedPatchBatchSampler
from train.dataset_manifest import SampleRecord


def _write_pair(root, stem="dish", height=800, width=900):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), 35, 255, -1)
    image[mask > 0] = (0, 0, 255)
    image_path = root / f"{stem}.png"
    mask_path = root / f"{stem}_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return SampleRecord(stem, "source", image_path.name, mask_path.name, "train")


def test_patch_dataset_returns_aligned_full_resolution_tiles(tmp_path):
    record = _write_pair(tmp_path)
    dataset = ColonyPatchDataset(
        [record], tmp_path, patch_size=512, patches_per_image=4, seed=23, augment=False
    )

    image, mask = dataset[0]

    assert image.shape == (3, 512, 512)
    assert mask.shape == (1, 512, 512)
    assert image.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    # Исходная ROI-сигнатура в красном канале точно совпадает с бинарной маской.
    expected = (image[0] > 0).float()
    assert torch.equal(expected, mask[0])


def test_patch_indices_keep_their_source_record_mapping(tmp_path):
    records = []
    colors = [(10, 20, 30), (200, 180, 160)]  # BGR
    for index, color in enumerate(colors):
        image = np.full((64, 64, 3), color, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"source-{index}.png"), image)
        cv2.imwrite(str(tmp_path / f"source-{index}_mask.png"), mask)
        records.append(
            SampleRecord(
                f"source-{index}",
                "source",
                f"source-{index}.png",
                f"source-{index}_mask.png",
                "train",
            )
        )
    dataset = ColonyPatchDataset(
        records,
        tmp_path,
        patch_size=32,
        patches_per_image=1,
        seed=5,
        augment=False,
    )

    first_image, _ = dataset[0]
    second_image, _ = dataset[1]
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    expected_first = (np.asarray(colors[0][::-1], dtype=np.float32) / 255 - mean) / std
    expected_second = (np.asarray(colors[1][::-1], dtype=np.float32) / 255 - mean) / std

    assert np.allclose(first_image[:, 0, 0].numpy(), expected_first)
    assert np.allclose(second_image[:, 0, 0].numpy(), expected_second)


def test_patch_selection_is_repeatable_for_seed_and_epoch(tmp_path):
    record = _write_pair(tmp_path)
    dataset = ColonyPatchDataset(
        [record], tmp_path, patch_size=256, patches_per_image=8, seed=9, augment=False
    )

    first = [dataset._crop_coordinates(record, patch_index) for patch_index in range(8)]
    repeated = [
        dataset._crop_coordinates(record, patch_index) for patch_index in range(8)
    ]
    dataset.set_epoch(1)
    next_epoch = [
        dataset._crop_coordinates(record, patch_index) for patch_index in range(8)
    ]

    assert first == repeated
    assert first != next_epoch


def test_validation_coordinates_do_not_change_with_epoch(tmp_path):
    record = _write_pair(tmp_path)
    dataset = ColonyPatchDataset(
        [record],
        tmp_path,
        patch_size=256,
        patches_per_image=6,
        seed=3,
        augment=False,
        training=False,
    )

    first = [dataset._crop_coordinates(record, i) for i in range(len(dataset))]
    dataset.set_epoch(10)

    assert [dataset._crop_coordinates(record, i) for i in range(len(dataset))] == first


def test_validation_sampling_includes_positive_and_low_label_background_patch(tmp_path):
    height, width = 1200, 1400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), 45, 255, -1)
    image_path = tmp_path / "validation.png"
    mask_path = tmp_path / "validation_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    record = SampleRecord(
        "validation", "source", image_path.name, mask_path.name, "val"
    )
    dataset = ColonyPatchDataset(
        [record],
        tmp_path,
        patch_size=256,
        patches_per_image=2,
        seed=37,
        augment=False,
        training=False,
    )

    positive_origin = dataset._crop_coordinates(record, 0)
    background_origin = dataset._crop_coordinates(record, 1)
    y, x = background_origin

    assert np.any(
        mask[
            positive_origin[0] : positive_origin[0] + 256,
            positive_origin[1] : positive_origin[1] + 256,
        ]
    )
    assert np.count_nonzero(mask[y : y + 256, x : x + 256]) == 0


def test_grouped_sampler_visits_every_patch_and_keeps_image_locality():
    records = [
        SampleRecord(f"dish-{index}", "source", "unused", "unused_mask", "train")
        for index in range(5)
    ]
    dataset = ColonyPatchDataset(
        records,
        ".",
        patch_size=32,
        patches_per_image=4,
        seed=21,
        augment=False,
    )
    sampler = GroupedPatchBatchSampler(dataset, batch_size=8, seed=21)

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == list(range(len(dataset)))
    assert len(batches[-1]) == 4
    assert all(len(set(index // 4 for index in batch)) <= 2 for batch in batches)


def test_persistent_workers_observe_epoch_changes(tmp_path):
    height, width = 800, 900
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), 45, 255, -1)
    image_path = tmp_path / "gradient.png"
    mask_path = tmp_path / "gradient_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    record = SampleRecord(
        "gradient", "source", image_path.name, mask_path.name, "train"
    )
    dataset = ColonyPatchDataset(
        [record], tmp_path, patch_size=256, patches_per_image=8, seed=13, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    first_epoch = torch.cat([images for images, _ in loader])
    dataset.set_epoch(1)
    second_epoch = torch.cat([images for images, _ in loader])
    del loader

    assert not torch.equal(first_epoch, second_epoch)


def test_validation_first_patch_is_anchored_to_foreground(tmp_path):
    record = _write_pair(tmp_path)
    dataset = ColonyPatchDataset(
        [record],
        tmp_path,
        patch_size=128,
        patches_per_image=2,
        seed=3,
        augment=False,
        training=False,
    )

    y, x = dataset._crop_coordinates(record, 0)
    _, mask = dataset._load_pair(record)

    assert np.any(mask[y : y + dataset.patch_size, x : x + dataset.patch_size] > 127)
