"""Проверки разделения записей обучения (нужен full ML extra)."""

import pytest

pytest.importorskip("torch")

from train.dataset import _subsets_from_manifest, make_test_dataset
from train.dataset_manifest import DatasetManifest, SampleRecord


def test_explicit_test_records_are_not_folded_into_train_or_validation(tmp_path):
    records = [
        SampleRecord("train-1", "source", "train-1.png", "train-1_mask.png", "train"),
        SampleRecord("train-2", "source", "train-2.png", "train-2_mask.png", "train"),
        SampleRecord("val-1", "source", "val-1.png", "val-1_mask.png", "val"),
        SampleRecord("test-1", "source", "test-1.png", "test-1_mask.png", "test"),
        SampleRecord("test-2", "source", "test-2.png", "test-2_mask.png", "test"),
    ]
    manifest = DatasetManifest("fixture", "external", "binary", "reference", records)

    train_records, val_records = _subsets_from_manifest(manifest, val_split=0.9, seed=1)
    test_dataset = make_test_dataset(tmp_path, records=records)

    assert [record.id for record in train_records] == ["train-1", "train-2"]
    assert [record.id for record in val_records] == ["val-1"]
    assert [record.id for record in test_dataset.records] == ["test-1", "test-2"]


def test_fixed_test_subset_without_train_and_val_raises():
    manifest = DatasetManifest(
        "fixture",
        "external",
        "binary",
        "reference",
        [SampleRecord("test", "source", "test.png", "test_mask.png", "test")],
    )

    with pytest.raises(ValueError, match="only test subset"):
        _subsets_from_manifest(manifest, val_split=0.2, seed=42)
