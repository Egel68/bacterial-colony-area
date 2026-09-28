import pytest
import numpy as np

from train.dataset_split import DEFAULT_SPLIT_RATIOS, stratified_group_split
from train.evaluation import compute_binary_metrics


def test_split_is_stable_when_input_order_changes():
    labels = {
        f"class-{category}-image-{image}": category
        for category in range(4)
        for image in range(20)
    }

    first = stratified_group_split(labels, seed=17)
    second = stratified_group_split(dict(reversed(list(labels.items()))), seed=17)

    assert first == second


def test_split_uses_all_subsets_and_approximately_requested_proportions():
    labels = {f"image-{image}": image % 3 for image in range(99)}

    split = stratified_group_split(labels, seed=42)

    counts = {
        name: list(split.values()).count(name) for name in ("train", "val", "test")
    }
    assert set(split) == set(labels)
    assert set(split.values()) == {"train", "val", "test"}
    assert counts == {"train": 69, "val": 15, "test": 15}


def test_rare_strata_are_distributed_across_subsets_where_possible():
    labels = {
        "rare-a": "rare",
        "rare-b": "rare",
        "rare-c": "rare",
        "common-a": "common",
        "common-b": "common",
        "common-c": "common",
        "common-d": "common",
    }

    split = stratified_group_split(labels, seed=42)

    assert {split[f"rare-{letter}"] for letter in "abc"} == {"train", "val", "test"}
    assert set(split.values()) == {"train", "val", "test"}


@pytest.mark.parametrize(
    "ratios",
    [
        (0.0, 0.2, 0.8),
        (0.7, 0.2, 0.2),
        (0.7, -0.1, 0.4),
    ],
)
def test_split_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError, match="ratios"):
        stratified_group_split({f"image-{i}": i % 2 for i in range(8)}, ratios=ratios)


def test_split_rejects_empty_or_too_small_dataset():
    with pytest.raises(ValueError, match="at least three"):
        stratified_group_split({"one": "a", "two": "a"})

    with pytest.raises(ValueError, match="empty"):
        stratified_group_split({})


def test_split_rejects_unknown_or_duplicate_ids_after_string_conversion():
    with pytest.raises(ValueError, match="unique"):
        stratified_group_split({1: "a", "1": "b", 2: "c"})


def test_default_split_ratios_are_70_15_15():
    assert DEFAULT_SPLIT_RATIOS == (0.70, 0.15, 0.15)


def test_binary_metrics_report_segmentation_quality():
    target = np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
    prediction = np.asarray([[0, 255], [0, 255]], dtype=np.uint8)

    metrics = compute_binary_metrics(prediction, target)

    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
