import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from testing.metrics import compute_segmentation_metrics


def test_perfect_match():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    m = compute_segmentation_metrics(mask, mask)
    assert m["iou"] == pytest.approx(1.0)
    assert m["dice"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_no_match():
    pred = np.zeros((100, 100), dtype=np.uint8)
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:80, 20:80] = 255
    m = compute_segmentation_metrics(pred, gt)
    assert m["iou"] == pytest.approx(0.0)
    assert m["dice"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)


def test_partial_match():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:80, 20:80] = 255
    pred = np.zeros_like(gt)
    pred[40:90, 40:90] = 255
    m = compute_segmentation_metrics(pred, gt)
    assert 0 < m["iou"] < 1
    assert 0 < m["dice"] < 1
    assert 0 < m["f1"] < 1


def test_empty_both():
    mask = np.zeros((50, 50), dtype=np.uint8)
    m = compute_segmentation_metrics(mask, mask)
    assert m["iou"] >= 0
    assert m["dice"] >= 0
    assert m["accuracy"] >= 0


def test_full_both():
    mask = np.ones((50, 50), dtype=np.uint8) * 255
    m = compute_segmentation_metrics(mask, mask)
    assert m["iou"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_symmetry():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:80, 20:80] = 255
    pred = np.zeros_like(gt)
    pred[40:90, 40:90] = 255
    m1 = compute_segmentation_metrics(pred, gt)
    m2 = compute_segmentation_metrics(gt, pred)
    assert m1["iou"] == pytest.approx(m2["iou"])
    assert m1["dice"] == pytest.approx(m2["dice"])
    assert m1["f1"] == pytest.approx(m2["f1"])


@given(st.integers(10, 100), st.integers(10, 100))
@settings(max_examples=50)
def test_metrics_bounds_property(h, w):
    pred = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    gt = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    m = compute_segmentation_metrics(pred, gt)
    for key in ("iou", "dice", "f1", "precision", "recall", "accuracy"):
        assert 0 <= m[key] <= 1, f"{key}={m[key]} out of [0,1]"
    assert m["iou"] <= m["dice"] + 1e-10
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == h * w
