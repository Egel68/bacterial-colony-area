import cv2
import numpy as np
import pytest

from testing.classic_algorithms import (
    ClassicDefault,
    ClassicHighSensitivity,
    ClassicSolidFill,
    ClassicLowSensitivity,
)
from testing.dataset import TestDataset
from testing.metrics import compute_segmentation_metrics
from testing.runner import run_algorithm, _mean_metrics


class TestAlgorithmsRun:
    @pytest.mark.parametrize("algo_cls", [
        ClassicDefault,
        ClassicHighSensitivity,
        ClassicSolidFill,
        ClassicLowSensitivity,
    ])
    def test_each_returns_mask(self, algo_cls, test_source_paths):
        if not test_source_paths:
            pytest.skip("no test images available")
        image = cv2.imread(str(test_source_paths[0]))
        if image is None:
            pytest.skip("could not load test image")
        algo = algo_cls()
        mask = algo.detect(image, is_cropped=False)
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.dtype == np.uint8
        assert mask.shape[:2] == image.shape[:2]


@pytest.mark.slow
class TestAlgorithmsOnDataset:
    def test_metrics_nonzero(self):
        dataset = TestDataset()
        if len(dataset) == 0:
            pytest.skip("no test dataset found")
        algo = ClassicDefault()
        results = run_algorithm(algo, dataset)
        all_metrics = []
        for sample_key, variants in results.items():
            for variant_key, metrics in variants.items():
                all_metrics.append(metrics)
        means = _mean_metrics(all_metrics)
        assert "mean_iou" in means
        assert "mean_f1" in means
        assert means["mean_f1"] > 0


class TestMeanMetrics:
    def test_empty_list(self):
        assert _mean_metrics([]) == {}

    def test_single_metric(self):
        m = {"iou": 0.5, "dice": 0.6, "tp": 10, "fp": 5, "fn": 5, "tn": 80}
        result = _mean_metrics([m])
        assert result["mean_iou"] == pytest.approx(0.5)
        assert result["std_iou"] == pytest.approx(0.0)
        assert "mean_tp" not in result

    def test_multiple_metrics(self):
        metrics = [
            {"iou": 0.5, "dice": 0.6, "tp": 10, "fp": 5, "fn": 5, "tn": 80},
            {"iou": 0.7, "dice": 0.8, "tp": 20, "fp": 3, "fn": 2, "tn": 75},
        ]
        result = _mean_metrics(metrics)
        assert result["mean_iou"] == pytest.approx(0.6)
        assert result["std_iou"] == pytest.approx(0.1)
