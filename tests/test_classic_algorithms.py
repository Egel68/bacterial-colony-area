import cv2
import numpy as np
import pytest

from testing.classic_algorithms import (
    ClassicDefault,
    ClassicHighSensitivity,
    ClassicSolidFill,
    ClassicLowSensitivity,
)
from testing.dashboard import generate_report
from testing.dataset import TestDataset
from testing.runner import run_algorithm, _mean_metrics

_SAMPLE_METRICS = {
    "iou": 0.9,
    "dice": 0.95,
    "f1": 0.95,
    "precision": 1.0,
    "recall": 0.9,
    "accuracy": 0.99,
    "tp": 90,
    "fp": 0,
    "fn": 10,
    "tn": 900,
}


class TestAlgorithmsRun:
    @pytest.mark.parametrize(
        "algo_cls",
        [
            ClassicDefault,
            ClassicHighSensitivity,
            ClassicSolidFill,
            ClassicLowSensitivity,
        ],
    )
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

    @pytest.mark.parametrize(
        "algo_cls",
        [
            ClassicDefault,
            ClassicHighSensitivity,
            ClassicSolidFill,
            ClassicLowSensitivity,
        ],
    )
    def test_each_returns_cropped_mask(self, algo_cls):
        dataset = TestDataset()
        if len(dataset) == 0:
            pytest.skip("no test dataset found")
        sample = dataset[0]
        if sample.cropped_image is None or sample.cropped_mask is None:
            pytest.skip("no cropped pair found")
        algo = algo_cls()
        mask = algo.detect(sample.cropped_image, is_cropped=True)
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.dtype == np.uint8
        assert mask.shape[:2] == sample.cropped_image.shape[:2]


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


class TestGenerateReport:
    def test_report_created(self, tmp_path):
        all_results = {
            "ClassicDefault": {
                "sample1": {
                    "source": {
                        "iou": 0.9,
                        "dice": 0.95,
                        "f1": 0.95,
                        "precision": 1.0,
                        "recall": 0.9,
                        "accuracy": 0.99,
                        "tp": 90,
                        "fp": 0,
                        "fn": 10,
                        "tn": 900,
                    }
                }
            }
        }
        out = str(tmp_path / "report.html")
        generate_report(all_results, output_path=out)
        text = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "ClassicDefault" in text
        assert "0.9000" in text

    def test_report_without_per_snapshot_hides_tables(self, tmp_path):
        all_results = {
            "ClassicDefault": {
                "sample1": {
                    "source": _SAMPLE_METRICS,
                }
            }
        }
        out = str(tmp_path / "report.html")
        generate_report(all_results, output_path=out, include_per_snapshot=False)
        text = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "sample1" not in text
        assert "ClassicDefault" in text

    def test_report_with_comparison_includes_winner_table(self, tmp_path):
        all_results = {
            "ClassicDefault": {"sample1": {"source": _SAMPLE_METRICS}},
            "ClassicSolidFill": {"sample1": {"source": _SAMPLE_METRICS}},
        }
        comparison = {
            "iou": {
                "sample1": {"source": "a"},
            },
            "dice": {
                "sample1": {"source": "tie"},
            },
        }
        out = str(tmp_path / "report.html")
        generate_report(
            all_results,
            output_path=out,
            include_per_snapshot=False,
            comparison=comparison,
        )
        text = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "Сравнение: IoU" in text
        assert "<b>A</b>" in text
        assert "<b>Ничья</b>" in text
