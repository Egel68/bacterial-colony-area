"""Проверки парного эксперимента и pixel-level confusion matrix."""

import cv2
import numpy as np
import pytest

from train.dataset_manifest import SampleRecord
from train.evaluation import binary_confusion_counts, summarize_confusion_counts
from train.background_experiment import compare_paired_models, write_comparison_html


def test_confusion_counts_orientation_and_exact_precision_recall():
    prediction = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    target = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    counts = binary_confusion_counts(prediction, target)
    summary = summarize_confusion_counts(counts)

    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert summary["confusion_matrix"] == [[1, 1], [1, 1]]
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5


def test_confusion_metrics_define_zero_denominators_and_shape_errors():
    counts = binary_confusion_counts(
        np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)
    )
    summary = summarize_confusion_counts(counts)

    assert summary["confusion_matrix"] == [[4, 0], [0, 0]]
    assert summary["precision"] == summary["recall"] == 0.0
    assert summary["accuracy"] == 1.0
    with pytest.raises(ValueError, match="shape mismatch"):
        binary_confusion_counts(np.zeros((2, 2)), np.zeros((3, 2)))


def _write_paired_records(root):
    records = []
    for sample_id in ("alpha", "beta"):
        mask_name = f"{sample_id}_mask.png"
        mask = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        cv2.imwrite(str(root / mask_name), mask)
        for variant, kind in (("original", "source"), ("black", "augmented")):
            image_name = f"{sample_id}_{variant}.png"
            cv2.imwrite(
                str(root / image_name),
                np.zeros((2, 2, 3), dtype=np.uint8),
            )
            records.append(
                SampleRecord(
                    id=(
                        sample_id
                        if kind == "source"
                        else f"{sample_id}__black_background"
                    ),
                    kind=kind,
                    image=image_name,
                    mask=mask_name,
                    subset="test",
                    source_id=sample_id if kind == "augmented" else None,
                )
            )
    return records


def test_paired_model_comparison_uses_same_ids_and_aggregates_pixels(
    tmp_path, monkeypatch
):
    import testing.tiled_onnx_algorithm as tiled

    records = _write_paired_records(tmp_path)
    baseline_path = tmp_path / "baseline.onnx"
    retrained_path = tmp_path / "retrained.onnx"
    baseline_path.write_bytes(b"baseline")
    retrained_path.write_bytes(b"retrained")
    calls = []

    class _FakeTiledAlgorithm:
        def __init__(self, model_path, **kwargs):
            self.model_path = model_path
            self.kwargs = kwargs

        def detect(self, image, is_cropped=False):
            calls.append((self.model_path, image.shape, is_cropped))
            if "baseline" in self.model_path:
                return np.zeros(image.shape[:2], dtype=np.uint8)
            return np.full(image.shape[:2], 255, dtype=np.uint8)

    monkeypatch.setattr(tiled, "TiledOnnxModelAlgorithm", _FakeTiledAlgorithm)
    result = compare_paired_models(
        baseline_path, retrained_path, records, tmp_path, tile_size=2, stride=2
    )

    assert result["baseline_training_overlap"] == "unknown"
    assert result["test_source_ids"] == ["alpha", "beta"]
    for model_name in ("baseline", "retrained"):
        model = result["models"][model_name]
        assert model["source_ids"] == result["test_source_ids"]
        for background in ("original", "black"):
            condition = model["backgrounds"][background]
            assert condition["source_ids"] == result["test_source_ids"]
            assert set(condition["per_sample"]) == set(result["test_source_ids"])
    assert result["models"]["baseline"]["backgrounds"]["original"]["summary"][
        "confusion_matrix"
    ] == [[4, 0], [4, 0]]
    assert result["models"]["retrained"]["backgrounds"]["black"]["summary"][
        "confusion_matrix"
    ] == [[0, 4], [0, 4]]
    assert (
        result["models"]["baseline"]["backgrounds"]["original"]["summary"]["precision"]
        == 0.0
    )
    assert (
        result["models"]["retrained"]["backgrounds"]["black"]["summary"]["recall"]
        == 1.0
    )
    assert result["models"]["baseline"]["backgrounds"]["original"]["per_sample"][
        "alpha"
    ]["confusion_matrix"] == [[2, 0], [2, 0]]
    assert result["models"]["retrained"]["backgrounds"]["black"]["per_sample"]["beta"][
        "confusion_matrix"
    ] == [[0, 2], [0, 2]]
    assert len(calls) == 8

    result.update(
        {
            "seed": 42,
            "split_ids": {
                "train": ["train-a"],
                "val": ["val-a"],
                "test": ["alpha", "beta"],
            },
            "configuration": {"architecture": "unet", "initialization": "random"},
            "source_provenance": {"unchanged": True},
        }
    )
    html_path = write_comparison_html(result, tmp_path / "comparison.html")
    page = html_path.read_text(encoding="utf-8")
    assert "alpha, beta" in page
    assert "[[4, 0], [4, 0]]" in page
    assert "Baseline training overlap: <strong>unknown</strong>" in page
    assert "source-level split ids" in page.lower()


def test_paired_comparison_fails_if_test_variant_is_missing(tmp_path, monkeypatch):
    from train.background_experiment import _test_pairs

    records = _write_paired_records(tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        _test_pairs(records[:-1])
