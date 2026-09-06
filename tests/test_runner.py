import numpy as np
import pytest

from testing.interface import BaseDetectionAlgorithm
from testing.registry import register_algorithm_instance
from testing.runner import run_all, compare_algorithms


class _FakeBase(BaseDetectionAlgorithm):
    name = ""
    description = ""
    fill = 0

    def detect(self, image, is_cropped=False):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:20, 10:20] = 255 if self.fill == 1 else 0
        return mask


class _AlgoA(_FakeBase):
    name = "AlgoA"
    description = "Обнаруживает всё"
    fill = 1


class _AlgoB(_FakeBase):
    name = "AlgoB"
    description = "Не обнаруживает ничего"
    fill = 0


@pytest.fixture
def paired_dataset(tmp_path):
    """Датасет из одного снимка с source и cropped парами."""
    source_dir = tmp_path / "source"
    masks_dir = tmp_path / "masks"
    cropped_dir = tmp_path / "cropped"
    cropped_masks_dir = tmp_path / "cropped_masks"
    for d in (source_dir, masks_dir, cropped_dir, cropped_masks_dir):
        d.mkdir(parents=True)

    import cv2

    img = np.ones((40, 40, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(source_dir / "s1.png"), img)
    cv2.imwrite(str(cropped_dir / "s1_cropped.png"), img)

    # GT: колония квадратом 20×20 в центре
    gt = np.zeros((40, 40), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    cv2.imwrite(str(masks_dir / "s1_mask.png"), gt)
    cv2.imwrite(str(cropped_masks_dir / "s1_cropped_mask.png"), gt)

    from testing.dataset import TestDataset

    return TestDataset(root=str(tmp_path))


@pytest.fixture(autouse=True)
def _register_algo():
    register_algorithm_instance("AlgoA", _AlgoA())
    register_algorithm_instance("AlgoB", _AlgoB())


class TestRunAll:
    def test_runs_all_when_algorithms_none(self, paired_dataset):
        results = run_all(paired_dataset)
        assert "AlgoA" in results
        assert "AlgoB" in results

    def test_runs_subset(self, paired_dataset):
        results = run_all(paired_dataset, algorithms=["AlgoA"])
        assert "AlgoA" in results
        assert "AlgoB" not in results


class TestCompareAlgorithms:
    def test_a_should_beat_b_on_overlap(self, paired_dataset):
        comparison = compare_algorithms(paired_dataset, "AlgoA", "AlgoB")
        # AlgoA выдаёт полную маску => IoU выше, чем пустая.
        assert comparison["iou"]["s1"]["source"] == "a"
        assert comparison["dice"]["s1"]["source"] == "a"

    def test_b_never_wins(self, paired_dataset):
        comparison = compare_algorithms(paired_dataset, "AlgoA", "AlgoB")
        for metric in comparison:
            for sample in comparison[metric]:
                for winner in comparison[metric][sample].values():
                    assert winner in ("a", "tie")

    def test_cropped_variant_compared_in_variant(self, paired_dataset):
        comparison = compare_algorithms(paired_dataset, "AlgoA", "AlgoB")
        assert "cropped" in comparison["iou"]["s1"]
        assert comparison["iou"]["s1"]["cropped"] == "a"
