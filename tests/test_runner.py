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

    def test_threaded_runner_reuses_loaded_arrays(self, paired_dataset, monkeypatch):
        import testing.scheduler as scheduler
        from testing.dataset import TestDataset

        paired_dataset = TestDataset(
            root=str(paired_dataset.root),
            load_images=False,
        )

        reads = []
        original = scheduler._decode_file

        def counting_decode(*args, **kwargs):
            reads.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(scheduler, "_decode_file", counting_decode)

        results = run_all(
            paired_dataset,
            algorithms=["AlgoA", "AlgoB"],
            workers=2,
            batch_size=1,
        )

        assert set(results) == {"AlgoA", "AlgoB"}
        assert len(reads) == 4

    def test_threaded_runner_matches_sequential_results(self, paired_dataset):
        from testing.runner import _run_algorithm_seq
        from testing.registry import get_algorithm

        sequential = {
            name: _run_algorithm_seq(get_algorithm(name), paired_dataset)
            for name in ("AlgoA", "AlgoB")
        }
        parallel = run_all(
            paired_dataset,
            algorithms=["AlgoA", "AlgoB"],
            workers=2,
            batch_size=1,
        )

        assert parallel == sequential

    def test_manifest_pairs_under_unicode_path_run_successfully(self, tmp_path):
        import json

        import cv2

        from testing.baseline import BaselineDataset

        root = tmp_path / "Егор" / "22022540_imported"
        source = root / "source"
        source.mkdir(parents=True)
        image_path = source / "sample.jpg"
        mask_path = source / "sample_mask.png"
        image = np.full((24, 20, 3), 92, dtype=np.uint8)
        mask = np.zeros((24, 20), dtype=np.uint8)

        for path, array, extension in (
            (image_path, image, ".jpg"),
            (mask_path, mask, ".png"),
        ):
            ok, encoded = cv2.imencode(extension, array)
            assert ok
            path.write_bytes(encoded.tobytes())

        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "id": "sample",
                            "kind": "source",
                            "image": "source/sample.jpg",
                            "mask": "source/sample_mask.png",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        dataset = BaselineDataset(root)
        assert len(dataset) == 1
        assert dataset.samples[0].image_path == image_path
        assert dataset.samples[0].mask_path == mask_path

        results = run_all(
            dataset,
            algorithms=["AlgoA"],
            workers=1,
            batch_size=1,
        )

        assert set(results["AlgoA"]) == {"sample"}
        assert set(results["AlgoA"]["sample"]) == {"source"}

    def test_pipeline_diagnostics_separate_load_and_algorithm_failures(self, tmp_path):
        import json

        import cv2

        from testing.baseline import BaselineDataset
        from testing.scheduler import PipelineDiagnostics

        root = tmp_path / "diagnostics"
        source = root / "source"
        source.mkdir(parents=True)
        image = np.full((24, 20, 3), 92, dtype=np.uint8)
        mask = np.zeros((24, 20), dtype=np.uint8)
        assert cv2.imwrite(str(source / "good.jpg"), image)
        assert cv2.imwrite(str(source / "good_mask.png"), mask)
        (source / "broken.jpg").write_bytes(b"not an image")
        (source / "broken_mask.png").write_bytes(b"not a mask")
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "id": "good",
                            "kind": "source",
                            "image": "source/good.jpg",
                            "mask": "source/good_mask.png",
                        },
                        {
                            "id": "broken",
                            "kind": "source",
                            "image": "source/broken.jpg",
                            "mask": "source/broken_mask.png",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        diagnostics = PipelineDiagnostics()

        results = run_all(
            BaselineDataset(root),
            algorithms=["AlgoA"],
            workers=1,
            batch_size=1,
            diagnostics=diagnostics,
        )

        assert set(results["AlgoA"]) == {"good"}
        assert diagnostics.loaded_pairs == 1
        assert diagnostics.completed_tasks == 1
        assert diagnostics.load_failure_count == 1
        assert "broken.jpg" in diagnostics.load_failure_details[0]
        assert diagnostics.algorithm_failure_count == 0

    def test_pipeline_diagnostics_for_all_algorithm_tasks_failing(self, tmp_path):
        import json

        import cv2

        from testing.baseline import BaselineDataset
        from testing.scheduler import PipelineDiagnostics

        root = tmp_path / "all-algorithms-fail"
        source = root / "source"
        source.mkdir(parents=True)
        assert cv2.imwrite(
            str(source / "sample.jpg"), np.full((12, 12, 3), 64, dtype=np.uint8)
        )
        assert cv2.imwrite(
            str(source / "sample_mask.png"), np.zeros((12, 12), dtype=np.uint8)
        )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "id": "sample",
                            "kind": "source",
                            "image": "source/sample.jpg",
                            "mask": "source/sample_mask.png",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        class FailingAlgorithm(_FakeBase):
            name = "FailingDiagnosticsAlgorithm"

            def detect(self, image, is_cropped=False):
                raise RuntimeError("synthetic algorithm task failure")

        register_algorithm_instance(FailingAlgorithm.name, FailingAlgorithm())
        try:
            diagnostics = PipelineDiagnostics()
            results = run_all(
                BaselineDataset(root),
                algorithms=[FailingAlgorithm.name],
                workers=1,
                batch_size=1,
                diagnostics=diagnostics,
            )
        finally:
            from testing.registry import _INSTANCES

            _INSTANCES.pop(FailingAlgorithm.name, None)

        assert results[FailingAlgorithm.name] == {}
        assert diagnostics.loaded_pairs == 1
        assert diagnostics.completed_tasks == 0
        assert diagnostics.load_failure_count == 0
        assert diagnostics.algorithm_failure_count == 1
        assert "sample/source" in diagnostics.algorithm_failure_details[0]
        assert (
            "synthetic algorithm task failure"
            in diagnostics.algorithm_failure_details[0]
        )


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
