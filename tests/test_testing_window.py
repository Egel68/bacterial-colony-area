import pytest

from ui.testing_window import TestingWindow, _RunWorker

pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qtbot):
    win = TestingWindow()
    qtbot.addWidget(win)
    win.show()
    return win


def test_window_opens_with_algorithms(qtbot, window):
    assert window.isVisible()
    assert len(window._algo_checkboxes) >= 4
    names = list(window._algo_checkboxes)
    assert "ClassicDefault" in names


def test_results_table_header_stretches(qtbot, window):
    from PyQt6.QtWidgets import QHeaderView

    assert (
        window.table.horizontalHeader().sectionResizeMode(0)
        == QHeaderView.ResizeMode.Stretch
    )
    assert (
        window.comparison_table.horizontalHeader().sectionResizeMode(0)
        == QHeaderView.ResizeMode.Stretch
    )


def test_dataset_hint_mentions_directories(qtbot, window):
    from PyQt6.QtWidgets import QLabel

    hints = [
        label
        for label in window.findChildren(QLabel)
        if "source/" in label.text() and "dataset.json" in label.text()
    ]
    assert hints
    hint = hints[0]
    text = hint.text()
    for part in ("source/", "masks/", "cropped/", "cropped_masks/"):
        assert part in text
    assert "dataset.json" in text
    assert "datasets/22022540_imported" in text
    assert "docs/datasets.md" in text
    assert hint.openExternalLinks()
    assert (
        "https://github.com/Egel68/bacterial-colony-area/blob/main/docs/datasets.md"
        in text
    )


def test_missing_dataset_dir_shows_warning(qtbot, window, tmp_path, monkeypatch):
    window.dataset_input.setText(str(tmp_path / "nope"))
    warnings = []
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.warning", lambda *a, **k: warnings.append(a)
    )
    window._start_run()
    assert warnings
    assert window._thread is None


def test_worker_without_samples_fails(qtbot, tmp_path):
    worker = _RunWorker(str(tmp_path), ["ClassicDefault"])
    failed = []
    worker.failed.connect(lambda msg: failed.append(msg))
    worker.run()
    assert failed
    assert "No test samples found" in failed[0]


def test_worker_loads_manifest_jpeg_png_without_modifying_files(
    qtbot, tmp_path, monkeypatch
):
    import json

    import cv2
    import numpy as np

    from testing.baseline import BaselineDataset

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image_path = source_dir / "sample.jpg"
    mask_path = source_dir / "sample_mask.png"
    cv2.imwrite(str(image_path), np.full((12, 16, 3), 80, dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.zeros((12, 16), dtype=np.uint8))
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "manifest-smoke",
                "origin": "external",
                "mask_mode": "binary",
                "storage": "copy",
                "samples": [
                    {
                        "id": "sample",
                        "kind": "source",
                        "image": "source/sample.jpg",
                        "mask": "source/sample_mask.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    original_files = {
        path: path.read_bytes() for path in (image_path, mask_path, manifest_path)
    }
    runner_calls = {}

    def fake_run_all(dataset, **kwargs):
        runner_calls["dataset"] = dataset
        runner_calls["algorithms"] = kwargs["algorithms"]
        return {
            "ManifestSmoke": {
                "sample": {
                    "source": {
                        "iou": 1.0,
                        "dice": 1.0,
                        "f1": 1.0,
                        "precision": 1.0,
                        "recall": 1.0,
                    }
                }
            }
        }

    monkeypatch.setattr("ui.testing_window.run_all", fake_run_all)
    worker = _RunWorker(str(tmp_path), ["ManifestSmoke"])
    finished = []
    failed = []
    worker.finished.connect(lambda results, extra: finished.append((results, extra)))
    worker.failed.connect(lambda message: failed.append(message))
    worker.run()

    assert not failed
    assert len(finished) == 1
    assert isinstance(runner_calls["dataset"], BaselineDataset)
    assert runner_calls["algorithms"] == ["ManifestSmoke"]
    sample = runner_calls["dataset"].samples[0]
    assert sample.image_path == image_path
    assert sample.mask_path == mask_path
    assert finished[0][0]["ManifestSmoke"]["sample"]["source"]["iou"] == 1.0
    assert {path: path.read_bytes() for path in original_files} == original_files


def _write_manifest_dataset(root, stems=("sample-a", "sample-b"), include_cropped=True):
    import json

    import cv2
    import numpy as np

    samples = []
    for index, stem in enumerate(stems):
        source_image = root / "source" / f"{stem}.jpg"
        source_mask = root / "source" / f"{stem}_mask.png"
        source_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(source_image),
            np.full((16, 16, 3), 60 + index, dtype=np.uint8),
        )
        cv2.imwrite(str(source_mask), np.zeros((16, 16), dtype=np.uint8))
        samples.append(
            {
                "id": stem,
                "kind": "source",
                "image": f"source/{stem}.jpg",
                "mask": f"source/{stem}_mask.png",
            }
        )

        if include_cropped:
            cropped_image = root / "cropped" / f"{stem}_cropped.jpg"
            cropped_mask = root / "cropped" / f"{stem}_cropped_mask.png"
            cropped_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(cropped_image),
                np.full((12, 12, 3), 100 + index, dtype=np.uint8),
            )
            cv2.imwrite(str(cropped_mask), np.zeros((12, 12), dtype=np.uint8))
            samples.append(
                {
                    "id": f"{stem}_cropped",
                    "kind": "cropped",
                    "image": f"cropped/{stem}_cropped.jpg",
                    "mask": f"cropped/{stem}_cropped_mask.png",
                }
            )

    (root / "dataset.json").write_text(
        json.dumps(
            {
                "name": "manifest-worker-tests",
                "origin": "external",
                "mask_mode": "binary",
                "storage": "copy",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def _register_mask_algorithm(name, seen):
    import numpy as np

    from testing.interface import BaseDetectionAlgorithm
    from testing.registry import register_algorithm_instance

    class RecordingMaskAlgorithm(BaseDetectionAlgorithm):
        @property
        def name(self):
            return name

        @property
        def description(self):
            return "manifest worker test"

        def detect(self, image, is_cropped=False):
            seen.append((image.shape[:2], is_cropped))
            return np.zeros(image.shape[:2], dtype=np.uint8)

    instance = RecordingMaskAlgorithm()
    register_algorithm_instance(name, instance)
    return instance


def test_worker_runs_manifest_source_only_pairs(qtbot, tmp_path):
    from testing.registry import _INSTANCES

    _write_manifest_dataset(tmp_path, include_cropped=False)
    seen = []
    _register_mask_algorithm("ManifestSourceOnly", seen)
    try:
        worker = _RunWorker(
            str(tmp_path), ["ManifestSourceOnly"], batch_size=1, workers=1
        )
        finished = []
        failed = []
        worker.finished.connect(
            lambda results, extra: finished.append((results, extra))
        )
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("ManifestSourceOnly", None)

    assert not failed
    assert len(finished) == 1
    samples = finished[0][0]["ManifestSourceOnly"]
    assert set(samples) == {"sample-a", "sample-b"}
    assert all(set(variants) == {"source"} for variants in samples.values())
    assert seen == [((16, 16), False), ((16, 16), False)]


def test_worker_manifest_results_keep_source_cropped_limit_and_comparison(
    qtbot, tmp_path
):
    from testing.registry import _INSTANCES

    _write_manifest_dataset(tmp_path)
    seen = []
    _register_mask_algorithm("ManifestWorkerAlgorithm", seen)
    _register_mask_algorithm("ManifestWorkerAlgorithmB", seen)
    try:
        worker = _RunWorker(
            str(tmp_path),
            ["ManifestWorkerAlgorithm"],
            compare_pair=("ManifestWorkerAlgorithm", "ManifestWorkerAlgorithmB"),
            sample_limit=1,
            batch_size=1,
            workers=1,
        )
        finished = []
        failed = []
        worker.finished.connect(
            lambda results, extra: finished.append((results, extra))
        )
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("ManifestWorkerAlgorithm", None)
        _INSTANCES.pop("ManifestWorkerAlgorithmB", None)

    assert not failed
    assert len(finished) == 1
    results, extra = finished[0]
    samples = results["ManifestWorkerAlgorithm"]
    assert set(samples) == {"sample-a", "sample-a_cropped"}
    assert set(samples["sample-a"]) == {"source"}
    assert set(samples["sample-a_cropped"]) == {"cropped"}
    assert extra["summary"][0]["metrics"]["iou"] == 0.0
    assert extra["comparison"]["iou"]["sample-a"]["source"] == "tie"
    assert extra["comparison"]["iou"]["sample-a_cropped"]["cropped"] == "tie"
    assert ((16, 16), False) in seen
    assert ((12, 12), True) in seen
    assert len(seen) == 6  # run + comparison, each processes source and cropped.


def test_worker_runs_selected_algorithm_on_coco_importer_output(
    qtbot, tmp_path, monkeypatch
):
    import json

    import cv2
    import numpy as np

    from testing.registry import _INSTANCES
    from train.dataset_adapters import CocoBboxImporter

    source_root = tmp_path / "coco-source"
    source_root.mkdir()
    image = np.full((32, 32, 3), 90, dtype=np.uint8)
    cv2.circle(image, (16, 16), 10, (170, 170, 170), -1)
    cv2.imwrite(str(source_root / "sample.jpg"), image)
    (source_root / "annot_COCO.json").write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "colony"}],
                "images": [
                    {
                        "id": 1,
                        "file_name": "sample.jpg",
                        "width": 32,
                        "height": 32,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "bbox": [12, 12, 8, 8],
                        "category_id": 1,
                        "area": 64,
                        "iscrowd": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "train.dataset_adapters._find_dish",
        lambda image: {"cx": 16, "cy": 16, "r": 14},
    )
    output_root = tmp_path / "imported"
    manifest = CocoBboxImporter().build(
        data_root=source_root,
        output_dir=output_root,
        crop=True,
    )
    assert {sample.kind for sample in manifest.samples} == {"source", "cropped"}
    assert len(manifest.samples) == 2

    seen = []
    _register_mask_algorithm("CocoImportedSmoke", seen)
    _register_mask_algorithm("CocoImportedSmokeB", seen)
    try:
        worker = _RunWorker(
            str(output_root),
            ["CocoImportedSmoke"],
            compare_pair=("CocoImportedSmoke", "CocoImportedSmokeB"),
            batch_size=1,
            workers=1,
        )
        finished = []
        failed = []
        worker.finished.connect(
            lambda results, extra: finished.append((results, extra))
        )
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("CocoImportedSmoke", None)
        _INSTANCES.pop("CocoImportedSmokeB", None)

    assert not failed
    assert len(finished) == 1
    results, extra = finished[0]
    samples = results["CocoImportedSmoke"]
    assert set(samples) == {"sample", "sample_cropped"}
    assert set(samples["sample"]) == {"source"}
    assert set(samples["sample_cropped"]) == {"cropped"}
    assert extra["summary"][0]["name"] == "CocoImportedSmoke"
    assert extra["comparison"]["iou"]["sample"]["source"] == "tie"
    assert extra["comparison"]["iou"]["sample_cropped"]["cropped"] == "tie"
    assert ((32, 32), False) in seen
    assert ((32, 32), True) in seen
    assert (output_root / "dataset.json").is_file()


def test_worker_manifest_run_fails_if_every_pair_cannot_be_decoded(qtbot, tmp_path):
    import json

    from testing.interface import BaseDetectionAlgorithm
    from testing.registry import _INSTANCES, register_algorithm_instance

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "broken.jpg").write_bytes(b"not a decodable image")
    (source_dir / "broken_mask.png").write_bytes(b"not a decodable mask")
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "broken",
                        "kind": "source",
                        "image": "source/broken.jpg",
                        "mask": "source/broken_mask.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class MustNotRunAlgorithm(BaseDetectionAlgorithm):
        @property
        def name(self):
            return "MustNotRunAlgorithm"

        @property
        def description(self):
            return "not expected to run"

        def detect(self, image, is_cropped=False):
            raise AssertionError("algorithm must not run on a missing pair")

    register_algorithm_instance("MustNotRunAlgorithm", MustNotRunAlgorithm())
    try:
        worker = _RunWorker(str(tmp_path), ["MustNotRunAlgorithm"], telemetry=True)
        finished = []
        failed = []
        worker.finished.connect(lambda results, extra: finished.append(results))
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("MustNotRunAlgorithm", None)

    assert not finished
    assert failed
    assert "не получил результатов" in failed[0]
    assert (tmp_path / "performance.json").is_file()


def test_worker_malformed_manifest_fails_with_clear_message(qtbot, tmp_path):
    (tmp_path / "dataset.json").write_text("{ invalid json", encoding="utf-8")
    worker = _RunWorker(str(tmp_path), ["ClassicDefault"])
    finished = []
    failed = []
    worker.finished.connect(lambda results, extra: finished.append(results))
    worker.failed.connect(lambda message: failed.append(message))

    worker.run()

    assert not finished
    assert failed
    assert "Не удалось загрузить датасет" in failed[0]
    assert "dataset.json" in failed[0]


def test_worker_runs_five_object_smoke_dataset(qtbot, tmp_path):
    import cv2
    import numpy as np

    for directory in ("source", "masks"):
        (tmp_path / directory).mkdir()
    for index in range(5):
        image = np.full((20, 20, 3), index, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "source" / f"s{index}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"s{index}_mask.png"), mask)

    worker = _RunWorker(
        str(tmp_path),
        ["ClassicDefault"],
        sample_limit=5,
        batch_size=2,
    )
    finished = []
    failed = []
    worker.finished.connect(lambda results, extra: finished.append(results))
    worker.failed.connect(lambda msg: failed.append(msg))
    worker.run()

    assert not failed
    assert len(finished[0]["ClassicDefault"]) == 5


def test_worker_legacy_sample_limit_keeps_crop_for_selected_source(qtbot, tmp_path):
    import cv2
    import numpy as np

    from testing.registry import _INSTANCES

    for directory in ("source", "masks", "cropped", "cropped_masks"):
        (tmp_path / directory).mkdir()
    for index, name in enumerate(("a", "b")):
        cv2.imwrite(
            str(tmp_path / "source" / f"{name}.png"),
            np.full((12, 12, 3), index, dtype=np.uint8),
        )
        cv2.imwrite(
            str(tmp_path / "masks" / f"{name}_mask.png"),
            np.zeros((12, 12), dtype=np.uint8),
        )
        cv2.imwrite(
            str(tmp_path / "cropped" / f"{name}_cropped.png"),
            np.full((8, 8, 3), index, dtype=np.uint8),
        )
        cv2.imwrite(
            str(tmp_path / "cropped_masks" / f"{name}_cropped_mask.png"),
            np.zeros((8, 8), dtype=np.uint8),
        )

    seen = []
    _register_mask_algorithm("LegacyLimitWithCrop", seen)
    try:
        worker = _RunWorker(
            str(tmp_path), ["LegacyLimitWithCrop"], sample_limit=1, workers=1
        )
        finished = []
        failed = []
        worker.finished.connect(lambda results, extra: finished.append(results))
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("LegacyLimitWithCrop", None)

    assert not failed
    assert len(finished) == 1
    samples = finished[0]["LegacyLimitWithCrop"]
    assert set(samples) == {"a", "a_cropped"}
    assert set(samples["a"]) == {"source"}
    assert set(samples["a_cropped"]) == {"cropped"}
    assert set(seen) == {((12, 12), False), ((8, 8), True)}
