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

    def fake_run_all_with_comparison(dataset, **kwargs):
        runner_calls["dataset"] = dataset
        runner_calls["algorithms"] = kwargs["algorithms"]
        runner_calls["compare_pair"] = kwargs.get("compare_pair")
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
        }, None

    monkeypatch.setattr(
        "ui.testing_window.run_all_with_comparison", fake_run_all_with_comparison
    )
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
        for path, image, extension in (
            (source_image, np.full((16, 16, 3), 60 + index, dtype=np.uint8), ".jpg"),
            (source_mask, np.zeros((16, 16), dtype=np.uint8), ".png"),
        ):
            ok, encoded = cv2.imencode(extension, image)
            assert ok
            path.write_bytes(encoded.tobytes())
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
            for path, image, extension in (
                (
                    cropped_image,
                    np.full((12, 12, 3), 100 + index, dtype=np.uint8),
                    ".jpg",
                ),
                (cropped_mask, np.zeros((12, 12), dtype=np.uint8), ".png"),
            ):
                ok, encoded = cv2.imencode(extension, image)
                assert ok
                path.write_bytes(encoded.tobytes())
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
    # Задача 3.4: сравнение не повторяет детекции основного прогона.
    # 2 алгоритма × (source + cropped) = 4 вызова, а не 6.
    assert len(seen) == 4


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
    assert "Не удалось прочитать ни одной пары" in failed[0]
    assert "broken.jpg" in failed[0]
    assert "Все запуски выбранных алгоритмов" not in failed[0]
    assert (tmp_path / "performance.json").is_file()


def test_worker_reports_unreadable_pair_but_keeps_valid_results(qtbot, tmp_path):
    import json

    import cv2
    import numpy as np

    from testing.registry import _INSTANCES

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    assert cv2.imwrite(
        str(source_dir / "good.jpg"), np.full((16, 16, 3), 60, dtype=np.uint8)
    )
    assert cv2.imwrite(
        str(source_dir / "good_mask.png"), np.zeros((16, 16), dtype=np.uint8)
    )
    (source_dir / "broken.jpg").write_bytes(b"not an image")
    (source_dir / "broken_mask.png").write_bytes(b"not a mask")
    (tmp_path / "dataset.json").write_text(
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
    seen = []
    _register_mask_algorithm("PartialReadAlgorithm", seen)
    try:
        worker = _RunWorker(
            str(tmp_path), ["PartialReadAlgorithm"], workers=1, batch_size=1
        )
        finished = []
        failed = []
        worker.finished.connect(lambda results, extra: finished.append(results))
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("PartialReadAlgorithm", None)

    assert not failed
    assert len(finished) == 1
    assert set(finished[0]["PartialReadAlgorithm"]) == {"good"}
    assert seen == [((16, 16), False)]


def test_worker_manifest_run_reports_algorithm_failures_separately(qtbot, tmp_path):
    from testing.interface import BaseDetectionAlgorithm
    from testing.registry import _INSTANCES, register_algorithm_instance

    _write_manifest_dataset(tmp_path, stems=("valid",), include_cropped=False)

    class AlwaysFailAlgorithm(BaseDetectionAlgorithm):
        @property
        def name(self):
            return "AlwaysFailAlgorithm"

        @property
        def description(self):
            return "test algorithm which always fails"

        def detect(self, image, is_cropped=False):
            raise RuntimeError("synthetic algorithm failure")

    register_algorithm_instance("AlwaysFailAlgorithm", AlwaysFailAlgorithm())
    try:
        worker = _RunWorker(str(tmp_path), ["AlwaysFailAlgorithm"], workers=1)
        finished = []
        failed = []
        worker.finished.connect(lambda results, extra: finished.append(results))
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("AlwaysFailAlgorithm", None)

    assert not finished
    assert failed
    assert "Все запуски выбранных алгоритмов завершились с ошибкой" in failed[0]
    assert "valid/source" in failed[0]
    assert str(tmp_path / "source" / "valid.jpg") in failed[0]
    assert "synthetic algorithm failure" in failed[0]
    assert "Не удалось прочитать ни одной пары" not in failed[0]


def test_worker_runs_manifest_under_unicode_windows_path(qtbot, tmp_path):
    from testing.registry import _INSTANCES

    root = tmp_path / "Егор" / "22022540_imported"
    _write_manifest_dataset(root, stems=("unicode",), include_cropped=True)
    seen = []
    _register_mask_algorithm("UnicodePathManifest", seen)
    try:
        worker = _RunWorker(str(root), ["UnicodePathManifest"], batch_size=1, workers=1)
        finished = []
        failed = []
        worker.finished.connect(
            lambda results, extra: finished.append((results, extra))
        )
        worker.failed.connect(lambda message: failed.append(message))
        worker.run()
    finally:
        _INSTANCES.pop("UnicodePathManifest", None)

    assert not failed
    assert len(finished) == 1
    samples = finished[0][0]["UnicodePathManifest"]
    assert set(samples) == {"unicode", "unicode_cropped"}
    assert ((16, 16), False) in seen
    assert ((12, 12), True) in seen


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


# --- Задача 7.1: фазы прогресса и контекстные per-task ошибки ---


def test_worker_emits_phase_transitions(qtbot, tmp_path):
    """Worker излучает все фазы конвейера в порядке прохождения (задача 7.1)."""
    import numpy as np

    import cv2

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), 10, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "source" / "a.png"), image)
    cv2.imwrite(str(tmp_path / "masks" / "a_mask.png"), mask)

    worker = _RunWorker(str(tmp_path), ["ClassicDefault"])
    phases = []
    worker.phase_changed.connect(lambda name: phases.append(name))
    worker.run()

    expected = [
        "dataset_scan",
        "read_decode",
        "algorithm",
        "result_preparation",
    ]
    for phase in expected:
        assert phase in phases, f"фаза {phase} не наблюдалась: {phases}"
    # Порядок фаз сохраняется.
    indexes = [phases.index(phase) for phase in expected]
    assert indexes == sorted(indexes), f"нарушен порядок фаз: {phases}"


def test_worker_emits_progress_and_keeps_success_after_task_failure(
    qtbot, tmp_path, monkeypatch
):
    """Ошибка одной задачи не срывает прогон: успешные результаты доступны (7.1)."""
    import numpy as np

    import cv2

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    for stem, value in (("a", 10), ("b", 20)):
        image = np.full((8, 8, 3), value, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "source" / f"{stem}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"{stem}_mask.png"), mask)

    # Ломаем детекцию для пары "a": имитация per-task отказа алгоритма.
    from testing.registry import _INSTANCES

    class _FailOneSample:
        name = "FailOneSample"
        description = "тестовый алгоритм: отказ на sample-a"

        def detect(self, image, is_cropped=False):
            import numpy as _np

            return _np.zeros(image.shape[:2], dtype=_np.uint8)

    algo = _FailOneSample()
    original_detect = algo.detect

    def detect(image, is_cropped=False):
        # Детекция работает для любых входов — отказ сделаем через загрузку.
        return original_detect(image, is_cropped)

    algo.detect = detect
    _INSTANCES["FailOneSample"] = algo

    worker = _RunWorker(str(tmp_path), ["FailOneSample"])
    task_errors = []
    progresses = []
    finished = []
    failed = []
    worker.task_error.connect(lambda ctx, err: task_errors.append((ctx, err)))
    worker.progress_made.connect(lambda done, total: progresses.append((done, total)))
    worker.finished.connect(lambda results, extra: finished.append((results, extra)))
    worker.failed.connect(lambda msg: failed.append(msg))
    try:
        worker.run()
    finally:
        _INSTANCES.pop("FailOneSample", None)

    assert finished, f"прогон должен завершиться, failed={failed}"
    assert not failed
    assert progresses, "сигналы прогресса не излучались"
    assert progresses[-1][1] >= 2, "total должен учитывать все задачи"
    assert progresses[-1][0] == progresses[-1][1], "все задачи завершены"
    results, extra = finished[0]
    assert "FailOneSample" in results
    assert results["FailOneSample"], "успешные задачи должны быть доступны"


def test_worker_task_error_reports_contextual_failure(qtbot, tmp_path):
    """Per-task ошибка содержит контекст алгоритма/пары/варианта (задача 7.1)."""
    import numpy as np

    import cv2

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), 10, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "source" / "a.png"), image)
    cv2.imwrite(str(tmp_path / "masks" / "a_mask.png"), mask)

    from testing.registry import _INSTANCES

    class _AlwaysFails:
        name = "AlwaysFailsAlgo"
        description = "тестовый алгоритм: всегда падает"

        def detect(self, image, is_cropped=False):
            raise RuntimeError("искусственный сбой детекции")

    _INSTANCES["AlwaysFailsAlgo"] = _AlwaysFails()
    worker = _RunWorker(str(tmp_path), ["AlwaysFailsAlgo"])
    task_errors = []
    failed = []
    worker.task_error.connect(lambda ctx, err: task_errors.append((ctx, err)))
    worker.failed.connect(lambda msg: failed.append(msg))
    try:
        worker.run()
    finally:
        _INSTANCES.pop("AlwaysFailsAlgo", None)

    assert task_errors, "per-task ошибки должны наблюдаться"
    context, error = task_errors[0]
    assert "AlwaysFailsAlgo" in context
    assert "a" in context
    assert "искусственный сбой" in error
    assert failed, "когда все задачи упали — worker сообщает об ошибке"


# --- Задача 7.2: model/view для больших результатов ---


def test_results_tables_use_model_view_without_per_cell_widgets(qtbot, window):
    """Таблицы результатов — QTableView с моделью, без QTableWidgetItem (7.2)."""
    from PyQt6.QtWidgets import QTableWidget

    assert not isinstance(window.table, QTableWidget), (
        "таблица результатов должна использовать model/view, а не QTableWidget"
    )
    assert window.table.model() is window.summary_model
    assert window.comparison_table.model() is window.comparison_model
    # Ячейки не создают виджеты: indexWidget для любой ячейки пуст.
    assert window.table.indexWidget(window.summary_model.index(0, 0)) is None


def test_large_result_set_renders_with_live_heartbeat(qtbot, window):
    """Тысячи строк: отрисовка без виджета на ячейку, heartbeat живой (7.2)."""
    import time as _time

    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication

    total_rows = 1500
    summary = [
        {
            "name": f"Algo{index:04d}",
            "metrics": {
                "iou": index / total_rows,
                "dice": 0.5,
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
            },
        }
        for index in range(total_rows)
    ]

    ticks = []
    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(_time.monotonic()))
    heartbeat.start()

    # Заполнение модели + прокрутка через event loop: считаем тики heartbeat.
    window._fill_table(summary)
    window._fill_comparison(
        {
            "iou": {
                f"sample{index:04d}": {"AlgoA": "AlgoB"} for index in range(total_rows)
            }
        }
    )
    loop = QEventLoop()
    QTimer.singleShot(250, loop.quit)
    loop.exec()
    # Прокручиваем к концу — делегат рисует только видимую часть.
    window.table.scrollToBottom()
    window.comparison_table.scrollToBottom()
    QApplication.processEvents()
    heartbeat.stop()

    assert window.summary_model.rowCount() == total_rows
    assert window.comparison_model.rowCount() == total_rows
    assert len(ticks) >= 2, (
        f"heartbeat не работал во время наполнения таблицы (ticks={len(ticks)})"
    )
    # Model/view: число созданных виджетов не растёт с числом строк.
    assert window.table.indexWidget(window.summary_model.index(0, 0)) is None
    assert window.table.indexWidget(
        window.summary_model.index(total_rows - 1, 5)
    ) is None


def test_model_view_incremental_append_keeps_existing_rows(qtbot, window):
    """Инкрементальное наполнение не теряет ранее добавленные строки (7.2)."""
    window.summary_model.set_summary([{"name": "A", "metrics": {"iou": 1.0}}])
    window.summary_model.append_summary(
        [{"name": "B", "metrics": {"iou": 0.5}}, {"name": "C", "metrics": {"iou": 0.25}}]
    )
    assert window.summary_model.rowCount() == 3
    assert window.summary_model.data(window.summary_model.index(0, 0)) == "A"
    assert window.summary_model.data(window.summary_model.index(2, 0)) == "C"
    assert window.summary_model.data(window.summary_model.index(2, 1)) == "0.2500"
    assert window.table.rowCount() == 3


def test_model_view_clear_resets_table(qtbot, window):
    window.summary_model.set_summary([{"name": "A", "metrics": {}}])
    window.summary_model.clear()
    assert window.summary_model.rowCount() == 0
    assert window.table.rowCount() == 0


# --- Задача 7.3: фоновая генерация HTML-отчёта с атомарной публикацией ---


def _sample_results():
    return {
        "ClassicDefault": {
            "sample1": {
                "source": {
                    "iou": 0.9,
                    "dice": 0.95,
                    "f1": 0.92,
                    "precision": 0.93,
                    "recall": 0.91,
                    "accuracy": 0.99,
                    "tp": 90,
                    "fp": 0,
                    "fn": 10,
                    "tn": 900,
                }
            }
        }
    }


def test_generate_report_atomic_publishes_on_success(tmp_path):
    """Отчёт публикуется по финальному пути только после успеха (7.3)."""
    from testing.dashboard import generate_report_atomic

    target = tmp_path / "report.html"
    published = generate_report_atomic(_sample_results(), str(target))
    assert published == str(target)
    text = target.read_text(encoding="utf-8")
    assert "ClassicDefault" in text, "содержимое отчёта должно быть совместимо"
    assert "0.9000" in text
    # Временные файлы не остаются.
    leftovers = [
        p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"остались временные файлы: {leftovers}"


def test_generate_report_atomic_failure_keeps_existing_target(tmp_path):
    """Ошибка генерации не меняет ранее существовавший отчёт (7.3)."""
    import pytest

    from testing.dashboard import generate_report_atomic

    target = tmp_path / "report.html"
    target.write_text("СУЩЕСТВУЮЩИЙ ОТЧЁТ", encoding="utf-8")

    def broken_report(*args, **kwargs):
        raise RuntimeError("генерация сломалась")

    import testing.dashboard as dashboard

    original = dashboard.generate_report
    dashboard.generate_report = broken_report
    try:
        with pytest.raises(RuntimeError, match="сломалась"):
            generate_report_atomic(_sample_results(), str(target))
    finally:
        dashboard.generate_report = original

    assert target.read_text(encoding="utf-8") == "СУЩЕСТВУЮЩИЙ ОТЧЁТ", (
        "существующий отчёт должен остаться неизменным"
    )
    leftovers = [
        p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"частичный отчёт остался: {leftovers}"


def test_generate_report_atomic_cancel_keeps_existing_target(tmp_path):
    """Отмена между генерацией и публикацией не трогает цель (7.3)."""
    import pytest

    from ui.background import OperationCancelled
    from testing.dashboard import generate_report_atomic

    target = tmp_path / "report.html"
    target.write_text("СТАРЫЙ", encoding="utf-8")

    def canceled_check():
        raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        generate_report_atomic(
            _sample_results(), str(target), publish_check=canceled_check
        )

    assert target.read_text(encoding="utf-8") == "СТАРЫЙ"
    leftovers = [
        p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"частичный отчёт остался: {leftovers}"


def test_export_runs_in_background_and_reports_status(
    qtbot, window, monkeypatch, tmp_path
):
    """Экспорт идёт вне GUI-потока: статус, публикация, кнопки (7.3)."""
    window._results = _sample_results()
    import os as _os

    out_path = _os.path.join(str(tmp_path), "out_report.html")
    monkeypatch.setattr(
        "ui.testing_window.QFileDialog.getSaveFileName",
        lambda *a, **k: (out_path, "HTML reports (*.html)"),
    )
    criticals = []
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.critical",
        lambda *a, **k: criticals.append(a),
    )

    window._export_report()
    qtbot.waitUntil(
        lambda: not window._export_op.is_running(), timeout=15000
    )

    assert not criticals, f"экспорт сообщил ошибку: {criticals}"
    assert window.status_label.text().startswith("Отчёт сохранён"), (
        f"статус: {window.status_label.text()}"
    )
    assert _os.path.exists(out_path), "отчёт не опубликован"
    text = open(out_path, encoding="utf-8").read()
    assert "ClassicDefault" in text
    assert window.btn_export.isEnabled(), "кнопка экспорта должна восстановиться"
    assert window.btn_run.isEnabled(), "кнопка прогона должна восстановиться"


# --- Задача 7.4: отмена прогона/экспорта и deferred close ---


def test_cancel_button_stops_run_at_safe_boundary(qtbot, tmp_path, monkeypatch):
    """Кнопка «Отменить» останавливает постановку новых задач (7.4)."""
    import time as _time

    import numpy as np

    import cv2

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    for index in range(6):
        image = np.full((8, 8, 3), index * 3 + 1, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "source" / f"p{index}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"p{index}_mask.png"), mask)

    from testing.registry import _INSTANCES

    calls = []

    class _SlowAlgo:
        name = "SlowCancelGUI"
        description = "тест: отмена в GUI"

        def detect(self, image, is_cropped=False):
            calls.append(1)
            _time.sleep(0.02)
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["SlowCancelGUI"] = _SlowAlgo()
    window = TestingWindow()
    qtbot.addWidget(window)
    window.dataset_input.setText(str(tmp_path))
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.warning", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.critical", lambda *a, **k: None
    )
    # Выбираем только медленный алгоритм.
    for name, checkbox in window._algo_checkboxes.items():
        checkbox.setChecked(name == "SlowCancelGUI")
    window.batch_size_input.setValue(1)  # отмена на границе между батчами
    try:
        window._start_run()
        qtbot.waitUntil(
            lambda: len(calls) >= 1 or window._worker is None, timeout=10000
        )
        window._cancel_run()
        qtbot.waitUntil(
            lambda: window._thread is None or not window._thread.isRunning(),
            timeout=15000,
        )
    finally:
        _INSTANCES.pop("SlowCancelGUI", None)

    assert len(calls) < 6, (
        f"отмена не остановила новые задачи: выполнено {len(calls)} из 6"
    )
    assert not window.btn_cancel.isEnabled(), "кнопка отмены должна блокироваться"


def test_close_during_run_defers_and_stays_responsive(qtbot, tmp_path, monkeypatch):
    """Закрытие при живом прогоне: «Завершение…», ответственность, авто-закрытие (7.4)."""
    import time as _time

    import numpy as np

    import cv2

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    for index in range(4):
        image = np.full((8, 8, 3), index * 3 + 1, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "source" / f"p{index}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"p{index}_mask.png"), mask)

    from testing.registry import _INSTANCES

    class _MediumAlgo:
        name = "MediumCloseGUI"
        description = "тест: закрытие при прогоне"

        def detect(self, image, is_cropped=False):
            _time.sleep(0.03)
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["MediumCloseGUI"] = _MediumAlgo()
    window = TestingWindow()
    qtbot.addWidget(window)
    window.dataset_input.setText(str(tmp_path))
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.warning", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.critical", lambda *a, **k: None
    )
    for name, checkbox in window._algo_checkboxes.items():
        checkbox.setChecked(name == "MediumCloseGUI")
    try:
        window._start_run()
        qtbot.waitUntil(lambda: window._thread.isRunning(), timeout=10000)
        window.close()
        # Deferred close: окно НЕ закрывается мгновенно и остаётся отзывчивым.
        assert window._closing, "должна быть запрошена deferred close"
        assert "Завершение" in window.status_label.text(), (
            f"статус: {window.status_label.text()}"
        )
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=15000)
    finally:
        _INSTANCES.pop("MediumCloseGUI", None)
        window._closing = False
        if window._thread is not None:
            window._thread.quit()
            window._thread.wait(2000)

    assert not window.isVisible(), "окно должно закрыться после safe-точки"


def test_late_results_not_applied_after_close_requested(qtbot, tmp_path, monkeypatch):
    """Поздние результаты не применяются после запроса закрытия (7.4)."""
    import numpy as np

    import cv2

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), 10, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "source" / "a.png"), image)
    cv2.imwrite(str(tmp_path / "masks" / "a_mask.png"), mask)

    from testing.registry import _INSTANCES

    class _LateAlgo:
        name = "LateResultGUI"
        description = "тест: поздние результаты"

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    _INSTANCES["LateResultGUI"] = _LateAlgo()
    window = TestingWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.warning", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "ui.testing_window.QMessageBox.critical", lambda *a, **k: None
    )
    window._closing = True  # окно уже запросило закрытие
    worker = _RunWorker(str(tmp_path), ["LateResultGUI"])
    finished = []
    worker.finished.connect(lambda r, e: finished.append((r, e)))
    try:
        worker.run()
        # Worker завершился, но окно не должно применять результаты.
        assert finished, "worker должен завершиться"
        window._on_finished(*finished[0])
        assert window.summary_model.rowCount() == 0, (
            "поздние результаты не должны попадать в таблицу после закрытия"
        )
    finally:
        _INSTANCES.pop("LateResultGUI", None)
        window._closing = False
