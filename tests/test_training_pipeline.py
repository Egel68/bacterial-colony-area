"""Focused checks for the real COCO training orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import train.training_pipeline as pipeline


def test_cpu_benchmark_excludes_full_image_warmups_and_counts_tiles(
    tmp_path, monkeypatch
):
    samples = []
    for index in range(5):
        filename = f"dish-{index}.png"
        mask_filename = f"dish-{index}_mask.png"
        cv2.imwrite(str(tmp_path / filename), np.zeros((900, 900, 3), np.uint8))
        cv2.imwrite(str(tmp_path / mask_filename), np.zeros((900, 900), np.uint8))
        samples.append(
            {
                "id": f"dish-{index}",
                "kind": "source",
                "image": filename,
                "mask": mask_filename,
                "subset": "test",
            }
        )
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "name": "benchmark",
                "origin": "external",
                "mask_mode": "binary",
                "storage": "copy",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    durations = iter((0.0, 3.0, 10.0, 14.0, 20.0, 40.0, 50.0, 80.0, 90.0, 130.0))
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(durations))
    observed = []

    class _FakeTiledAlgorithm:
        def __init__(self, **kwargs):
            assert kwargs["img_size"] == 512
            assert kwargs["stride"] == 384

        def detect(self, image, is_cropped=False):
            observed.append(image.shape[:2])
            return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        "testing.tiled_onnx_algorithm.TiledOnnxModelAlgorithm", _FakeTiledAlgorithm
    )

    report = pipeline._cpu_full_image_benchmark(
        tmp_path / "model.onnx", tmp_path, tile_size=512, stride=384
    )

    assert observed == [(900, 900)] * 5
    assert [row["warmup"] for row in report["images"]] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert [row["tiles"] for row in report["images"]] == [9] * 5
    assert report["warmup_seconds"] == [3.0, 4.0, 20.0]
    assert report["median_seconds"] == pytest.approx(35.0)
    assert report["p95_seconds"] == pytest.approx(39.5)
    assert report["measured_images"] == 2
    assert report["onnxruntime_intra_op_num_threads"] == 0
    assert report["logical_cores"] == pipeline.os.cpu_count()
    assert report["metrics"]["n_images"] == 5
    assert report["metrics"]["per_image"]["dish-0"]["tn"] == 900 * 900


def test_cpu_benchmark_keeps_one_measurement_after_warmups(tmp_path, monkeypatch):
    samples = []
    for index in range(2):
        image_name = f"dish-{index}.png"
        mask_name = f"dish-{index}_mask.png"
        cv2.imwrite(str(tmp_path / image_name), np.zeros((32, 32, 3), np.uint8))
        cv2.imwrite(str(tmp_path / mask_name), np.zeros((32, 32), np.uint8))
        samples.append(
            {
                "id": f"dish-{index}",
                "kind": "source",
                "image": image_name,
                "mask": mask_name,
                "subset": "test",
            }
        )
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "name": "small-benchmark",
                "origin": "external",
                "mask_mode": "binary",
                "storage": "copy",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    durations = iter((0.0, 2.0, 10.0, 14.0))
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(durations))

    class _FakeTiledAlgorithm:
        def __init__(self, **_kwargs):
            pass

        def detect(self, image, is_cropped=False):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        "testing.tiled_onnx_algorithm.TiledOnnxModelAlgorithm", _FakeTiledAlgorithm
    )

    report = pipeline._cpu_full_image_benchmark(
        tmp_path / "model.onnx", tmp_path, tile_size=32, stride=24
    )

    assert report["warmup_images"] == 1
    assert report["measured_images"] == 1
    assert report["median_seconds"] == pytest.approx(4.0)


def test_training_pipeline_stops_if_import_changes_read_only_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "annot_COCO.json").write_text("initial", encoding="utf-8")

    def _mutating_import(self, data_root, output_dir, **kwargs):
        (Path(data_root) / "annot_COCO.json").write_text("mutated", encoding="utf-8")

    monkeypatch.setattr(pipeline.CocoBboxImporter, "build", _mutating_import)
    monkeypatch.setattr(
        pipeline,
        "run_training",
        lambda *_args, **_kwargs: pytest.fail(
            "training must not start after source mutation"
        ),
    )

    with pytest.raises(RuntimeError, match="(?i)read-only COCO source changed"):
        pipeline.run_colony_training(
            data_root=source,
            output_root=tmp_path / "run",
            pretrained=False,
        )


def test_training_pipeline_stops_if_source_changes_after_training(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "annot_COCO.json"
    marker.write_text("stable", encoding="utf-8")
    output_root = tmp_path / "run"
    prepared = output_root / "dataset"
    prepared.mkdir(parents=True)
    coco = {
        "images": [{"id": index, "file_name": f"{index}.png"} for index in (1, 2, 3)],
        "annotations": [
            {"image_id": index, "bbox": [0, 0, 1, 1]} for index in (1, 2, 3)
        ],
    }
    marker.write_text(json.dumps(coco), encoding="utf-8")
    split_groups = {
        subset: [
            {
                "image_id": index,
                "sample_id": f"sample-{index}",
                "category_id": 1,
            }
        ]
        for index, subset in zip((1, 2, 3), ("train", "val", "test"), strict=True)
    }
    (prepared / "dataset.json").write_text("{}", encoding="utf-8")
    (prepared / "split.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                "groups": split_groups,
            }
        ),
        encoding="utf-8",
    )
    resume_run = output_root / "runs" / "partial"
    checkpoint = resume_run / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    signatures = iter(("stable", "stable", "changed-during-training"))
    monkeypatch.setattr(pipeline, "_tree_signature", lambda _root: next(signatures))
    monkeypatch.setattr(pipeline.CocoBboxImporter, "build", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pipeline,
        "load_manifest",
        lambda _root: SimpleNamespace(
            samples=[
                SimpleNamespace(
                    id=f"sample-{index}",
                    subset=subset,
                    kind="source",
                    image=f"sample-{index}.png",
                    mask=f"sample-{index}_mask.png",
                )
                for index, subset in zip(
                    (1, 2, 3), ("train", "val", "test"), strict=True
                )
            ]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_training",
        lambda _cfg: ({"best_epoch": 1}, [], [], resume_run),
    )

    with pytest.raises(RuntimeError, match="changed before checkpoint evaluation"):
        pipeline.run_colony_training(
            data_root=source,
            output_root=output_root,
            pretrained=False,
            resume_run=resume_run,
        )
