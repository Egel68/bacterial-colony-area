import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from train.main import main  # noqa: E402


def test_training_cli_runs_tiny_dataset_without_pretrained_weights(
    tmp_path, monkeypatch
):
    import train.training_pipeline as pipeline

    source = tmp_path / "source"
    source.mkdir()
    coco_images = []
    annotations = []
    for index in range(6):
        filename = f"dish-{index}.png"
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:] = (20 + index, 40, 60)
        cv2.imwrite(str(source / filename), image)
        coco_images.append(
            {"id": index + 1, "file_name": filename, "width": 64, "height": 64}
        )
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "bbox": [20, 20, 18, 18],
                "category_id": 1 if index < 3 else 2,
            }
        )
    (source / "annot_COCO.json").write_text(
        json.dumps(
            {
                "images": coco_images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "run_training", _fake_run_training)
    monkeypatch.setattr(
        pipeline,
        "evaluate_test_subset",
        lambda *_args, **_kwargs: pytest.fail(
            "CPU benchmark path must calculate test metrics in its single ONNX pass"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_patch_records",
        lambda _model, records, *_args, **_kwargs: {
            "mean": {"iou": 0.5, "dice": 0.66},
            "per_image": {record.id: {"iou": 0.5} for record in records},
            "n_images": len(records),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "verify_onnx_cpu_parity",
        lambda *_args, **_kwargs: {
            "providers": ["CPUExecutionProvider"],
            "patches": 1,
            "comparison": "logits_and_sigmoid_probabilities",
            "max_abs_diff": 0.0,
            "mean_abs_max_diff": 0.0,
            "tolerance": 1e-4,
            "logit_tolerance": 2e-4,
            "max_abs_probability_diff": 0.0,
            "mean_abs_max_probability_diff": 0.0,
            "probability_tolerance": 1e-4,
            "threshold_disagreements": 0,
            "compared_pixels": 1,
            "threshold_disagreement_fraction": 0.0,
            "passed": True,
        },
    )
    from train import models
    import torch

    class _FakeModel:
        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

    monkeypatch.setattr(
        models,
        "get_model",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pipeline,
        "_cpu_full_image_benchmark",
        lambda *args, **kwargs: {
            "images": [],
            "median_seconds": 0.01,
            "metrics": {
                "mean": {"iou": 0.5},
                "per_image": {"test": {"iou": 0.5}},
                "n_images": 1,
                "evaluation_mode": "onnx_cpu_full_resolution_tiled_test_once",
            },
        },
    )

    main(
        [
            "train-colony",
            "--data-root",
            str(source),
            "--output",
            str(tmp_path / "output"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--patches-per-image",
            "1",
            "--cpu-benchmark",
            "--no-pretrained",
        ]
    )

    report_path = next((tmp_path / "output").glob("runs/*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "output/dataset/dataset.json").read_text())
    assert {sample["subset"] for sample in manifest["samples"]} == {
        "train",
        "val",
        "test",
    }
    assert report["config"]["pretrained"] is False
    assert report["test"]["n_images"] == 1
    assert report["cpu_benchmark"]["median_seconds"] == 0.01
    assert (
        report["test"]["evaluation_mode"] == "onnx_cpu_full_resolution_tiled_test_once"
    )
    assert report["training_summary"]["best_epoch"] == 1


def test_black_background_pipeline_configures_scratch_unet_and_train_only_pairs(
    tmp_path, monkeypatch
):
    import train.dataset_adapters as adapters
    import train.training_pipeline as pipeline
    from train import models
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    source = tmp_path / "source"
    source.mkdir()
    coco_images, annotations = [], []
    for index in range(6):
        filename = f"dish-{index}.png"
        image = np.full((64, 64, 3), (40 + index, 60, 80), dtype=np.uint8)
        cv2.imwrite(str(source / filename), image)
        coco_images.append(
            {"id": index + 1, "file_name": filename, "width": 64, "height": 64}
        )
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "bbox": [18, 18, 24, 24],
                "category_id": 1 if index < 3 else 2,
            }
        )
    (source / "annot_COCO.json").write_text(
        json.dumps({"images": coco_images, "annotations": annotations}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        adapters, "_find_dish", lambda _image: {"cx": 32, "cy": 32, "r": 30}
    )

    observed = {}
    events = []
    original_fake_training = _fake_run_training

    def capture_training_config(cfg, progress_callback=None):
        observed["config"] = cfg
        events.append("train")
        return original_fake_training(cfg, progress_callback)

    monkeypatch.setattr(pipeline, "run_training", capture_training_config)
    patch_record_subsets = []

    def evaluate_training_subset(_model, records, *_args, **_kwargs):
        patch_record_subsets.append({record.subset for record in records})
        return {
            "mean": {"iou": 0.5, "dice": 0.66},
            "per_image": {record.id: {"iou": 0.5} for record in records},
            "n_images": len(records),
        }

    monkeypatch.setattr(pipeline, "evaluate_patch_records", evaluate_training_subset)
    monkeypatch.setattr(
        pipeline,
        "verify_onnx_cpu_parity",
        lambda *_args, **_kwargs: {"passed": True},
    )
    metric_row = {
        "tp": 1,
        "fp": 2,
        "fn": 3,
        "tn": 4,
        "confusion_matrix": [[4, 2], [3, 1]],
        "precision": 1 / 3,
        "recall": 1 / 4,
        "f1": 2 / 7,
        "iou": 1 / 6,
        "dice": 2 / 7,
        "accuracy": 0.5,
    }

    def fake_paired_evaluation(model_path, records, data_root, **_kwargs):
        del data_root
        label = (
            "baseline" if str(model_path).endswith("colony_seg.onnx") else "retrained"
        )
        events.append(label)
        sources = {
            record.id: record
            for record in records
            if record.kind == "source" and record.subset == "test"
        }
        variants = {
            record.source_id: record
            for record in records
            if record.kind == "augmented" and record.subset == "test"
        }
        ids = sorted(sources)
        backgrounds = {}
        for background, selected in (("original", sources), ("black", variants)):
            per_sample = {
                source_id: {
                    "source_id": source_id,
                    "record_id": record.id,
                    "image": str(record.image),
                    "mask": str(record.mask),
                    "width": 64,
                    "height": 64,
                    **metric_row,
                }
                for source_id, record in selected.items()
            }
            backgrounds[background] = {
                "source_ids": ids,
                "per_sample": per_sample,
                "summary": metric_row,
            }
        return {
            "path": str(model_path),
            "sha256": label,
            "source_ids": ids,
            "backgrounds": backgrounds,
        }

    monkeypatch.setattr(pipeline, "evaluate_paired_model", fake_paired_evaluation)

    class _FakeModel:
        def load_state_dict(self, _state):
            pass

        def to(self, _device):
            return self

    monkeypatch.setattr(models, "get_model", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})

    main(
        [
            "train-colony",
            "--data-root",
            str(source),
            "--output",
            str(tmp_path / "output"),
            "--model",
            "unet",
            "--black-background",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--patches-per-image",
            "1",
        ]
    )

    cfg = observed["config"]
    assert events == ["baseline", "train", "retrained"]
    assert cfg.model_name == "unet"
    assert cfg.pretrained is False
    assert cfg.resume_checkpoint is None
    assert cfg.require_cuda is True
    assert patch_record_subsets == [{"train"}, {"val"}]
    manifest = json.loads((tmp_path / "output/dataset/dataset.json").read_text())
    assert len(manifest["samples"]) == 12
    for variant in (
        sample for sample in manifest["samples"] if sample["kind"] == "augmented"
    ):
        source_record = next(
            sample
            for sample in manifest["samples"]
            if sample["id"] == variant["source_id"]
        )
        assert source_record["subset"] == variant["subset"]
    report_path = next((tmp_path / "output").glob("runs/*/report.json"))
    report = json.loads(report_path.read_text())
    comparison_path = next((tmp_path / "output").glob("runs/*/comparison.json"))
    comparison = json.loads(comparison_path.read_text())
    assert report["source_signature"]["unchanged"] is True
    assert comparison["baseline_training_overlap"] == "unknown"
    assert comparison["test_source_ids"] == comparison["split_ids"]["test"]
    assert comparison["seed"] == 42
    assert comparison["ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert comparison["threshold"] == 0.5
    assert comparison["configuration"]["architecture"] == "unet"
    assert comparison["configuration"]["initialization"] == "random"
    assert comparison["source_provenance"]["unchanged"] is True
    assert comparison["source_provenance"]["sha256"]
    paired_report = report["paired_background_experiment"]
    assert paired_report["test_source_ids"] == comparison["test_source_ids"]
    for model_name in ("baseline", "retrained"):
        for background in ("original", "black"):
            condition = comparison["models"][model_name]["backgrounds"][background]
            assert condition["source_ids"] == comparison["test_source_ids"]
            assert set(condition["per_sample"]) == set(comparison["test_source_ids"])
            assert (
                paired_report["models"][model_name]["backgrounds"][background][
                    "summary"
                ]["confusion_matrix"]
                == condition["summary"]["confusion_matrix"]
            )
    html_path = next((tmp_path / "output").glob("runs/*/comparison.html"))
    page = html_path.read_text(encoding="utf-8")
    assert all(source_id in page for source_id in comparison["test_source_ids"])
    assert "[[4, 2], [3, 1]]" in page
    assert report["comparison_json"] == str(comparison_path)
    assert report["comparison_html"] == str(html_path)


def test_model_promotion_refuses_to_overwrite_without_explicit_consent(
    tmp_path, monkeypatch
):
    import train.training_pipeline as pipeline

    source = tmp_path / "source"
    source.mkdir()
    # Reuse the integration test's real split importer while stubbing train/eval.
    coco_images = []
    annotations = []
    for index in range(6):
        filename = f"sample-{index}.png"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(source / filename), image)
        coco_images.append({"id": index + 1, "file_name": filename})
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "bbox": [5, 5, 10, 10],
                "category_id": 1,
            }
        )
    (source / "annot_COCO.json").write_text(
        json.dumps({"images": coco_images, "annotations": annotations}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "run_training", _fake_run_training)
    monkeypatch.setattr(pipeline, "evaluate_test_subset", _fake_evaluate)
    monkeypatch.setattr(
        pipeline,
        "evaluate_patch_records",
        lambda _model, records, *_args, **_kwargs: {
            "mean": {},
            "per_image": {record.id: {} for record in records},
            "n_images": len(records),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "verify_onnx_cpu_parity",
        lambda *_args, **_kwargs: {
            "providers": ["CPUExecutionProvider"],
            "patches": 1,
            "comparison": "logits_and_sigmoid_probabilities",
            "max_abs_diff": 0.0,
            "mean_abs_max_diff": 0.0,
            "tolerance": 1e-4,
            "logit_tolerance": 2e-4,
            "max_abs_probability_diff": 0.0,
            "mean_abs_max_probability_diff": 0.0,
            "probability_tolerance": 1e-4,
            "threshold_disagreements": 0,
            "compared_pixels": 1,
            "threshold_disagreement_fraction": 0.0,
            "passed": True,
        },
    )
    from train import models
    import torch

    class _FakeModel:
        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

    monkeypatch.setattr(
        models,
        "get_model",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    target = tmp_path / "existing.onnx"
    target.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pipeline.run_colony_training(
            data_root=source,
            output_root=tmp_path / "output",
            epochs=1,
            batch_size=1,
            patches_per_image=1,
            pretrained=False,
            model_promotion_path=target,
        )

    assert target.read_bytes() == b"keep me"
    assert not (tmp_path / "output").exists()


def test_resume_reuses_prepared_dataset_and_checkpoint_without_import(
    tmp_path, monkeypatch
):
    import train.training_pipeline as pipeline

    output_root = tmp_path / "output"
    dataset_root = output_root / "dataset"
    dataset_root.mkdir(parents=True)
    records = []
    samples = []
    for image_id, subset in ((1, "train"), (2, "val"), (3, "test")):
        image_name = f"dish-{image_id}.png"
        mask_name = f"dish-{image_id}_mask.png"
        cv2.imwrite(
            str(dataset_root / image_name), np.zeros((32, 32, 3), dtype=np.uint8)
        )
        cv2.imwrite(str(dataset_root / mask_name), np.zeros((32, 32), dtype=np.uint8))
        samples.append(
            {
                "id": f"dish-{image_id}",
                "kind": "source",
                "image": image_name,
                "mask": mask_name,
                "subset": subset,
            }
        )
        records.append((image_id, subset))
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "name": "prepared",
                "origin": "external",
                "mask_mode": "binary",
                "storage": "copy",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    split_payload = {
        "seed": 42,
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "groups": {
            subset: [
                {
                    "image_id": image_id,
                    "sample_id": f"dish-{image_id}",
                    "category_id": 1,
                }
            ]
            for image_id, subset in records
        },
    }
    (dataset_root / "split.json").write_text(
        json.dumps(split_payload), encoding="utf-8"
    )
    resume_run = output_root / "runs" / "partial"
    (resume_run / "checkpoints").mkdir(parents=True)
    (resume_run / "checkpoints" / "last.pt").write_bytes(b"checkpoint")
    (resume_run / "checkpoints" / "best.onnx").write_bytes(b"onnx model")
    source = tmp_path / "source"
    source.mkdir()
    coco = {
        "images": [{"id": i, "file_name": f"{i}.png"} for i in (1, 2, 3)],
        "annotations": [{"image_id": i, "bbox": [0, 0, 1, 1]} for i in (1, 2, 3)],
    }
    (source / "annot_COCO.json").write_text(json.dumps(coco), encoding="utf-8")
    monkeypatch.setattr(
        pipeline.CocoBboxImporter,
        "build",
        lambda *_args, **_kwargs: pytest.fail("resume must not repeat dataset import"),
    )
    received = {}

    def resume_training(cfg):
        received["config"] = cfg
        return {"best_epoch": 3}, [], [], resume_run

    monkeypatch.setattr(pipeline, "run_training", resume_training)
    monkeypatch.setattr(
        pipeline,
        "evaluate_patch_records",
        lambda _model, rows, *_args, **_kwargs: {
            "mean": {},
            "per_image": {row.id: {} for row in rows},
            "n_images": len(rows),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "verify_onnx_cpu_parity",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_test_subset",
        lambda *_args, **_kwargs: {"mean": {}, "per_image": {}, "n_images": 1},
    )
    from train import models
    import torch

    class _FakeModel:
        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

    monkeypatch.setattr(
        models,
        "get_model",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    report = pipeline.run_colony_training(
        data_root=source,
        output_root=output_root,
        pretrained=False,
        resume_run=resume_run,
    )

    assert (
        received["config"].resume_checkpoint == resume_run / "checkpoints" / "last.pt"
    )
    assert report["resumed_from"].endswith("checkpoints/last.pt")
    assert report["source_signature"]["unchanged"] is True


def _fake_run_training(cfg, progress_callback=None):
    run_dir = cfg.run_dir / "mobilenet_test"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best.pt").write_bytes(b"fake checkpoint")
    (checkpoint_dir / "best.onnx").write_bytes(b"fake onnx")
    return (
        {
            "model": cfg.model_name,
            "best_epoch": 1,
            "best_iou": 0.5,
            "best_val_dice": 0.66,
            "best_val_precision": 0.6,
            "best_val_recall": 0.7,
            "epochs": 1,
            "train_samples": 4,
            "val_samples": 1,
            "test_samples": 1,
            "img_size": cfg.img_size,
        },
        [{"seconds": 4.0}],
        [{"iou": 0.5, "dice": 0.66}],
        run_dir,
    )


def _fake_evaluate(model, data_root, **kwargs):
    return {"mean": {"iou": 0.5, "dice": 0.66}, "per_image": {}, "n_images": 1}
