"""Slow synthetic end-to-end check for the black-background training experiment."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="black-background experiment requires CUDA"
)
def test_synthetic_coco_black_background_experiment_is_end_to_end_and_read_only(
    tmp_path, monkeypatch
):
    import train.dataset_adapters as adapters
    import train.training_pipeline as pipeline

    source = tmp_path / "coco-source"
    output = tmp_path / "experiment-output"
    source.mkdir()
    images, annotations = [], []
    for index in range(6):
        filename = f"dish-{index}.png"
        image = np.full((64, 64, 3), (30 + index, 70, 110), dtype=np.uint8)
        cv2.circle(image, (32, 32), 30, (140, 150, 160), thickness=-1)
        assert cv2.imwrite(str(source / filename), image)
        images.append(
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
    coco_path = source / "annot_COCO.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "type-a"},
                    {"id": 2, "name": "type-b"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        adapters, "_find_dish", lambda _image: {"cx": 32, "cy": 32, "r": 30}
    )
    original_config = pipeline.TrainingConfig

    def single_worker_config(**kwargs):
        cfg = original_config(**kwargs)
        cfg.num_workers = 0
        return cfg

    monkeypatch.setattr(pipeline, "TrainingConfig", single_worker_config)
    source_signature_before = pipeline._tree_signature(source)
    adapters.CocoBboxImporter().build(
        data_root=source,
        output_dir=output / "dataset",
        split=True,
        seed=42,
        split_ratios=(0.70, 0.15, 0.15),
        black_background=True,
    )

    report = pipeline.run_colony_training(
        data_root=source,
        output_root=output,
        seed=42,
        ratios=(0.70, 0.15, 0.15),
        epochs=1,
        batch_size=1,
        patches_per_image=1,
        img_size=512,
        pretrained=False,
        model_name="unet",
        black_background=True,
        reuse_prepared_dataset=True,
    )

    source_signature_after = pipeline._tree_signature(source)
    assert source_signature_after == source_signature_before
    assert report["source_signature"]["unchanged"] is True
    assert report["prepared_dataset_reused"] is True
    assert report["source_signature"]["before_import"] == source_signature_before
    assert report["source_signature"]["after_run"] == source_signature_after
    assert report["training_summary"]["pretrained"] is False
    assert report["training_summary"]["initialization_source"] == "random"
    assert report["training_summary"]["resume_mode"] == "fresh"
    provenance = report["training_summary"]["initialization_provenance"]
    assert provenance["pretrained_used_for_current_run"] is False
    assert provenance["resume_mode"] == "fresh"

    dataset_root = output / "dataset"
    manifest = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
    records = manifest["samples"]
    assert len(records) == 12
    sources = {record["id"]: record for record in records if record["kind"] == "source"}
    variants = {
        record["source_id"]: record
        for record in records
        if record["kind"] == "augmented"
    }
    assert set(sources) == set(variants)
    assert all(
        sources[key]["subset"] == variant["subset"] for key, variant in variants.items()
    )
    for variant in variants.values():
        assert (dataset_root / variant["image"]).is_file()
        assert variant["mask"] == sources[variant["source_id"]]["mask"]

    split_ids = {
        subset: sorted(
            record["id"]
            for record in records
            if record["kind"] == "source" and record["subset"] == subset
        )
        for subset in ("train", "val", "test")
    }
    assert all(split_ids.values())
    comparison = json.loads(Path(report["comparison_json"]).read_text(encoding="utf-8"))
    assert comparison["seed"] == 42
    assert comparison["ratios"] == {"train": 0.70, "val": 0.15, "test": 0.15}
    assert comparison["split_ids"] == split_ids
    assert comparison["test_source_ids"] == split_ids["test"]
    assert comparison["source_provenance"]["sha256"] == source_signature_before
    assert comparison["source_provenance"]["unchanged"] is True
    assert comparison["baseline_training_overlap"] == "unknown"
    assert comparison["models"]["baseline"]["training_membership"] == "unknown"
    assert comparison["models"]["retrained"]["initialization"] == "random"
    assert (
        comparison["models"]["retrained"]["initialization_provenance"]["resume_mode"]
        == "fresh"
    )

    for model_name in ("baseline", "retrained"):
        model = comparison["models"][model_name]
        for background in ("original", "black"):
            condition = model["backgrounds"][background]
            assert condition["source_ids"] == split_ids["test"]
            assert set(condition["per_sample"]) == set(split_ids["test"])
            aggregate = condition["summary"]
            assert aggregate["confusion_matrix"] == [
                [aggregate["tn"], aggregate["fp"]],
                [aggregate["fn"], aggregate["tp"]],
            ]
            assert aggregate["precision"] == pytest.approx(
                aggregate["tp"] / (aggregate["tp"] + aggregate["fp"])
                if aggregate["tp"] + aggregate["fp"]
                else 0.0
            )
            assert aggregate["recall"] == pytest.approx(
                aggregate["tp"] / (aggregate["tp"] + aggregate["fn"])
                if aggregate["tp"] + aggregate["fn"]
                else 0.0
            )
            for sample in condition["per_sample"].values():
                assert sample["confusion_matrix"] == [
                    [sample["tn"], sample["fp"]],
                    [sample["fn"], sample["tp"]],
                ]
                assert sample["width"] == sample["height"] == 64

    assert Path(report["comparison_html"]).is_file()
    assert (output / "runs").is_dir()
