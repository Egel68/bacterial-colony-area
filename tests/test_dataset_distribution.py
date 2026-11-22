import json

import cv2
import numpy as np

from train.dataset_adapters import load_manifest
from train.dataset_archive import pack_training_dataset, unpack_training_archive
from train.dataset_distribution import (
    prepare_training_dataset,
    validate_prepared_dataset,
)


def _make_source(root):
    root.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(6):
        filename = f"dish-{index}.jpg"
        image = np.full((64, 72, 3), (30 + index, 60, 90), dtype=np.uint8)
        assert cv2.imwrite(str(root / filename), image)
        images.append(
            {"id": index + 1, "file_name": filename, "width": 72, "height": 64}
        )
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "bbox": [18, 16, 20, 20],
                "category_id": 1 if index < 3 else 2,
            }
        )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    }
    (root / "annot_COCO.json").write_text(json.dumps(coco), encoding="utf-8")
    return coco


def test_prepare_materializes_full_dataset_and_repeatable_group_split(
    tmp_path, monkeypatch
):
    import train.dataset_adapters as adapters

    source = tmp_path / "coco"
    coco = _make_source(source)
    monkeypatch.setattr(
        adapters, "_find_dish", lambda _image: {"cx": 36, "cy": 32, "r": 30}
    )

    output_a = tmp_path / "prepared-a"
    output_b = tmp_path / "prepared-b"
    summary = prepare_training_dataset(source, output_a, seed=42, black_background=True)
    prepare_training_dataset(source, output_b, seed=42, black_background=True)

    assert summary["annotated_images"] == len(coco["images"]) == 6
    assert summary["annotations"] == len(coco["annotations"]) == 6
    assert summary["sources"] == summary["black_background_variants"] == 6
    assert summary["samples"] == 12
    assert summary["split_counts"] == {"train": 2, "val": 2, "test": 2}
    assert (output_a / "annot_COCO.json").read_bytes() == (
        source / "annot_COCO.json"
    ).read_bytes()
    assert json.loads((output_a / "split.json").read_text()) == json.loads(
        (output_b / "split.json").read_text()
    )

    split = json.loads((output_a / "split.json").read_text(encoding="utf-8"))
    assignments = {
        group["sample_id"]: subset
        for subset, groups in split["groups"].items()
        for group in groups
    }
    manifest = load_manifest(output_a)
    sources = {
        record.id: record for record in manifest.samples if record.kind == "source"
    }
    variants = [record for record in manifest.samples if record.kind == "augmented"]
    assert set(assignments) == set(sources) == {f"dish-{i}" for i in range(6)}
    assert (
        len(
            {
                group["image_id"]
                for groups in split["groups"].values()
                for group in groups
            }
        )
        == 6
    )
    for source_id, source_record in sources.items():
        assert source_record.subset == assignments[source_id]
        original_bytes = (source / f"{source_id}.jpg").read_bytes()
        assert (output_a / source_record.image).read_bytes() == original_bytes
    for variant in variants:
        assert variant.source_id in sources
        assert variant.subset == assignments[variant.source_id]
        assert variant.subset == sources[variant.source_id].subset

    checked, checked_split = validate_prepared_dataset(
        output_a, require_black_background=True
    )
    assert len(checked.samples) == 12
    assert checked_split == split


def test_prepare_can_omit_optional_black_background_variants(tmp_path):
    source = tmp_path / "coco"
    _make_source(source)
    output = tmp_path / "prepared"

    summary = prepare_training_dataset(source, output, black_background=False)

    assert summary["sources"] == 6
    assert summary["black_background_variants"] == 0
    assert summary["samples"] == 6
    assert len(load_manifest(output).samples) == 6


def test_archive_is_deterministic_and_round_trips_all_file_hashes(tmp_path):
    source = tmp_path / "coco"
    _make_source(source)
    prepared = tmp_path / "prepared"
    prepare_training_dataset(source, prepared, black_background=False)
    archives = tmp_path / "archives"
    archive_a = archives / "snapshot-a.tar.zst"
    archive_b = archives / "snapshot-b.tar.zst"

    report = pack_training_dataset(prepared, archive_a)
    pack_training_dataset(prepared, archive_b)

    assert report["compression_level"] == 19
    assert report["roundtrip_sha256_verified"] is True
    assert report["archive_bytes"] == archive_a.stat().st_size
    assert report["uncompressed_bytes"] > report["archive_bytes"] > 0
    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert (archives / "snapshot-a.dataset.json").read_bytes() == (
        prepared / "dataset.json"
    ).read_bytes()
    assert (archives / "snapshot-a.split.json").read_bytes() == (
        prepared / "split.json"
    ).read_bytes()
    assert (archives / "snapshot-a.annot_COCO.json").read_bytes() == (
        prepared / "annot_COCO.json"
    ).read_bytes()

    extracted = tmp_path / "extracted"
    summary = unpack_training_archive(archive_a, extracted)
    assert summary["samples"] == 6
    assert summary["sources"] == 6
    assert summary["black_background_variants"] == 0
    for path in prepared.rglob("*"):
        if path.is_file():
            relative = path.relative_to(prepared)
            assert (extracted / relative).read_bytes() == path.read_bytes()
