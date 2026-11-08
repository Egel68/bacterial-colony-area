"""Тесты импортёра датасета 22022540 (CocoBboxImporter)."""

import json
import hashlib

import cv2
import numpy as np

from train.dataset_adapters import CocoBboxImporter, load_manifest


def _make_source(root):
    """Создание мини-источника COCO-bbox: 3 изображения, 2 из которых с боксами."""
    root = root / "src"
    root.mkdir(parents=True)

    # Изображение 1: два бокса
    img1 = np.zeros((100, 120, 3), dtype=np.uint8)
    img1[:] = (60, 60, 60)
    cv2.imwrite(str(root / "sp01_img01.jpg"), img1)

    # Изображение 2: один бокс на пропуск
    img2 = np.zeros((100, 120, 3), dtype=np.uint8)
    img2[:] = (60, 60, 60)
    cv2.imwrite(str(root / "sp02_img02.jpg"), img2)

    # Изображение 3: без боксов — должно быть пропущено
    img3 = np.zeros((100, 120, 3), dtype=np.uint8)
    img3[:] = (60, 60, 60)
    cv2.imwrite(str(root / "sp03_imgnone.jpg"), img3)

    coco = {
        "type": "instances",
        "categories": [{"id": 1, "name": "sp01", "supercategory": "cfu"}],
        "images": [
            {"file_name": "sp01_img01.jpg", "width": 120, "height": 100, "id": 1},
            {"file_name": "sp02_img02.jpg", "width": 120, "height": 100, "id": 2},
            {"file_name": "sp03_imgnone.jpg", "width": 120, "height": 100, "id": 3},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [10, 10, 20, 30],
                "category_id": 1,
                "area": 600,
                "iscrowd": False,
            },
            {
                "id": 2,
                "image_id": 1,
                "bbox": [60, 50, 25, 25],
                "category_id": 1,
                "area": 625,
                "iscrowd": False,
            },
            {
                "id": 3,
                "image_id": 2,
                "bbox": [30, 40, 20, 20],
                "category_id": 1,
                "area": 400,
                "iscrowd": False,
            },
        ],
    }
    (root / "annot_COCO.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


def _add_third_annotated_image(src):
    coco_path = src / "annot_COCO.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["annotations"].append(
        {
            "id": 4,
            "image_id": 3,
            "bbox": [40, 30, 15, 15],
            "category_id": 1,
            "area": 225,
            "iscrowd": False,
        }
    )
    coco_path.write_text(json.dumps(coco), encoding="utf-8")


def test_importer_materializes_manifest(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"

    manifest = CocoBboxImporter().build(data_root=src, output_dir=out)

    # только изображения с боксами
    assert len(manifest.samples) == 2
    ids = {s.id for s in manifest.samples}
    assert ids == {"sp01_img01", "sp02_img02"}

    # storage / origin / mask_mode
    assert manifest.origin == "external"
    assert manifest.mask_mode == "binary"
    assert manifest.storage == "copy"

    # файлы существуют, mask — бинарная 255/0
    for sample in manifest.samples:
        img_path = out / sample.image
        mask_path = out / sample.mask
        assert img_path.is_file()
        assert mask_path.is_file()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert set(np.unique(mask)) <= {0, 255}

    # маска sp01_img01 содержит оба эллипса
    m1 = cv2.imread(str(out / "source/sp01_img01_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert (m1 == 255).sum() >= 600 * 0.4


def test_importer_roundtrip_via_load_manifest(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    CocoBboxImporter().build(data_root=src, output_dir=out)

    loaded = load_manifest(out)
    assert len(loaded.samples) == 2
    assert (out / "dataset.json").is_file()


def test_importer_cropped_variant(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    manifest = CocoBboxImporter().build(data_root=src, output_dir=out, crop=True)

    # в ручном тесте авто-поиск чашки может не сработать на синтетике — допускаем
    # наличие cropped-записей только если dish найден; гарантированно должны быть source
    kinds = {s.id: s.kind for s in manifest.samples}
    assert any(kind == "source" for kind in kinds.values())
    assert all(
        s.kind != "cropped" or (out / s.mask).is_file() for s in manifest.samples
    )


def test_importer_persists_deterministic_split_without_changing_source(tmp_path):
    src = _make_source(tmp_path)
    _add_third_annotated_image(src)
    source_bytes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in src.iterdir()
        if path.is_file()
    }
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    manifest_a = CocoBboxImporter().build(
        data_root=src, output_dir=out_a, split=True, seed=42
    )
    manifest_b = CocoBboxImporter().build(
        data_root=src, output_dir=out_b, split=True, seed=42
    )

    assignments_a = {sample.id: sample.subset for sample in manifest_a.samples}
    assignments_b = {sample.id: sample.subset for sample in manifest_b.samples}
    assert assignments_a == assignments_b
    assert set(assignments_a.values()) <= {"train", "val", "test"}
    assert all(assignments_a.values())
    assert json.loads((out_a / "split.json").read_text()) == json.loads(
        (out_b / "split.json").read_text()
    )
    assert {sample.id for sample in load_manifest(out_a).samples} == set(assignments_a)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in src.iterdir()
        if path.is_file()
    } == source_bytes


def test_importer_split_keeps_source_and_cropped_records_in_same_subset(
    tmp_path, monkeypatch
):
    import train.dataset_adapters as adapters

    src = _make_source(tmp_path)
    _add_third_annotated_image(src)
    monkeypatch.setattr(
        adapters, "_find_dish", lambda image: {"cx": 60, "cy": 50, "r": 45}
    )
    out = tmp_path / "out"

    manifest = CocoBboxImporter().build(
        data_root=src, output_dir=out, crop=True, split=True
    )

    for sample in manifest.samples:
        if sample.kind != "cropped":
            continue
        source_id = sample.id.removesuffix("_cropped")
        source_record = next(item for item in manifest.samples if item.id == source_id)
        assert sample.subset == source_record.subset


def test_split_import_rejects_incomplete_source_images(tmp_path, monkeypatch):
    import pytest

    src = _make_source(tmp_path)
    _add_third_annotated_image(src)
    (src / "sp02_img02.jpg").unlink()
    out = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="referenced by COCO is missing"):
        CocoBboxImporter().build(data_root=src, output_dir=out, split=True)

    assert not (out / "dataset.json").exists()


def test_split_import_rejects_output_nested_under_read_only_source(tmp_path):
    import pytest

    src = _make_source(tmp_path)
    _add_third_annotated_image(src)
    out = src / "generated"

    with pytest.raises(ValueError, match="must be disjoint"):
        CocoBboxImporter().build(data_root=src, output_dir=out, split=True)

    assert not out.exists()


def test_black_background_pairs_preserve_split_geometry_and_source(
    tmp_path, monkeypatch
):
    import train.dataset_adapters as adapters

    src = _make_source(tmp_path)
    _add_third_annotated_image(src)
    dish = {"cx": 60, "cy": 50, "r": 32}
    monkeypatch.setattr(adapters, "_find_dish", lambda _image: dish)

    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in src.iterdir()
        if path.is_file()
    }
    original_writer = CocoBboxImporter._write_black_background

    def assert_split_was_persisted(out, sample_id, image):
        split_path = out / "split.json"
        assert split_path.is_file()
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        assert split_data["seed"] == 42
        assert (
            len([group for groups in split_data["groups"].values() for group in groups])
            == 3
        )
        return original_writer(out, sample_id, image)

    monkeypatch.setattr(
        CocoBboxImporter,
        "_write_black_background",
        staticmethod(assert_split_was_persisted),
    )

    out_a, out_b = tmp_path / "black-a", tmp_path / "black-b"
    manifest_a = CocoBboxImporter().build(
        data_root=src, output_dir=out_a, split=True, seed=42, black_background=True
    )
    manifest_b = CocoBboxImporter().build(
        data_root=src, output_dir=out_b, split=True, seed=42, black_background=True
    )

    sources = {
        record.id: record for record in manifest_a.samples if record.kind == "source"
    }
    variants = [record for record in manifest_a.samples if record.kind == "augmented"]
    assert len(sources) == len(variants) == 3
    assert {record.kind for record in manifest_a.samples} == {"source", "augmented"}
    for variant in variants:
        source = sources[variant.source_id]
        assert source.subset == variant.subset
        assert source.dish == variant.dish
        assert source.mask == variant.mask
        assert variant.id == f"{source.id}__black_background"

        source_image = cv2.imread(str(out_a / source.image), cv2.IMREAD_COLOR)
        black_image = cv2.imread(str(out_a / variant.image), cv2.IMREAD_COLOR)
        assert source_image.shape == black_image.shape
        yy, xx = np.ogrid[: source_image.shape[0], : source_image.shape[1]]
        inside = (xx - dish["cx"]) ** 2 + (yy - dish["cy"]) ** 2 <= dish["r"] ** 2
        assert np.array_equal(source_image[inside], black_image[inside])
        assert np.all(black_image[~inside] == 0)
        assert np.array_equal(
            cv2.imread(str(out_a / source.mask), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(out_a / variant.mask), cv2.IMREAD_GRAYSCALE),
        )

    split = json.loads((out_a / "split.json").read_text(encoding="utf-8"))
    split_by_id = {
        group["sample_id"]: subset
        for subset, groups in split["groups"].items()
        for group in groups
    }
    assert {
        sample_id: record.subset for sample_id, record in sources.items()
    } == split_by_id
    assert {
        (record.id, record.source_id, record.subset)
        for record in manifest_a.samples
        if record.kind == "augmented"
    } == {
        (record.id, record.source_id, record.subset)
        for record in manifest_b.samples
        if record.kind == "augmented"
    }
    roundtrip = load_manifest(out_a)
    assert {
        record.id: record.source_id
        for record in roundtrip.samples
        if record.kind == "augmented"
    } == {record.id: record.source_id for record in variants}
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in src.iterdir()
        if path.is_file()
    } == source_hashes


def test_black_background_requires_split_and_valid_dish_geometry(tmp_path, monkeypatch):
    import pytest
    import train.dataset_adapters as adapters

    src = _make_source(tmp_path)
    _add_third_annotated_image(src)
    importer = CocoBboxImporter()
    with pytest.raises(ValueError, match="require a persisted"):
        importer.build(
            data_root=src, output_dir=tmp_path / "no-split", black_background=True
        )

    monkeypatch.setattr(adapters, "_find_dish", lambda _image: None)
    out = tmp_path / "missing-dish"
    with pytest.raises(ValueError, match="sp01_img01"):
        importer.build(data_root=src, output_dir=out, split=True, black_background=True)
    assert not (out / "dataset.json").exists()
    assert not (out / "augmented").exists()
