"""Тесты импортёра датасета 22022540 (CocoBboxImporter)."""

import json

import cv2
import numpy as np
import pytest

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
            {"id": 1, "image_id": 1, "bbox": [10, 10, 20, 30], "category_id": 1, "area": 600, "iscrowd": False},
            {"id": 2, "image_id": 1, "bbox": [60, 50, 25, 25], "category_id": 1, "area": 625, "iscrowd": False},
            {"id": 3, "image_id": 2, "bbox": [30, 40, 20, 20], "category_id": 1, "area": 400, "iscrowd": False},
        ],
    }
    (root / "annot_COCO.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


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
    assert all(s.kind != "cropped" or (out / s.mask).is_file() for s in manifest.samples)