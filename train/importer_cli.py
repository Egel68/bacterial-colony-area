"""CLI импортёра внешнего датасета 22022540 в контракт данных проекта.

Запуск:
    uv run import-22022540 --data-root datasets/22022540 --output datasets/22022540_imported [--crop]
"""

import argparse
import logging
from pathlib import Path

from .dataset_adapters import CocoBboxImporter

log = logging.getLogger(__name__)


def _verify_counts(source_root, output_manifest):
    """Сверка числа боксов из COCO с YOLO/VOC-дублями.

    При расхождении числа аннотаций для изображения между форматами больше порога
    выдаётся предупреждение. При отсутствии ZIP/YAML-дублей проверка пропускается.
    """
    import io
    import json
    import zipfile

    root = Path(source_root)
    coco_path = root / "annot_COCO.json"
    yolo_zip = root / "annot_YOLO.zip"
    voc_zip = root / "annot_VOC_XML.zip"
    if not coco_path.is_file() or not (yolo_zip.is_file() or voc_zip.is_file()):
        log.info("Верификация пропущена: нет COCO или ZIP-дублей (YOLO/VOC)")
        return

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    counts = {}
    for ann in coco.get("annotations", []):
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    coco_by_file = {
        im["file_name"]: counts.get(im["id"], 0) for im in coco.get("images", [])
    }

    def count_yolo(zip_path):
        """Число боксов = число строк в `{stem}.txt` (YOLO нормализованный)."""
        with zipfile.ZipFile(zip_path) as zf:
            return {
                Path(name).stem: len(zf.read(name).decode("utf-8", "ignore").strip().splitlines())
                for name in zf.namelist()
                if name.lower().endswith(".txt")
            }

    def count_voc(zip_path):
        """Число боксов = число тегов `<object>` в `{stem}.xml` (VOC)."""
        with zipfile.ZipFile(zip_path) as zf:
            counts = {}
            for name in zf.namelist():
                if not name.lower().endswith(".xml"):
                    continue
                xml = zf.read(name).decode("utf-8", "ignore")
                counts[Path(name).stem] = xml.count("<object>")
            return counts

    ref_counts = {}
    if yolo_zip.is_file():
        ref_counts = count_yolo(yolo_zip)
    elif voc_zip.is_file():
        ref_counts = count_voc(voc_zip)
    else:
        ref_counts = None

    mismatches = 0
    for fname, coco_cnt in coco_by_file.items():
        stem = Path(fname).stem
        ref_cnt = ref_counts.get(stem) if ref_counts else None
        if ref_cnt is not None and abs(coco_cnt - ref_cnt) > 5:
            mismatches += 1
            log.warning("Расхождение аннотаций %s: COCO=%d, дубль=%d", fname, coco_cnt, ref_cnt)
    log.info(
        "Верификация завершена: %d расхождений > порога из %d изображений",
        mismatches,
        len(coco_by_file),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Import 22022540 dataset into project contract")
    parser.add_argument("--data-root", required=True, help="Path to the 22022540 source dir")
    parser.add_argument("--output", default=None, help="Output dataset dir (defaults to <root>_imported)")
    parser.add_argument("--crop", action="store_true", help="Also produce cropped-to-dish pairs")
    parser.add_argument("--verify", action="store_true", help="Cross-check COCO vs YOLO/VOC counts")
    args = parser.parse_args()

    manifest = CocoBboxImporter().build(
        data_root=args.data_root,
        output_dir=args.output,
        crop=args.crop,
    )

    if args.verify:
        _verify_counts(args.data_root, manifest)

    n_source = sum(1 for s in manifest.samples if s.kind == "source")
    n_cropped = sum(1 for s in manifest.samples if s.kind == "cropped")
    out = args.output or f"{args.data_root}_imported"
    log.info(
        "Импорт завершён: изображений обработано=%d, записей source=%d, cropped=%d, "
        "dataset.json=%s",
        len(manifest.samples),
        n_source,
        n_cropped,
        out + "/dataset.json",
    )


if __name__ == "__main__":
    main()