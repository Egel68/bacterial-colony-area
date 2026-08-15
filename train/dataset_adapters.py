"""Адаптеры источников данных обучения.

Каждый адаптер реализует единый интерфейс `build(data_root) -> DatasetManifest`
и знает про одну конкретную структуру папок. Ядро обучения и фабрика
`load_manifest` не зависят от форматов — ветвление по структурам полностью
скрыто здесь. Это позволяет подключать новые источники данных точечно.

Задел на будущее:
- внешние импортёры (COCO/VOC/произвольные пары) будут регистрироваться через
  декоратор `@register_importer`, материализовать файлы (`storage="copy"`) и
  порождать манифест этой же схемы — без изменения ядра обучения.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset_manifest import (
    DishGeometry,
    DatasetManifest,
    Origin,
    SampleKind,
    SampleRecord,
    StorageMode,
)

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class BaseAdapter(ABC):
    """Базовый интерфейс адаптера источника данных."""

    @abstractmethod
    def build(self, data_root: Path) -> DatasetManifest:
        """Построение манифеста по корню датасета."""
        raise NotImplementedError


def _image_files(directory: Path) -> list[Path]:
    """Отсортированный список файлов изображений в директории."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


class ManifestAdapter(BaseAdapter):
    """Чтение готового `dataset.json` из корня датасета."""

    def build(self, data_root: Path) -> DatasetManifest:
        import json

        manifest_path = Path(data_root) / "dataset.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))

        samples = []
        for entry in raw.get("samples", []):
            dish = None
            if entry.get("dish"):
                dish = DishGeometry(
                    cx=entry["dish"]["cx"],
                    cy=entry["dish"]["cy"],
                    r=entry["dish"]["r"],
                )
            samples.append(
                SampleRecord(
                    id=entry["id"],
                    kind=entry.get("kind", SampleKind.SOURCE.value),
                    image=Path(entry["image"]),
                    mask=Path(entry["mask"]),
                    subset=entry.get("subset"),
                    dish=dish,
                )
            )

        manifest = DatasetManifest(
            name=raw.get("name", manifest_path.parent.name),
            origin=raw.get("origin", Origin.EXTERNAL.value),
            mask_mode=raw.get("mask_mode", "binary"),
            storage=raw.get("storage", StorageMode.REFERENCE.value),
            samples=samples,
        )
        manifest.validate(manifest_path.parent)
        return manifest


class LabelingAdapter(BaseAdapter):
    """Сканирование структуры сессии разметки.

    Строит манифест в памяти в режиме `storage="reference"` (без копирования
    файлов). Пары ищутся как `source/*` + `masks/{stem}_mask.*` и
    `cropped/*` + `cropped_masks/{stem}_cropped_mask.*`.
    """

    def build(self, data_root: Path) -> DatasetManifest:
        root = Path(data_root)
        samples: list[SampleRecord] = []

        for image_path in _image_files(root / "source"):
            mask_path = root / "masks" / f"{image_path.stem}_mask{image_path.suffix}"
            if not mask_path.is_file():
                continue
            samples.append(
                SampleRecord(
                    id=image_path.stem,
                    kind=SampleKind.SOURCE.value,
                    image=image_path.relative_to(root),
                    mask=mask_path.relative_to(root),
                )
            )

        for image_path in _image_files(root / "cropped"):
            mask_path = root / "cropped_masks" / f"{image_path.stem}_mask{image_path.suffix}"
            if not mask_path.is_file():
                continue
            samples.append(
                SampleRecord(
                    id=image_path.stem,
                    kind=SampleKind.CROPPED.value,
                    image=image_path.relative_to(root),
                    mask=mask_path.relative_to(root),
                )
            )

        return DatasetManifest(
            name=root.name,
            origin=Origin.LABELING.value,
            mask_mode="binary",
            storage=StorageMode.REFERENCE.value,
            samples=samples,
        )


class PairsAdapter(BaseAdapter):
    """Чтение легаси-структуры `images/` + `masks/` по совпадающим именам."""

    def build(self, data_root: Path) -> DatasetManifest:
        root = Path(data_root)
        samples: list[SampleRecord] = []

        for image_path in _image_files(root / "images"):
            mask_path = root / "masks" / f"{image_path.name}"
            if not mask_path.is_file():
                continue
            samples.append(
                SampleRecord(
                    id=image_path.stem,
                    kind=SampleKind.SOURCE.value,
                    image=image_path.relative_to(root),
                    mask=mask_path.relative_to(root),
                )
            )

        return DatasetManifest(
            name=root.name,
            origin=Origin.LEGACY.value,
            mask_mode="binary",
            storage=StorageMode.REFERENCE.value,
            samples=samples,
        )


def _rasterize_ellipse(bbox: list[float], shape: tuple[int, int]) -> np.ndarray:
    """Растризация бокса во вписанный эллипс в маску 255/0.

    Бокс задан как [x, y, w, h] (COCO). Эллипс вписан в бокс: центр в середине,
    полуоси равны w/2 и h/2. Так форма ближе к реальной круглой колонии,
    чем заливка прямоугольника (меньше ложноположительного фона в углах).
    """
    x, y, w, h = (float(v) for v in bbox[:4])
    center = (int(x + w / 2), int(y + h / 2))
    axes = (max(int(w / 2), 1), max(int(h / 2), 1))
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _find_dish(image: np.ndarray) -> Optional[dict]:
    """Автопоиск геометрии чашки Петри через `analysis.colony_detector`.

    Возвращает `{cx, cy, r}` или None, если чашка не найдена.
    """
    try:
        from analysis.colony_detector import ColonyDetector

        _, petri = ColonyDetector().detect_petri_dish(image)
    except Exception:
        petri = None
    if petri is None:
        return None
    h, w = image.shape[:2]
    if petri.cx < 0 or petri.cy < 0 or petri.radius <= 0:
        return None
    return {"cx": int(petri.cx), "cy": int(petri.cy), "r": int(petri.radius)}


class CocoBboxImporter(BaseAdapter):
    """Импорт внешнего датасета формата COCO-bbox в контракт данных проекта.

    Источником служит папка вида `datasets/22022540/` с `annot_COCO.json` и
    снимками чашек Петри. Импортёр:
    - растризует боксы колоний во вписанные эллипсы бинарной маской (255=колония);
    - опционально формирует `kind=cropped` — обрезку по кругу чашки с чёрным фоном;
    - материализует самодостаточную папку датасета (`storage="copy"`) с `dataset.json`,
      читаемую существующим `ManifestAdapter` без изменения ядра обучения.

    Источник только читается; выход пишется в `output_dir`.
    """

    def build(
        self,
        data_root: Path,
        output_dir: Optional[Path] = None,
        crop: bool = False,
    ) -> DatasetManifest:
        """Запуск импорта по корню источника в `output_dir`.

        Если `output_dir` не задан, используется соседняя папка
        `<root>_imported`. Возвращается манифест выходного датасета.
        """
        import json

        source = Path(data_root)
        coco_path = source / "annot_COCO.json"
        if not coco_path.is_file():
            raise FileNotFoundError(f"annot_COCO.json not found in {source}")

        out = Path(output_dir) if output_dir else Path(f"{source}_imported")
        out.mkdir(parents=True, exist_ok=True)

        raw = json.loads(coco_path.read_text(encoding="utf-8"))

        images_by_id = {im["id"]: im for im in raw.get("images", [])}
        boxes_by_image: dict[int, list[tuple[int, list[float]]]] = {}
        for ann in raw.get("annotations", []):
            image_id = ann["image_id"]
            boxes_by_image.setdefault(image_id, []).append(
                (ann.get("category_id", 1), list(ann["bbox"]))
            )

        samples: list[SampleRecord] = []
        processed = 0
        for image_id, image_meta in images_by_id.items():
            boxes = boxes_by_image.get(image_id)
            if not boxes:
                continue

            filename = image_meta["file_name"]
            src_img = source / filename
            if not src_img.is_file():
                continue

            image = cv2.imread(str(src_img))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for _, bbox in boxes:
                mask |= _rasterize_ellipse(bbox, image.shape)

            # Геометрия чашки нужна для cropped-варианта и как метаданное source.
            # Для source-only-импорта (обучение) авто-поиск пропускается — это
            # заметно ускоряет импорт без потери обучающих масок.
            dish = _find_dish(image) if crop else None

            rel_img, rel_mask = self._write_source(out, filename, image, mask)
            samples.append(
                SampleRecord(
                    id=Path(filename).stem,
                    kind=SampleKind.SOURCE.value,
                    image=rel_img,
                    mask=rel_mask,
                    dish=(
                        DishGeometry(cx=dish["cx"], cy=dish["cy"], r=dish["r"])
                        if dish
                        else None
                    ),
                )
            )

            if crop and dish is not None:
                rel_img_c, rel_mask_c = self._write_cropped(
                    out, filename, image, mask, dish
                )
                samples.append(
                    SampleRecord(
                        id=f"{Path(filename).stem}_cropped",
                        kind=SampleKind.CROPPED.value,
                        image=rel_img_c,
                        mask=rel_mask_c,
                        dish=DishGeometry(cx=dish["cx"], cy=dish["cy"], r=dish["r"]),
                    )
                )

            processed += 1

        manifest = DatasetManifest(
            name=out.name,
            origin=Origin.EXTERNAL.value,
            mask_mode="binary",
            storage=StorageMode.COPY.value,
            samples=samples,
        )
        self._write_dataset_json(out, manifest)
        return manifest

    def _write_source(
        self,
        out: Path,
        filename: str,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[Path, Path]:
        """Запись полноразмерной пары изображение+маска в `out/source/`."""
        (out / SampleKind.SOURCE.value).mkdir(exist_ok=True)
        stem = Path(filename).stem
        rel_img = Path(SampleKind.SOURCE.value) / f"{stem}{Path(filename).suffix}"
        rel_mask = Path(SampleKind.SOURCE.value) / f"{stem}_mask.png"
        cv2.imwrite(str(out / rel_img), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / rel_mask), mask)
        return rel_img, rel_mask

    def _write_cropped(
        self,
        out: Path,
        filename: str,
        image: np.ndarray,
        mask: np.ndarray,
        dish: dict,
    ) -> tuple[Path, Path]:
        """Запись обрезки по чашке с чёрным фоном вне круга (`kind=cropped`)."""
        (out / SampleKind.CROPPED.value).mkdir(exist_ok=True)
        stem = Path(filename).stem
        cx, cy, r = dish["cx"], dish["cy"], dish["r"]

        crop_img = image.copy()
        crop_img[
            (np.arange(image.shape[0])[:, None] - cy) ** 2
            + (np.arange(image.shape[1])[None, :] - cx) ** 2
            > r * r
        ] = 0
        crop_mask = cv2.bitwise_and(mask, mask, mask=self._circle_mask(image.shape, dish))

        rel_img = Path(SampleKind.CROPPED.value) / f"{stem}_cropped.jpg"
        rel_mask = Path(SampleKind.CROPPED.value) / f"{stem}_cropped_mask.png"
        cv2.imwrite(str(out / rel_img), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / rel_mask), crop_mask)
        return rel_img, rel_mask

    @staticmethod
    def _circle_mask(shape: tuple[int, int], dish: dict) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(
            mask, (dish["cx"], dish["cy"]), dish["r"], 255, thickness=-1
        )
        return mask

    def _write_dataset_json(self, out: Path, manifest: DatasetManifest) -> None:
        """Материализация `dataset.json` (storage="copy")."""
        import json

        payload = {
            "name": manifest.name,
            "origin": manifest.origin,
            "mask_mode": manifest.mask_mode,
            "storage": manifest.storage,
            "samples": [],
        }
        for sample in manifest.samples:
            entry = {
                "id": sample.id,
                "kind": sample.kind,
                "image": str(sample.image),
                "mask": str(sample.mask),
            }
            if sample.subset is not None:
                entry["subset"] = sample.subset
            if sample.dish is not None:
                entry["dish"] = {
                    "cx": sample.dish.cx,
                    "cy": sample.dish.cy,
                    "r": sample.dish.r,
                }
            payload["samples"].append(entry)

        (out / "dataset.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_manifest(data_root: Path, allow_multiclass: bool = False) -> DatasetManifest:
    """Фабрика манифеста по корню датасета.

    Приоритет адаптеров: готовый `dataset.json` (ManifestAdapter) → структура
    сессии разметки `source/` (LabelingAdapter) → легаси `images/` (PairsAdapter).
    При пустом результате выбрасывается `FileNotFoundError`.
    """
    root = Path(data_root)
    if (root / "dataset.json").is_file():
        manifest = ManifestAdapter().build(root)
    elif (root / "source").is_dir():
        manifest = LabelingAdapter().build(root)
    elif (root / "images").is_dir():
        manifest = PairsAdapter().build(root)
    else:
        raise FileNotFoundError(f"No image-mask pairs found in {data_root}")

    if not manifest.samples:
        raise FileNotFoundError(f"No image-mask pairs found in {data_root}")
    manifest.validate(root, allow_multiclass=allow_multiclass)
    return manifest