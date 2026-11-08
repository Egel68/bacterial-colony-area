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
from .dataset_split import DEFAULT_SPLIT_RATIOS, stratified_group_split

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
                    source_id=entry.get("source_id"),
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
            mask_path = (
                root / "cropped_masks" / f"{image_path.stem}_mask{image_path.suffix}"
            )
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
    if (
        petri.cx < 0
        or petri.cy < 0
        or petri.cx >= w
        or petri.cy >= h
        or petri.radius <= 0
    ):
        return None
    return {"cx": int(petri.cx), "cy": int(petri.cy), "r": int(petri.radius)}


class CocoBboxImporter(BaseAdapter):
    """Импорт внешнего датасета формата COCO-bbox в контракт данных проекта.

    Источником служит папка вида `datasets/22022540/` с `annot_COCO.json` и
    снимками чашек Петри. Импортёр:
    - растризует боксы колоний во вписанные эллипсы бинарной маской (255=колония);
    - опционально сохраняет геометрию чашки и создаёт чёрнофоновую пару;
    - опционально формирует `kind=cropped` — обрезку по кругу чашки;
    - материализует самодостаточную папку датасета (`storage="copy"`) с `dataset.json`,
      читаемую существующим `ManifestAdapter` без изменения ядра обучения.

    Источник только читается; выход пишется в `output_dir`.
    """

    def build(
        self,
        data_root: Path,
        output_dir: Optional[Path] = None,
        crop: bool = False,
        split: bool = False,
        seed: int = 42,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        black_background: bool = False,
    ) -> DatasetManifest:
        """Запуск импорта по корню источника в `output_dir`.

        Если `output_dir` не задан, используется соседняя папка
        `<root>_imported`. Возвращается манифест выходного датасета.
        """
        import json

        if black_background and not split:
            raise ValueError(
                "Black-background variants require a persisted source-level split; "
                "pass split=True"
            )

        source = Path(data_root)
        if not source.is_dir():
            raise NotADirectoryError(f"COCO source directory does not exist: {source}")
        coco_path = source / "annot_COCO.json"
        if not coco_path.is_file():
            raise FileNotFoundError(f"annot_COCO.json not found in {source}")

        out = Path(output_dir) if output_dir else Path(f"{source}_imported")
        if split:
            source_resolved = source.resolve()
            output_resolved = out.resolve()
            if (
                output_resolved == source_resolved
                or source_resolved in output_resolved.parents
                or output_resolved in source_resolved.parents
            ):
                raise ValueError(
                    "Split import output must be disjoint from the read-only source"
                )
        out.mkdir(parents=True, exist_ok=True)

        raw = json.loads(coco_path.read_text(encoding="utf-8"))

        images_by_id = {im["id"]: im for im in raw.get("images", [])}
        boxes_by_image: dict[int, list[tuple[int, list[float]]]] = {}
        for ann in raw.get("annotations", []):
            image_id = ann["image_id"]
            boxes_by_image.setdefault(image_id, []).append(
                (ann.get("category_id", 1), list(ann["bbox"]))
            )

        group_labels: dict[str, str] = {}
        if split:
            for image_id, image_meta in images_by_id.items():
                if not boxes_by_image.get(image_id):
                    continue
                category_ids = {
                    category_id for category_id, _ in boxes_by_image[image_id]
                }
                if len(category_ids) != 1:
                    raise ValueError(
                        f"Expected one category per source image {image_id}, "
                        f"found {sorted(category_ids)}"
                    )
                group_labels[str(image_id)] = str(next(iter(category_ids)))
            assignments = stratified_group_split(
                group_labels, seed=seed, ratios=split_ratios
            )
        else:
            assignments = {}

        samples: list[SampleRecord] = []
        source_records: list[SampleRecord] = []
        processed = 0
        for image_id, image_meta in images_by_id.items():
            boxes = boxes_by_image.get(image_id)
            if not boxes:
                continue

            filename = image_meta["file_name"]
            src_img = source / filename
            if not src_img.is_file():
                if split:
                    raise FileNotFoundError(
                        f"Source image referenced by COCO is missing: {src_img}"
                    )
                continue

            image = cv2.imread(str(src_img))
            if image is None:
                if split:
                    raise ValueError(f"Could not decode COCO source image: {src_img}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for _, bbox in boxes:
                mask |= _rasterize_ellipse(bbox, image.shape)

            # Геометрия нужна для cropped-данных или чёрнофонового варианта.
            # Source-only-импорт без этих опций по-прежнему пропускает авто-поиск.
            dish = _find_dish(image) if crop or black_background else None
            sample_id = Path(filename).stem
            if black_background:
                dish = self._require_dish_geometry(dish, image.shape, sample_id)

            rel_img, rel_mask = self._write_source(out, filename, image, mask)
            source_record = SampleRecord(
                id=sample_id,
                kind=SampleKind.SOURCE.value,
                image=rel_img,
                mask=rel_mask,
                subset=(assignments.get(str(image_id)) if split else None),
                dish=(
                    DishGeometry(cx=dish["cx"], cy=dish["cy"], r=dish["r"])
                    if dish
                    else None
                ),
            )
            samples.append(source_record)
            source_records.append(source_record)

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
                        subset=(assignments.get(str(image_id)) if split else None),
                        dish=DishGeometry(cx=dish["cx"], cy=dish["cy"], r=dish["r"]),
                    )
                )

            processed += 1

        if split and processed != len(assignments):
            raise RuntimeError(
                f"Split assigned {len(assignments)} annotated images but imported "
                f"{processed}; refusing to write an incomplete split dataset"
            )

        if split:
            self._write_split_json(
                out,
                seed=seed,
                ratios=split_ratios,
                assignments=assignments,
                images_by_id=images_by_id,
                boxes_by_image=boxes_by_image,
            )

        if black_background:
            for source_record in source_records:
                if source_record.dish is None:
                    raise ValueError(
                        f"Missing dish geometry for source sample '{source_record.id}'"
                    )
                source_image_path = out / source_record.image
                source_image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
                if source_image is None:
                    raise ValueError(
                        f"Could not read prepared source image for '{source_record.id}'"
                    )
                dish = {
                    "cx": source_record.dish.cx,
                    "cy": source_record.dish.cy,
                    "r": source_record.dish.r,
                }
                black_image = self._blacken_outside_dish(source_image, dish)
                augmented_image = self._write_black_background(
                    out, source_record.id, black_image
                )
                samples.append(
                    SampleRecord(
                        id=f"{source_record.id}__black_background",
                        kind=SampleKind.AUGMENTED.value,
                        image=augmented_image,
                        mask=source_record.mask,
                        subset=source_record.subset,
                        dish=source_record.dish,
                        source_id=source_record.id,
                    )
                )

        manifest = DatasetManifest(
            name=out.name,
            origin=Origin.EXTERNAL.value,
            mask_mode="binary",
            storage=StorageMode.COPY.value,
            samples=samples,
        )
        self._write_dataset_json(out, manifest)
        return manifest

    def _write_split_json(
        self,
        out: Path,
        *,
        seed: int,
        ratios: tuple[float, float, float],
        assignments: dict[str, str],
        images_by_id: dict,
        boxes_by_image: dict,
    ) -> None:
        """Записывает стабильное распределение исходников и параметры split."""
        import json

        payload = {
            "seed": seed,
            "ratios": dict(zip(("train", "val", "test"), ratios, strict=True)),
            "strategy": "stratified_source_image_id",
            "groups": {
                subset: [
                    {
                        "image_id": image_id,
                        "sample_id": Path(
                            images_by_id[int(image_id)]["file_name"]
                        ).stem,
                        "category_id": next(
                            iter(
                                {
                                    category_id
                                    for category_id, _ in boxes_by_image[int(image_id)]
                                }
                            )
                        ),
                    }
                    for image_id, assigned_subset in sorted(
                        assignments.items(), key=lambda item: int(item[0])
                    )
                    if assigned_subset == subset
                ]
                for subset in ("train", "val", "test")
            },
        }
        (out / "split.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _require_dish_geometry(
        dish: Optional[dict], shape: tuple[int, ...], sample_id: str
    ) -> dict[str, int]:
        if not isinstance(dish, dict):
            raise ValueError(
                f"Missing or invalid Petri-dish geometry for sample '{sample_id}'"
            )
        try:
            cx, cy, radius = (int(dish[key]) for key in ("cx", "cy", "r"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Missing or invalid Petri-dish geometry for sample '{sample_id}'"
            ) from error
        height, width = shape[:2]
        if cx < 0 or cy < 0 or cx >= width or cy >= height or radius <= 0:
            raise ValueError(
                f"Missing or invalid Petri-dish geometry for sample '{sample_id}'"
            )
        return {"cx": cx, "cy": cy, "r": radius}

    @staticmethod
    def _blacken_outside_dish(image: np.ndarray, dish: dict[str, int]) -> np.ndarray:
        height, width = image.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        inside = (xx - dish["cx"]) ** 2 + (yy - dish["cy"]) ** 2 <= dish["r"] ** 2
        black_image = image.copy()
        black_image[~inside] = 0
        return black_image

    @staticmethod
    def _write_black_background(out: Path, sample_id: str, image: np.ndarray) -> Path:
        directory = out / SampleKind.AUGMENTED.value
        directory.mkdir(parents=True, exist_ok=True)
        rel_image = (
            Path(SampleKind.AUGMENTED.value) / f"{sample_id}_black_background.png"
        )
        if not cv2.imwrite(str(out / rel_image), image):
            raise OSError(f"Could not write black-background variant for '{sample_id}'")
        return rel_image

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
        crop_mask = cv2.bitwise_and(
            mask, mask, mask=self._circle_mask(image.shape, dish)
        )

        rel_img = Path(SampleKind.CROPPED.value) / f"{stem}_cropped.jpg"
        rel_mask = Path(SampleKind.CROPPED.value) / f"{stem}_cropped_mask.png"
        cv2.imwrite(str(out / rel_img), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / rel_mask), crop_mask)
        return rel_img, rel_mask

    @staticmethod
    def _circle_mask(shape: tuple[int, int], dish: dict) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(mask, (dish["cx"], dish["cy"]), dish["r"], 255, thickness=-1)
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
            if sample.source_id is not None:
                entry["source_id"] = sample.source_id
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
