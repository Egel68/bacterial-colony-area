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