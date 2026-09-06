"""Схема манифеста датасета для обучения (контракт данных).

Контракт отделяет «что является данными» от «как файлы разложены на диске».
Манифест описывает список пар «изображение + маска» с метаданными. Обучение
потребляет записи манифеста и не знает о конкретных структурах папок.

Задел на будущее (см. AGENTS.md «Контракт данных и импортёры»):
- `mask_mode="multiclass"` — поддержка нескольких классов масок (точка расширения
  в `validate_manifest` и в потере/метриках обучения);
- внешние импортёры (COCO/VOC/произвольные пары) порождают манифест этой же
  схемы без изменения ядра обучения.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional


class MaskMode(StrEnum):
    """Режим представления маски."""

    BINARY = "binary"
    MULTICLASS = "multiclass"  # задел на будущее, обучение пока не поддерживает


class StorageMode(StrEnum):
    """Режим хранения файлов датасета."""

    REFERENCE = "reference"
    COPY = "copy"


class SampleKind(StrEnum):
    """Тип снимка в составе датасета."""

    SOURCE = "source"
    CROPPED = "cropped"
    AUGMENTED = "augmented"


class Origin(StrEnum):
    """Происхождение датасета."""

    LABELING = "labeling"
    LEGACY = "legacy"
    EXTERNAL = "external"  # задел на будущее (внешние импортёры)


@dataclass
class DishGeometry:
    """Опциональная геометрия чашки Петри (задел для препроцессинга)."""

    cx: int
    cy: int
    r: int


@dataclass
class SampleRecord:
    """Запись о паре «изображение + маска»."""

    id: str
    kind: str
    image: Path  # относительный путь от корня датасета
    mask: Path  # относительный путь от корня датасета
    subset: Optional[str] = None  # train | val | test | None
    dish: Optional[DishGeometry] = None


@dataclass
class DatasetManifest:
    """Каноническое описание обучающего датасета."""

    name: str
    origin: str
    mask_mode: str
    storage: str
    samples: list[SampleRecord] = field(default_factory=list)

    def validate(self, data_root: Path, allow_multiclass: bool = False) -> None:
        """Валидация манифеста относительно корня датасета.

        Проверяет известные значения полей, режим маски (сейчас поддерживается
        только `binary`; `multiclass` — задел) и существование файлов пар.
        """
        validate_mask_mode(self.mask_mode, allow_multiclass=allow_multiclass)
        if self.origin not in [o.value for o in Origin]:
            raise ValueError(f"Unknown origin '{self.origin}'")
        if self.storage not in [s.value for s in StorageMode]:
            raise ValueError(f"Unknown storage '{self.storage}'")

        data_root = Path(data_root).resolve()
        for sample in self.samples:
            if sample.kind not in [k.value for k in SampleKind]:
                raise ValueError(f"Unknown kind '{sample.kind}'")
            if sample.subset is not None and sample.subset not in (
                "train",
                "val",
                "test",
            ):
                raise ValueError(f"Unknown subset '{sample.subset}'")
            image_path = (data_root / sample.image).resolve()
            mask_path = (data_root / sample.mask).resolve()
            if not image_path.is_file():
                raise ValueError(f"Missing image file for sample '{sample.id}': {image_path}")
            if not mask_path.is_file():
                raise ValueError(f"Missing mask file for sample '{sample.id}': {mask_path}")


def validate_mask_mode(mask_mode: str, allow_multiclass: bool = False) -> None:
    """Проверка режима маски.

    Сейчас конвейер обучения поддерживает только `binary`. `multiclass` —
    задел на будущее; при попытке использовать его текущим ядром выбрасывается
    понятная ошибка.
    """
    if mask_mode not in [m.value for m in MaskMode]:
        raise ValueError(f"Unknown mask_mode '{mask_mode}'")
    if mask_mode == MaskMode.MULTICLASS and not allow_multiclass:
        raise ValueError(
            f"mask_mode='{mask_mode}' не поддерживается текущим конвейером обучения; "
            "поддержка multiclass запланирована (задел на будущее)"
        )