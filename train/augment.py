"""Генерация аугментированных пар изображение-маска для обучения U-Net."""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

AUGMENTATIONS_PER_IMAGE = 15
SEED = 42

CROPPED_DIR = Path("test_images/cropped")
CROPPED_MASKS_DIR = Path("test_images/cropped_masks")
OUT_IMAGES_DIR = Path("train/data/images")
OUT_MASKS_DIR = Path("train/data/masks")


def _build_transforms() -> tuple[A.Compose, A.Compose]:
    geometric = A.Compose(
        [
            A.Rotate(limit=180, p=0.9, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                p=0.7,
            ),
        ],
        additional_targets={"mask": "image"},
    )
    pixel = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.8
            ),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=0, p=0.3
            ),
        ]
    )
    return geometric, pixel


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_image_mask(
    img_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mask_path = CROPPED_MASKS_DIR / f"{img_path.stem}_mask.png"
    if not mask_path.exists():
        print(f"Пропуск {img_path.name}: нет маски {mask_path.name}")
        return None, None
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка загрузки {img_path.name}")
        return None, None
    if mask is None:
        print(f"Ошибка загрузки маски {mask_path.name}")
        return None, None
    return image, mask


def _finalize(aug_img: np.ndarray, aug_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if aug_img.dtype != np.uint8:
        aug_img = np.clip(np.round(aug_img), 0, 255).astype(np.uint8)
    if aug_mask.dtype != np.uint8:
        aug_mask = np.clip(np.round(aug_mask), 0, 255).astype(np.uint8)
    aug_mask = (aug_mask > 127).astype(np.uint8) * 255
    return aug_img, aug_mask


def main() -> None:
    random.seed(SEED)

    if not CROPPED_DIR.exists():
        print(f"Директория {CROPPED_DIR} не найдена")
        return

    image_paths = sorted(CROPPED_DIR.glob("*.png"))
    if not image_paths:
        print(f"Нет PNG-изображений в {CROPPED_DIR}")
        return

    _ensure_dir(OUT_IMAGES_DIR)
    _ensure_dir(OUT_MASKS_DIR)

    geometric, pixel = _build_transforms()
    total = len(image_paths) * AUGMENTATIONS_PER_IMAGE
    print(
        f"Аугментация: {len(image_paths)} исходников × {AUGMENTATIONS_PER_IMAGE} "
        f"= {total} пар → {OUT_IMAGES_DIR.parent}"
    )

    index = 0
    for img_path in tqdm(image_paths, desc="Аугментация"):
        image, mask = _load_image_mask(img_path)
        if image is None or mask is None:
            continue

        for _ in range(AUGMENTATIONS_PER_IMAGE):
            random.seed(SEED + index)

            geo = geometric(image=image, mask=mask)
            aug_img = pixel(image=geo["image"])["image"]
            aug_mask = geo["mask"]

            aug_img, aug_mask = _finalize(aug_img, aug_mask)

            stem = f"{img_path.stem}_aug_{index:03d}"
            cv2.imwrite(str(OUT_IMAGES_DIR / f"{stem}.png"), aug_img)
            cv2.imwrite(str(OUT_MASKS_DIR / f"{stem}.png"), aug_mask)
            index += 1

    print(f"Готово: {total} пар сохранено")


if __name__ == "__main__":
    main()
