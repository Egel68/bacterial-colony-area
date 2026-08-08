from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.geometry import PetriInfo

TEST_IMAGES = Path("test_images")


@pytest.fixture
def blank_image_bgr():
    return np.ones((200, 200, 3), dtype=np.uint8) * 128


@pytest.fixture
def blank_image_gray():
    return np.ones((200, 200), dtype=np.uint8) * 128


@pytest.fixture
def binary_mask_circle():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    return mask


@pytest.fixture
def binary_mask_two_colonies():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (70, 100), 30, 255, -1)
    cv2.circle(mask, (140, 100), 20, 255, -1)
    return mask


@pytest.fixture
def petri_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 80, 255, -1)
    return mask


@pytest.fixture
def petri_info():
    return PetriInfo(cx=100, cy=100, radius=80, image_shape=(200, 200))


@pytest.fixture
def synthetic_colony_image():
    image = np.ones((200, 200, 3), dtype=np.uint8) * 50
    cv2.circle(image, (100, 100), 15, (150, 150, 150), -1)
    return image


@pytest.fixture
def test_source_paths():
    source_dir = TEST_IMAGES / "source"
    if not source_dir.exists():
        return []
    return sorted(source_dir.glob("*.png"))
