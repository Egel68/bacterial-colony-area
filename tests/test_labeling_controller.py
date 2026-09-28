import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.geometry import PetriInfo
from ui.controllers.labeling_controller import LabelingController


@pytest.fixture
def controller():
    return LabelingController()


@pytest.fixture
def session_dir(tmp_path):
    return tmp_path / "session"


def _make_image(path: Path, size=(120, 160, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ones(size, dtype=np.uint8) * 128
    cv2.imwrite(str(path), image)


def _make_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


class TestListImageFiles:
    def test_filters_unsupported(self, controller, session_dir):
        session_dir.mkdir()
        for name in ["a.png", "b.jpg", "c.webp"]:
            _make_image(session_dir / name)
        for name in ["d.gif", "e.svg"]:
            _make_placeholder(session_dir / name)
        files = controller.list_image_files(session_dir)
        names = {f.name for f in files}
        assert names == {"a.png", "b.jpg", "c.webp"}
        assert "d.gif" not in names
        assert "e.svg" not in names

    def test_only_unsupported_returns_empty(self, controller, session_dir):
        session_dir.mkdir()
        _make_placeholder(session_dir / "only.gif")
        assert controller.list_image_files(session_dir) == []

    def test_missing_dir_returns_empty(self, controller, session_dir):
        assert controller.list_image_files(session_dir / "nope") == []

    def test_case_insensitive(self, controller, session_dir):
        session_dir.mkdir()
        _make_image(session_dir / "UPPER.PNG")
        files = controller.list_image_files(session_dir)
        assert [f.name for f in files] == ["UPPER.PNG"]


class TestCropByPetri:
    def test_crop_shape_and_black_background(
        self, controller, blank_image_bgr, petri_info
    ):
        cropped = controller.crop_by_petri(blank_image_bgr, petri_info)
        assert cropped.shape == (2 * petri_info.radius, 2 * petri_info.radius, 3)
        corners = [
            (0, 0),
            (0, -1),
            (-1, 0),
            (-1, -1),
        ]
        for y, x in corners:
            assert np.all(cropped[y, x] == 0)

    def test_crop_near_edge_clamps(self, controller):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        info = PetriInfo(cx=10, cy=10, radius=80, image_shape=(100, 100))
        cropped = controller.crop_by_petri(image, info)
        assert cropped.shape[2] == 3
        assert cropped.shape[0] <= 2 * info.radius + 1
        assert cropped.shape[1] <= 2 * info.radius + 1

    def test_crop_center_preserved(
        self, controller, synthetic_colony_image, petri_info
    ):
        cropped = controller.crop_by_petri(synthetic_colony_image, petri_info)
        center = cropped[petri_info.radius, petri_info.radius]
        assert np.all(center == 150)


class TestGetCurrentDir:
    def test_source_dir_preferred(self, controller, session_dir):
        (session_dir / "source").mkdir(parents=True)
        _make_image(session_dir / "source" / "img.png")
        assert controller.get_current_dir(session_dir, "source") == (
            session_dir / "source"
        )

    def test_source_dir_created_if_missing(self, controller, session_dir):
        result = controller.get_current_dir(session_dir, "source")
        assert result == session_dir / "source"
        assert result.is_dir()

    def test_cropped_dir_preferred(self, controller, session_dir):
        (session_dir / "cropped").mkdir(parents=True)
        _make_image(session_dir / "cropped" / "img.png")
        assert controller.get_current_dir(session_dir, "cropped") == (
            session_dir / "cropped"
        )

    def test_cropped_dir_created_if_missing(self, controller, session_dir):
        result = controller.get_current_dir(session_dir, "cropped")
        assert result == session_dir / "cropped"
        assert result.is_dir()


class TestGetMaskDir:
    def test_source_mask_dir(self, controller, session_dir):
        assert controller.get_mask_dir(session_dir, "source") == session_dir / "masks"

    def test_cropped_mask_dir(self, controller, session_dir):
        assert controller.get_mask_dir(session_dir, "cropped") == (
            session_dir / "cropped_masks"
        )


class TestExportSessionToZip:
    def test_zip_contains_all_files(self, controller, session_dir):
        _make_image(session_dir / "img.png")
        _make_image(session_dir / "masks" / "img_mask.png")
        _make_image(session_dir / "cropped" / "img_cropped.png")
        output = session_dir.parent / "session.zip"
        controller.export_session_to_zip(session_dir, output)
        with zipfile.ZipFile(output) as zf:
            names = set(zf.namelist())
        assert names == {
            "session/img.png",
            "session/masks/img_mask.png",
            "session/cropped/img_cropped.png",
        }
