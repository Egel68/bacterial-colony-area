import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nuitka_build


def test_jobs_derived_from_core_count():
    with (
        mock.patch("nuitka_build.os.cpu_count", return_value=6),
        mock.patch.dict("os.environ", {}, clear=True),
    ):
        assert nuitka_build.compute_jobs() == 6


def test_jobs_capped_at_max():
    with (
        mock.patch("nuitka_build.os.cpu_count", return_value=64),
        mock.patch.dict("os.environ", {}, clear=True),
    ):
        assert nuitka_build.compute_jobs() == 8


def test_jobs_floor_at_one():
    with (
        mock.patch("nuitka_build.os.cpu_count", return_value=1),
        mock.patch.dict("os.environ", {}, clear=True),
    ):
        assert nuitka_build.compute_jobs() == 1


def test_nuitka_jobs_overrides_default():
    with (
        mock.patch("nuitka_build.os.cpu_count", return_value=6),
        mock.patch.dict("os.environ", {"NUITKA_JOBS": "3"}, clear=True),
    ):
        assert nuitka_build.compute_jobs() == 3


def test_nuitka_jobs_invalid_value():
    with mock.patch.dict("os.environ", {"NUITKA_JOBS": "abc"}, clear=True):
        with pytest.raises(SystemExit):
            nuitka_build.compute_jobs()
