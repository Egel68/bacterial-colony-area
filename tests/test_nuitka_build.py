import sys
import os
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nuitka_build
import nuitka_flags


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


def test_nuitka_build_includes_cpu_onnxruntime_and_models(tmp_path):
    for package in ("analysis", "testing", "ui"):
        package_dir = tmp_path / package
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("")
    models = tmp_path / "models"
    models.mkdir()
    (models / "new_model.onnx").write_bytes(b"model")
    ort_capi = tmp_path / ".venv-build/lib/python3.13/site-packages/onnxruntime/capi"
    ort_capi.mkdir(parents=True)
    (
        ort_capi / "onnxruntime_pybind11_state.cpython-313-x86_64-linux-gnu.so"
    ).write_bytes(b"extension")
    (ort_capi / "libonnxruntime.so.1").write_bytes(b"runtime")
    (ort_capi / "libonnxruntime_providers_shared.so").write_bytes(b"provider")

    with mock.patch.dict("os.environ", {"UV_PROJECT_ENVIRONMENT": ".venv-build"}):
        flags = nuitka_flags.build_flags(tmp_path)

    assert "--include-package=onnxruntime" in flags
    assert "--include-package-data=onnxruntime" in flags
    assert "--nofollow-import-to=datasets" in flags
    assert any("libonnxruntime.so.1" in flag for flag in flags)
    assert not any("libonnxruntime_providers_shared.so" in flag for flag in flags)
    assert not any("onnxruntime_pybind11_state" in flag for flag in flags)
    assert any(
        flag == f"--include-data-files={models.resolve()}/*.onnx=models/"
        for flag in flags
    )


def test_nuitka_build_collects_ort_dlls_from_windows_environment(tmp_path):
    ort_capi = tmp_path / ".venv-build/Lib/site-packages/onnxruntime/capi"
    ort_capi.mkdir(parents=True)
    (ort_capi / "onnxruntime.dll").write_bytes(b"runtime")
    (ort_capi / "onnxruntime_providers_shared.dll").write_bytes(b"providers")

    with mock.patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": ".venv-build"}):
        flags = nuitka_flags.build_flags(tmp_path)

    assert any("onnxruntime.dll=onnxruntime/capi/onnxruntime.dll" in f for f in flags)
    assert not any("onnxruntime_providers_shared.dll" in flag for flag in flags)
