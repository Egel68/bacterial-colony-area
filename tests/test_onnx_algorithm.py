import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import testing.onnx_algorithm as onnx_algorithm
from testing.onnx_algorithm import OnnxModelAlgorithm


class _FakeSession:
    """Имитирует onnxruntime.InferenceSession для тестов."""

    def __init__(self, model_path, output_value=0.9):
        self.model_path = model_path
        self.output_value = output_value

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, outputs, inputs):
        # Возвращаем постоянную вероятность на каждом пикселе.
        batch, _, h, w = inputs["input"].shape
        return [np.ones((batch, 1, h, w), dtype=np.float32) * self.output_value]


def _fake_module(output_value=0.9, providers_seen=None, sess_options_seen=None):
    """Возвращает фейковый модуль onnxruntime."""

    def create_session(path, providers=None, sess_options=None):
        if providers_seen is not None:
            providers_seen.append(providers)
        if sess_options_seen is not None:
            sess_options_seen.append(sess_options)
        return _FakeSession(path, output_value)

    return SimpleNamespace(
        InferenceSession=create_session,
        SessionOptions=lambda: SimpleNamespace(
            intra_op_num_threads=0, inter_op_num_threads=0
        ),
    )


@pytest.fixture
def bgr_image():
    return np.ones((64, 48, 3), dtype=np.uint8) * 100


class TestDetect:
    def test_returns_mask_same_shape(self, bgr_image, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _fake_module(0.9))
        algo = OnnxModelAlgorithm(model_path="model.onnx", img_size=64)
        mask = algo.detect(bgr_image)
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.dtype == np.uint8
        assert mask.shape == bgr_image.shape[:2]

    def test_threshold_above_makes_full_mask(self, bgr_image, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _fake_module(0.9))
        algo = OnnxModelAlgorithm(model_path="model.onnx", threshold=0.5)
        mask = algo.detect(bgr_image)
        assert set(np.unique(mask)) <= {0, 255}
        assert mask.max() == 255

    def test_threshold_below_makes_empty_mask(self, bgr_image, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _fake_module(0.1))
        algo = OnnxModelAlgorithm(model_path="model.onnx", threshold=0.5)
        mask = algo.detect(bgr_image)
        assert set(np.unique(mask)) == {0}

    def test_default_name_from_path(self):
        algo = OnnxModelAlgorithm(model_path="/some/dir/best.onnx")
        assert algo.name == "best"
        assert "best.onnx" in algo.description

    def test_custom_name_and_description(self):
        algo = OnnxModelAlgorithm(model_path="m.onnx", name="MyNet", description="desc")
        assert algo.name == "MyNet"
        assert algo.description == "desc"


class TestMissingOnnxRuntime:
    def test_raises_runtime_error_without_onnxruntime(self, bgr_image, monkeypatch):
        monkeypatch.delitem(__import__("sys").modules, "onnxruntime", raising=False)
        # Гарантируем отсутствие реального модуля в sys.path.
        import sys

        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            algo = OnnxModelAlgorithm(model_path="model.onnx")
            with pytest.raises(RuntimeError, match="onnxruntime"):
                algo.detect(bgr_image)


def test_session_explicitly_uses_cpu_provider(bgr_image, monkeypatch):
    providers = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        _fake_module(0.9, providers_seen=providers),
    )

    OnnxModelAlgorithm(model_path="model.onnx").detect(bgr_image)

    assert providers == [["CPUExecutionProvider"]]


def test_default_models_dir_uses_nuitka_onefile_payload(tmp_path, monkeypatch):
    extraction_dir = tmp_path / "onefile-payload"
    (extraction_dir / "models").mkdir(parents=True)
    binary_dir = tmp_path / "installed-app"
    binary_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(extraction_dir / "python"))
    monkeypatch.setattr(
        onnx_algorithm,
        "__compiled__",
        SimpleNamespace(containing_dir=str(binary_dir), onefile=True),
        raising=False,
    )

    assert onnx_algorithm._default_models_dir() == extraction_dir / "models"


def test_default_models_dir_uses_nuitka_standalone_dir(tmp_path, monkeypatch):
    binary_dir = tmp_path / "standalone"
    (binary_dir / "models").mkdir(parents=True)
    monkeypatch.setattr(
        onnx_algorithm,
        "__compiled__",
        SimpleNamespace(containing_dir=str(binary_dir), onefile=False),
        raising=False,
    )

    assert onnx_algorithm._default_models_dir() == binary_dir / "models"
