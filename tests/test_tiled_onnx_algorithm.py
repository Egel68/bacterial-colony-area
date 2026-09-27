"""Проверки full-resolution CPU tiled ONNX inference."""

from types import SimpleNamespace

import numpy as np

from testing.tiled_onnx_algorithm import TiledOnnxModelAlgorithm, tile_positions
from testing.onnx_algorithm import OnnxModelAlgorithm, scan_bundled_models


class _Input:
    name = "input"


class _SequenceSession:
    def __init__(self, logits):
        self.logits = list(logits)
        self.calls = 0

    def get_inputs(self):
        return [_Input()]

    def run(self, outputs, feeds):
        batch, channels, height, width = feeds["input"].shape
        logit = self.logits[min(self.calls, len(self.logits) - 1)]
        self.calls += 1
        return [np.full((batch, channels // 3, height, width), logit, np.float32)]


def _fake_ort(session, created):
    def create(path, providers, **kwargs):
        created.append(providers)
        return session

    return SimpleNamespace(
        InferenceSession=create,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=lambda: object(),
    )


def test_tiled_inference_uses_cpu_and_covers_non_divisible_edges(monkeypatch):
    import sys

    session = _SequenceSession([10.0])
    created = []
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(session, created))
    algorithm = TiledOnnxModelAlgorithm("model.onnx")
    image = np.zeros((700, 900, 3), dtype=np.uint8)
    progress = []

    result = algorithm.detect_with_progress(
        image,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}
    assert result.min() == 255
    assert created == [["CPUExecutionProvider"]]
    assert progress[-1] == (6, 6)
    assert len(progress) == 6


def test_tile_grid_uses_fixed_stride_and_pads_partial_right_edge():
    assert tile_positions(900) == [0, 384, 768]
    assert tile_positions(513) == [0, 384]


def test_tiled_inference_reports_every_tile_from_fixed_stride_grid(monkeypatch):
    import sys

    session = _SequenceSession([0.0])
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(session, []))
    algorithm = TiledOnnxModelAlgorithm("model.onnx")
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    progress = []

    mask = algorithm.detect_with_progress(
        image,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert mask.shape == image.shape[:2]
    assert progress == [(index, 9) for index in range(1, 10)]


def test_overlap_probabilities_are_averaged_before_threshold(monkeypatch):
    import sys

    # Sigmoid(1.386)=0.8 and sigmoid(-0.405)=0.4; their overlap mean is > 0.5.
    session = _SequenceSession([1.3862944, -0.4054651])
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(session, []))
    algorithm = TiledOnnxModelAlgorithm("model.onnx")
    image = np.zeros((64, 600, 3), dtype=np.uint8)

    result = algorithm.detect(image)

    assert result.shape == image.shape[:2]
    assert np.all(result[:, 384:512] == 255)
    assert np.all(result[:, 512:] == 0)


def test_small_image_gets_padded_and_reports_one_tile(monkeypatch):
    import sys

    session = _SequenceSession([10.0])
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(session, []))
    algorithm = TiledOnnxModelAlgorithm("model.onnx")
    image = np.zeros((97, 83, 3), dtype=np.uint8)
    progress = []

    result = algorithm.detect_with_progress(
        image,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert result.shape == image.shape[:2]
    assert result.min() == 255
    assert progress == [(1, 1)]


def test_missing_onnxruntime_has_clear_cpu_error(monkeypatch):
    import sys
    import pytest

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    algorithm = TiledOnnxModelAlgorithm("model.onnx")

    with pytest.raises(RuntimeError, match="onnxruntime.*CPU"):
        algorithm.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_bundled_compact_model_uses_tiled_adapter_and_legacy_model_is_unchanged(
    tmp_path,
):
    (tmp_path / "colony_mobilenet_v3_small.onnx").write_bytes(b"model")
    (tmp_path / "colony_seg.onnx").write_bytes(b"legacy")

    algorithms = {
        algorithm.name: algorithm for algorithm in scan_bundled_models(str(tmp_path))
    }

    assert isinstance(
        algorithms["NN:colony_mobilenet_v3_small"], TiledOnnxModelAlgorithm
    )
    assert isinstance(algorithms["NN:colony_seg"], OnnxModelAlgorithm)
