"""Тесты компактного MobileNetV3-Small segmenter."""

import pytest
import train.models as model_registry

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_mobilenet_segmenter_is_registered_and_returns_512_logits():
    model = model_registry.get_model("mobilenet_v3_small_unet", pretrained=False)
    model.eval()

    with torch.inference_mode():
        output = model(torch.randn(1, 3, 512, 512))

    assert "mobilenet_v3_small_unet" in model_registry.list_models()
    assert output.shape == (1, 1, 512, 512)
    assert torch.isfinite(output).all()


def test_mobilenet_segmenter_produces_finite_training_loss():
    model = model_registry.get_model("mobilenet_v3_small_unet", pretrained=False)
    model.train()
    image = torch.randn(2, 3, 128, 128)
    mask = torch.randint(0, 2, (2, 1, 128, 128)).float()

    loss = model.get_loss(model(image), mask)

    assert torch.isfinite(loss)


def test_mobilenet_segmenter_exports_to_cpu_onnx(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = model_registry.get_model("mobilenet_v3_small_unet", pretrained=False)
    output_path = tmp_path / "mobilenet.onnx"

    model.to_onnx(str(output_path), input_shape=(1, 3, 64, 64))
    assert output_path.is_file()
    assert not output_path.with_suffix(output_path.suffix + ".data").exists()

    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    prediction = session.run(None, {"input": torch.randn(1, 3, 64, 64).numpy()})[0]
    assert prediction.shape == (1, 1, 64, 64)


def test_mobilenet_segmenter_can_load_imagenet_pretrained_weights(monkeypatch):
    from train.models import mobilenet_v3_small_unet as model_module

    called = []
    original_builder = model_module.mobilenet_v3_small

    def record_pretrained(*, weights):
        called.append(weights)
        return original_builder(weights=None)

    monkeypatch.setattr(model_module, "mobilenet_v3_small", record_pretrained)
    model = model_registry.get_model("mobilenet_v3_small_unet", pretrained=True)

    assert model is not None
    assert called == [model_module.MobileNet_V3_Small_Weights.DEFAULT]


def test_mobilenet_segmenter_onnx_outputs_match_cpu_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    import onnxruntime

    torch.manual_seed(123)
    model = model_registry.get_model("mobilenet_v3_small_unet", pretrained=False).eval()
    input_tensor = torch.randn(1, 3, 64, 64)
    output_path = tmp_path / "parity.onnx"
    model.to_onnx(str(output_path), input_shape=tuple(input_tensor.shape))

    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    with torch.inference_mode():
        expected = model(input_tensor).cpu().numpy()
    actual = session.run(None, {"input": input_tensor.numpy()})[0]

    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-4)
