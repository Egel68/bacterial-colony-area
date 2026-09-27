"""Проверки full-image held-out evaluator и запуска report."""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from train.dataset_manifest import SampleRecord
from train.evaluation import (
    build_training_report,
    evaluate_model_on_records,
    evaluate_patch_records,
    verify_onnx_cpu_parity,
)
from train.models.base import BaseSegmenter


class _ColorSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        # Normalized red channel is positive in the deliberately red target ROI.
        return image[:, 0:1] * self.scale * 8


def test_full_image_evaluator_reports_metrics_and_source_dimensions(tmp_path):
    image = np.zeros((600, 700, 3), dtype=np.uint8)
    image[220:380, 260:430, 2] = 255
    mask = np.zeros((600, 700), dtype=np.uint8)
    mask[220:380, 260:430] = 255
    cv2.imwrite(str(tmp_path / "dish.png"), image)
    cv2.imwrite(str(tmp_path / "dish_mask.png"), mask)
    record = SampleRecord("dish", "source", "dish.png", "dish_mask.png", "test")

    result = evaluate_model_on_records(
        _ColorSegmenter(), [record], tmp_path, tile_size=128, stride=96
    )

    assert result["n_images"] == 1
    metrics = result["per_image"]["dish"]
    assert metrics["width"] == 700
    assert metrics["height"] == 600
    assert metrics["iou"] > 0.85


def test_patch_evaluator_reports_fixed_per_image_metrics(tmp_path):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[70:170, 80:180, 2] = 255
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[70:170, 80:180] = 255
    cv2.imwrite(str(tmp_path / "dish.png"), image)
    cv2.imwrite(str(tmp_path / "dish_mask.png"), mask)
    record = SampleRecord("dish", "source", "dish.png", "dish_mask.png", "train")

    result = evaluate_patch_records(
        _ColorSegmenter(),
        [record],
        tmp_path,
        patch_size=128,
        patches_per_image=2,
        batch_size=2,
        seed=3,
    )

    assert result["n_images"] == 1
    assert result["per_image"]["dish"]["subset"] == "train"
    assert result["per_image"]["dish"]["width"] == 256
    assert result["per_image"]["dish"]["height"] == 256
    assert result["per_image"]["dish"]["iou"] > 0.85


def test_onnx_cpu_parity_uses_validation_patches(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48, 2] = 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    cv2.imwrite(str(tmp_path / "dish.png"), image)
    cv2.imwrite(str(tmp_path / "dish_mask.png"), mask)
    record = SampleRecord("dish", "source", "dish.png", "dish_mask.png", "val")
    model = _ColorSegmenter().eval()
    onnx_path = tmp_path / "model.onnx"
    model.to_onnx(str(onnx_path), input_shape=(1, 3, 64, 64))
    assert not onnx_path.with_suffix(onnx_path.suffix + ".data").exists()

    result = verify_onnx_cpu_parity(
        model,
        onnx_path,
        [record],
        tmp_path,
        patch_size=64,
        patches_per_image=1,
        max_patches=1,
    )

    assert result["providers"] == ["CPUExecutionProvider"]
    assert result["patches"] == 1
    assert result["passed"]
    assert result["max_abs_diff"] <= result["logit_tolerance"]
    assert result["max_abs_probability_diff"] <= result["probability_tolerance"]
    assert result["comparison"] == "logits_and_sigmoid_probabilities"
    assert result["tolerance"] == result["probability_tolerance"]


def test_onnx_cpu_parity_allows_small_logit_drift_when_probabilities_match(
    tmp_path, monkeypatch
):
    pytest.importorskip("onnxruntime")
    import onnxruntime

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48, 2] = 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    record = SampleRecord("dish", "source", "dish.png", "dish_mask.png", "val")
    cv2.imwrite(str(tmp_path / "dish.png"), image)
    cv2.imwrite(str(tmp_path / "dish_mask.png"), mask)

    class _OffsetCpuSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [type("_Input", (), {"name": "input"})()]

        def run(self, outputs, inputs):
            del outputs
            return [inputs["input"][:, 0:1] * 8.0 + np.float32(1.5e-4)]

    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda *_args, **_kwargs: _OffsetCpuSession()
    )

    result = verify_onnx_cpu_parity(
        _ColorSegmenter().eval(),
        tmp_path / "model.onnx",
        [record],
        tmp_path,
        patch_size=64,
        patches_per_image=1,
        max_patches=1,
        tolerance=1e-4,
    )

    assert 1e-4 < result["max_abs_diff"] <= result["logit_tolerance"]
    assert result["max_abs_probability_diff"] < result["probability_tolerance"]
    assert result["passed"]
    assert result["tolerance"] == 1e-4


def test_training_report_contains_validation_test_split_and_artifact_data(tmp_path):
    (tmp_path / "split.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                "strategy": "stratified_source_image_id",
                "groups": {
                    "train": [{"image_id": 1, "sample_id": "a", "category_id": 1}],
                    "val": [{"image_id": 2, "sample_id": "b", "category_id": 1}],
                    "test": [{"image_id": 3, "sample_id": "c", "category_id": 1}],
                },
            }
        )
    )
    for sample_id, subset in (("train", "train"), ("val", "val"), ("test", "test")):
        cv2.imwrite(str(tmp_path / f"{sample_id}.png"), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(tmp_path / f"{sample_id}_mask.png"), np.zeros((8, 8), np.uint8))
    dataset_json = {
        "name": "tiny",
        "origin": "external",
        "mask_mode": "binary",
        "storage": "copy",
        "samples": [
            {
                "id": sample_id,
                "kind": "source",
                "image": f"{sample_id}.png",
                "mask": f"{sample_id}_mask.png",
                "subset": subset,
            }
            for sample_id, subset in (
                ("train", "train"),
                ("val", "val"),
                ("test", "test"),
            )
        ],
    }
    (tmp_path / "dataset.json").write_text(json.dumps(dataset_json))

    class _Config:
        model_name = "mobilenet_v3_small_unet"
        seed = 42
        img_size = 512
        epochs = 4
        batch_size = 2
        patches_per_image = 3
        pretrained = True
        lr = 0.001
        weight_decay = 1e-5
        patience = 5

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"tiny onnx")
    report = build_training_report(
        data_root=tmp_path,
        cfg=_Config(),
        summary={"best_epoch": 3},
        train_history=[{"epoch": 2, "iou": 0.7}, {"epoch": 3, "iou": 0.8}],
        val_history=[{"epoch": 2, "iou": 0.4}, {"epoch": 3, "iou": 0.5}],
        test_result={"mean": {"iou": 0.4}, "per_image": {}, "n_images": 1},
        onnx_path=onnx_path,
    )

    assert report["split"]["seed"] == 42
    assert report["best_epoch"] == 3
    assert report["split_metrics"]["train"]["mean"]["iou"] == 0.8
    assert report["split_metrics"]["val"]["mean"]["iou"] == 0.5
    assert report["test"]["mean"]["iou"] == 0.4
    assert report["onnx"]["bytes"] == len(b"tiny onnx")
    assert "bounding boxes" in report["weak_label_note"]
    assert report["git"]["revision"]
    assert isinstance(report["git"]["dirty"], bool)
