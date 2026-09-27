"""Интеграционные проверки patch-training и изоляции test subset."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
from torch import nn

from train.config import TrainingConfig
from train.dataset_manifest import SampleRecord
from train.models.base import BaseSegmenter
from train.train import run_training


class _TinySegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()
        self.out = nn.Conv2d(3, 1, kernel_size=1)
        self.exported = []

    def forward(self, image):
        return self.out(image)

    def to_onnx(self, path, input_shape=(1, 3, 512, 512)):
        self.exported.append(path)
        Path(path).write_bytes(b"tiny fake onnx")


def _write_record(root, sample_id, subset, color):
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[:] = color
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 10, 255, -1)
    cv2.imwrite(str(root / f"{sample_id}.png"), image)
    cv2.imwrite(str(root / f"{sample_id}_mask.png"), mask)
    return SampleRecord(
        sample_id,
        "source",
        f"{sample_id}.png",
        f"{sample_id}_mask.png",
        subset,
    )


class _FakeWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def test_patch_training_respects_test_and_records_epoch_times(tmp_path, monkeypatch):
    import train.train as training

    records = [
        _write_record(tmp_path, "train-a", "train", (20, 30, 40)),
        _write_record(tmp_path, "train-b", "train", (40, 30, 20)),
        _write_record(tmp_path, "val-a", "val", (10, 80, 30)),
        _write_record(tmp_path, "test-secret", "test", (0, 0, 255)),
    ]
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "origin": "external",
                "mask_mode": "binary",
                "storage": "copy",
                "samples": [
                    {
                        "id": record.id,
                        "kind": record.kind,
                        "image": str(record.image),
                        "mask": str(record.mask),
                        "subset": record.subset,
                    }
                    for record in records
                ],
            }
        ),
        encoding="utf-8",
    )
    models = []

    def make_model(_name):
        model = _TinySegmenter()
        models.append(model)
        return model

    monkeypatch.setattr(training, "get_model", make_model)
    monkeypatch.setattr(training, "SummaryWriter", _FakeWriter)
    seen_subsets = {}
    original_train_epoch = training.train_epoch
    original_val_epoch = training.val_epoch

    def checked_train_epoch(model, loader, optimizer, device, scaler=None):
        seen_subsets["train"] = [record.id for record in loader.dataset.records]
        assert "test-secret" not in seen_subsets["train"]
        return original_train_epoch(model, loader, optimizer, device, scaler)

    def checked_val_epoch(model, loader, device):
        seen_subsets["val"] = [record.id for record in loader.dataset.records]
        assert "test-secret" not in seen_subsets["val"]
        return original_val_epoch(model, loader, device)

    monkeypatch.setattr(training, "train_epoch", checked_train_epoch)
    monkeypatch.setattr(training, "val_epoch", checked_val_epoch)

    cfg = TrainingConfig(
        data_root=tmp_path,
        model_name="tiny",
        patch_training=True,
        img_size=32,
        patches_per_image=1,
        batch_size=2,
        epochs=2,
        patience=3,
        num_workers=0,
        run_dir=tmp_path / "runs",
        device="cpu",
    )

    class _SimulatedInterruption(Exception):
        pass

    def stop_after_first_epoch(epoch, *_args):
        if epoch == 1:
            raise _SimulatedInterruption

    with pytest.raises(_SimulatedInterruption):
        run_training(cfg, progress_callback=stop_after_first_epoch)
    run_dir = next((tmp_path / "runs").iterdir())

    assert seen_subsets == {"train": ["train-a", "train-b"], "val": ["val-a"]}
    assert (run_dir / "checkpoints/best.pt").is_file()
    assert (run_dir / "checkpoints/resume_state.pt").is_file()
    saved_state = training.torch.load(
        run_dir / "checkpoints/resume_state.pt", map_location="cpu", weights_only=True
    )
    assert saved_state["seed"] == 42
    assert saved_state["batch_size"] == 2
    assert saved_state["patches_per_image"] == 1
    assert "torch_rng_state" in saved_state
    assert (run_dir / "checkpoints/best.onnx").read_bytes() == b"tiny fake onnx"
    assert all("test-secret" not in path for path in models[0].exported)
    assert saved_state["initialization_provenance"]["resume_mode"] == "fresh"

    resume_cfg = TrainingConfig(
        data_root=tmp_path,
        model_name="tiny",
        patch_training=True,
        img_size=32,
        patches_per_image=1,
        batch_size=2,
        epochs=2,
        patience=3,
        num_workers=0,
        run_dir=tmp_path / "resumed-runs",
        device="cpu",
        resume_checkpoint=run_dir / "checkpoints" / "resume_state.pt",
    )
    resumed, resumed_train, resumed_val, resumed_dir = run_training(resume_cfg)

    assert resumed["start_epoch"] == 2
    assert resumed["resume_mode"] == "full_state"
    assert resumed["initialization_provenance"]["continued_from_full_checkpoint"]
    assert resumed["initialization_provenance"]["resume_mode"] == "fresh"
    assert resumed["best_epoch"] in (1, 2)
    assert resumed["train_samples"] == 2
    assert resumed["training_device"] == "cpu"
    assert resumed["training_accelerator"] == "cpu"
    assert resumed["val_samples"] == 1
    assert resumed["test_samples"] == 1
    assert resumed["train_patches"] == 2
    assert resumed["val_patches"] == 1
    assert len(resumed_train) == len(resumed_val) == 2
    assert all(row["seconds"] >= 0 for row in resumed_train)
    assert (resumed_dir / "checkpoints/best.pt").is_file()
    assert (resumed_dir / "checkpoints/resume_state.pt").is_file()


def test_warm_start_provenance_tracks_checkpoint_hash_without_inventing_origin(
    tmp_path,
):
    from train.train import _initialization_provenance

    checkpoint = tmp_path / "legacy-best.pt"
    checkpoint.write_bytes(b"legacy state dict")
    provenance = _initialization_provenance(
        TrainingConfig(resume_checkpoint=checkpoint, pretrained=True),
        pretrained_used=False,
        resume_mode="warm_start",
    )

    assert provenance["pretrained_requested"] is True
    assert "pretrained_used_for_initial_run" not in provenance
    assert provenance["pretrained_used_for_current_run"] is False
    assert provenance["resume_checkpoint_mode"] == "warm_start"
    assert (
        provenance["warm_start_checkpoint_sha256"]
        == hashlib.sha256(b"legacy state dict").hexdigest()
    )
    assert provenance["warm_start_lineage"] == "not_recorded_in_legacy_checkpoint"
