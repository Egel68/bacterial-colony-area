"""Финальная оценка сегментационной модели и отчёт запуска обучения."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .dataset_adapters import load_manifest

if TYPE_CHECKING:
    from .dataset_manifest import SampleRecord
    from .models.base import BaseSegmenter

log = logging.getLogger(__name__)


def compute_binary_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Вычислить бинарные метрики на бинарных/gray масках 0–255."""
    predicted = np.asarray(prediction) > 0
    expected = np.asarray(target) > 0
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected))
    fn = int(np.count_nonzero(~predicted & expected))
    tn = int(np.count_nonzero(~predicted & ~expected))
    smooth = 1e-6
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return {
        "iou": tp / (tp + fp + fn + smooth),
        "dice": 2 * tp / (2 * tp + fp + fn + smooth),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + smooth),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + smooth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    return list(range(0, length, stride))


def predict_full_image_tiled(
    model: BaseSegmenter,
    image_bgr: np.ndarray,
    *,
    tile_size: int = 512,
    stride: int = 384,
    progress_callback=None,
) -> np.ndarray:
    """PyTorch reference inference matching the production tile/stitch path."""
    import torch

    height, width = image_bgr.shape[:2]
    y_positions = _tile_positions(height, tile_size, stride)
    x_positions = _tile_positions(width, tile_size, stride)
    total = len(y_positions) * len(x_positions)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    device = next(model.parameters()).device
    model.eval()
    completed = 0
    for y in y_positions:
        for x in x_positions:
            y2, x2 = min(height, y + tile_size), min(width, x + tile_size)
            tile = image_bgr[y:y2, x:x2]
            bottom = tile_size - tile.shape[0]
            right = tile_size - tile.shape[1]
            if bottom or right:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    bottom,
                    0,
                    right,
                    cv2.BORDER_REFLECT_101
                    if tile.shape[0] > 1 and tile.shape[1] > 1
                    else cv2.BORDER_REPLICATE,
                )
            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(((rgb - mean) / std).copy())
                .permute(2, 0, 1)[None]
                .to(device)
            )
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
            patch_height, patch_width = y2 - y, x2 - x
            probability_sum[y:y2, x:x2] += probabilities[:patch_height, :patch_width]
            coverage[y:y2, x:x2] += 1
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    if np.any(coverage == 0):
        raise RuntimeError("Tiled inference left pixels uncovered")
    probability_sum /= coverage
    return (probability_sum >= 0.5).astype(np.uint8) * 255


def evaluate_model_on_records(
    model: BaseSegmenter,
    records: list[SampleRecord],
    data_root: Path,
    *,
    tile_size: int = 512,
    stride: int = 384,
    limit: int | None = None,
) -> dict[str, object]:
    """Оценить каждую исходную запись целиком и усреднить метрики по снимкам."""
    selected = records[:limit] if limit is not None else records
    per_image = {}
    for record in selected:
        image = cv2.imread(str(data_root / record.image), cv2.IMREAD_COLOR)
        target = cv2.imread(str(data_root / record.mask), cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise ValueError(f"Could not read test pair '{record.id}'")
        prediction = predict_full_image_tiled(
            model, image, tile_size=tile_size, stride=stride
        )
        per_image[record.id] = {
            **compute_binary_metrics(prediction, target),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "subset": record.subset,
        }
    metric_names = ("iou", "dice", "precision", "recall", "f1", "accuracy")
    mean = (
        {
            key: float(np.mean([metrics[key] for metrics in per_image.values()]))
            for key in metric_names
        }
        if per_image
        else {key: 0.0 for key in metric_names}
    )
    return {"mean": mean, "per_image": per_image, "n_images": len(per_image)}


def evaluate_test_subset(
    model: BaseSegmenter,
    data_root: Path,
    *,
    tile_size: int = 512,
    stride: int = 384,
    limit: int | None = None,
) -> dict[str, object]:
    from .dataset import make_test_dataset

    dataset = make_test_dataset(data_root, tile_size)
    if not dataset.records:
        raise ValueError("No test subset found in the manifest")
    return evaluate_model_on_records(
        model,
        dataset.records,
        data_root,
        tile_size=tile_size,
        stride=stride,
        limit=limit,
    )


def evaluate_patch_records(
    model: BaseSegmenter,
    records: list[SampleRecord],
    data_root: Path,
    *,
    patch_size: int = 512,
    patches_per_image: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, object]:
    """Оценить фиксированные патчи выбранных записей сгруппированно по исходнику."""
    import torch
    from torch.utils.data import DataLoader

    from .dataset import ColonyPatchDataset

    dataset = ColonyPatchDataset(
        records,
        data_root,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        seed=seed,
        augment=False,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = next(model.parameters()).device
    model.eval()
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        counts[record.id] = {key: 0 for key in ("tp", "fp", "fn", "tn")}

    patch_index = 0
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            predictions = torch.sigmoid(logits) >= 0.5
            expected = masks.to(device) > 0.5
            for offset in range(images.shape[0]):
                record_index = patch_index // patches_per_image
                record = records[record_index]
                prediction = predictions[offset]
                target = expected[offset]
                group = counts[record.id]
                group["tp"] += int((prediction & target).sum().item())
                group["fp"] += int((prediction & ~target).sum().item())
                group["fn"] += int((~prediction & target).sum().item())
                group["tn"] += int((~prediction & ~target).sum().item())
                patch_index += 1

    per_image = {}
    for record in records:
        counts_for_image = counts[record.id]
        metric_values = compute_binary_metrics_from_counts(counts_for_image)
        width, height = dataset.image_dimensions[record.id]
        per_image[record.id] = {
            **metric_values,
            "width": width,
            "height": height,
            "subset": record.subset,
            "patches": patches_per_image,
        }
    metric_names = ("iou", "dice", "precision", "recall", "f1", "accuracy")
    mean = (
        {
            key: float(np.mean([metrics[key] for metrics in per_image.values()]))
            for key in metric_names
        }
        if per_image
        else {key: 0.0 for key in metric_names}
    )
    return {"mean": mean, "per_image": per_image, "n_images": len(per_image)}


def verify_onnx_cpu_parity(
    model: BaseSegmenter,
    onnx_path: Path,
    records: list[SampleRecord],
    data_root: Path,
    *,
    patch_size: int = 512,
    patches_per_image: int = 2,
    max_patches: int = 4,
    seed: int = 42,
    tolerance: float = 1e-4,
    logit_tolerance: float = 2e-4,
) -> dict[str, object]:
    """Сверить вероятности ONNX с PyTorch на фиксированных validation-патчах."""
    import math
    import torch

    import onnxruntime
    from .dataset import ColonyPatchDataset

    validation_records = [record for record in records if record.subset == "val"]
    if not validation_records:
        raise ValueError("ONNX parity check requires validation records")
    if max_patches <= 0 or patches_per_image <= 0:
        raise ValueError("max_patches and patches_per_image must be positive")

    record_limit = math.ceil(max_patches / patches_per_image)
    dataset = ColonyPatchDataset(
        validation_records[:record_limit],
        data_root,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        seed=seed,
        augment=False,
        training=False,
    )
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Expected CPU-only ONNX Runtime session, got {providers}")

    original_device = next(model.parameters()).device
    model = model.to("cpu").eval()
    probability_errors = []
    logit_errors = []
    threshold_disagreements = 0
    compared_pixels = 0
    try:
        with torch.inference_mode():
            for index in range(min(max_patches, len(dataset))):
                image, _ = dataset[index]
                input_array = image.unsqueeze(0).numpy()
                torch_logits = model(torch.from_numpy(input_array)).numpy()
                onnx_logits = session.run(
                    ["output"], {session.get_inputs()[0].name: input_array}
                )[0]
                if torch_logits.shape != onnx_logits.shape:
                    raise RuntimeError(
                        "PyTorch/ONNX output shape mismatch: "
                        f"{torch_logits.shape} != {onnx_logits.shape}"
                    )
                torch_probabilities = torch.sigmoid(
                    torch.from_numpy(torch_logits)
                ).numpy()
                onnx_probabilities = torch.sigmoid(
                    torch.from_numpy(onnx_logits)
                ).numpy()
                probability_errors.append(
                    float(np.max(np.abs(torch_probabilities - onnx_probabilities)))
                )
                logit_errors.append(float(np.max(np.abs(torch_logits - onnx_logits))))
                threshold_disagreements += int(
                    np.count_nonzero(
                        (torch_probabilities >= 0.5) != (onnx_probabilities >= 0.5)
                    )
                )
                compared_pixels += int(torch_probabilities.size)
    finally:
        model.to(original_device)

    max_abs_diff = max(logit_errors, default=0.0)
    max_abs_probability_diff = max(probability_errors, default=0.0)
    result = {
        "providers": providers,
        "patches": len(probability_errors),
        "comparison": "logits_and_sigmoid_probabilities",
        "max_abs_diff": max_abs_diff,
        "mean_abs_max_diff": float(np.mean(logit_errors)) if logit_errors else 0.0,
        "tolerance": tolerance,
        "logit_tolerance": logit_tolerance,
        "max_abs_probability_diff": max_abs_probability_diff,
        "mean_abs_max_probability_diff": (
            float(np.mean(probability_errors)) if probability_errors else 0.0
        ),
        "probability_tolerance": tolerance,
        "threshold_disagreements": threshold_disagreements,
        "compared_pixels": compared_pixels,
        "threshold_disagreement_fraction": (
            threshold_disagreements / compared_pixels if compared_pixels else 0.0
        ),
        "passed": max_abs_diff <= logit_tolerance
        and max_abs_probability_diff <= tolerance,
    }
    if not result["passed"]:
        raise RuntimeError(
            "ONNX CPU output differs from PyTorch beyond tolerance: "
            f"logits {max_abs_diff:.6g} > {logit_tolerance:.6g} or "
            f"probabilities {max_abs_probability_diff:.6g} > {tolerance:.6g}"
        )
    return result


def compute_binary_metrics_from_counts(
    counts: dict[str, int],
) -> dict[str, float | int]:
    """Вычислить метрики из сумм confusion matrix нескольких patches."""
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    smooth = 1e-6
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return {
        "iou": tp / (tp + fp + fn + smooth),
        "dice": 2 * tp / (2 * tp + fp + fn + smooth),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + smooth),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + smooth),
        **counts,
    }


def class_counts(data_root: Path) -> dict[str, dict[str, int]]:
    """Посчитать число уникальных исходников на subset/category по split.json."""
    split_path = Path(data_root) / "split.json"
    if not split_path.is_file():
        return {}
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    for subset, groups in raw.get("groups", {}).items():
        by_category: dict[str, int] = defaultdict(int)
        for group in groups:
            by_category[str(group["category_id"])] += 1
        counts[subset] = dict(sorted(by_category.items()))
    return counts


def git_revision_info() -> dict[str, str | bool | None]:
    """Зафиксировать commit и наличие локальных изменений для воспроизводимости."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": True}
    return {"revision": revision, "dirty": dirty}


def build_training_report(
    *,
    data_root: Path,
    cfg,
    summary: dict,
    train_history: list[dict],
    val_history: list[dict],
    test_result: dict[str, object] | None,
    onnx_path: Path | None,
    train_result: dict[str, object] | None = None,
    validation_result: dict[str, object] | None = None,
    onnx_parity: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = load_manifest(data_root)
    best_epoch = summary.get("best_epoch")
    train_at_best = next(
        (row for row in train_history if row.get("epoch") == best_epoch),
        {},
    )
    val_at_best = next(
        (row for row in val_history if row.get("epoch") == best_epoch),
        {},
    )
    # Совместимость с сохранёнными до epoch field train histories.
    if best_epoch and not train_at_best and train_history:
        train_at_best = train_history[min(int(best_epoch) - 1, len(train_history) - 1)]
    if best_epoch and not val_at_best and val_history:
        val_at_best = val_history[min(int(best_epoch) - 1, len(val_history) - 1)]
    metric_names = ("iou", "dice", "precision", "recall", "f1", "accuracy")
    train_mean = {
        key: train_at_best[key] for key in metric_names if key in train_at_best
    }
    validation_mean = {
        key: val_at_best[key] for key in metric_names if key in val_at_best
    }
    split_counts = {
        subset: sum(record.subset == subset for record in manifest.samples)
        for subset in ("train", "val", "test")
    }
    source_signature = [
        {
            "id": record.id,
            "image": str(record.image),
            "mask": str(record.mask),
            "subset": record.subset,
        }
        for record in manifest.samples
    ]
    train_result = train_result or {"mean": train_mean, "per_image": {}, "n_images": 0}
    validation_result = validation_result or {
        "mean": validation_mean,
        "per_image": {},
        "n_images": 0,
    }
    return {
        "model": cfg.model_name,
        "git": git_revision_info(),
        "config": {
            "seed": cfg.seed,
            "img_size": cfg.img_size,
            "epochs": summary.get("training_config", {}).get(
                "epochs_requested", cfg.epochs
            ),
            "batch_size": summary.get("training_config", {}).get(
                "batch_size", cfg.batch_size
            ),
            "patches_per_image": summary.get("training_config", {}).get(
                "patches_per_image", cfg.patches_per_image
            ),
            "pretrained": summary.get("pretrained", cfg.pretrained),
            "pretrained_requested": cfg.pretrained,
            "pretrained_used_for_current_run": summary.get("pretrained"),
            "pretrained_used_for_initial_run": summary.get(
                "pretrained_used_for_initial_run"
            ),
            "initialization_source": summary.get("initialization_source"),
            "initialization_provenance": summary.get("initialization_provenance"),
            "lr": summary.get("training_config", {}).get("learning_rate", cfg.lr),
            "weight_decay": summary.get("training_config", {}).get(
                "weight_decay", cfg.weight_decay
            ),
            "patience": summary.get("training_config", {}).get(
                "patience", cfg.patience
            ),
        },
        "split": json.loads(
            (Path(data_root) / "split.json").read_text(encoding="utf-8")
        )
        if (Path(data_root) / "split.json").is_file()
        else None,
        "class_counts": class_counts(data_root),
        "source_signature": source_signature,
        "training_source_sha256": summary.get("training_source_sha256"),
        "training_summary": summary,
        "initialization_provenance": summary.get("initialization_provenance"),
        "best_epoch": summary.get("best_epoch"),
        "train_history": train_history,
        "validation_history": val_history,
        "validation_mean": {
            key: summary.get(f"best_val_{key}")
            for key in ("iou", "dice", "precision", "recall")
        },
        "test": test_result,
        "onnx_parity": onnx_parity,
        "split_metrics": {
            "train": {
                "mean": train_result.get("mean", train_mean),
                "sample_count": split_counts["train"],
                "evaluation_mode": "fixed_train_only_patches_after_checkpoint_selection",
                "n_images": train_result.get("n_images", 0),
                "per_image": train_result.get("per_image", {}),
            },
            "val": {
                "mean": validation_result.get("mean", validation_mean),
                "sample_count": split_counts["val"],
                "evaluation_mode": "fixed_validation_patches_after_checkpoint_selection",
                "n_images": validation_result.get("n_images", 0),
                "per_image": validation_result.get("per_image", {}),
            },
            "test": {
                "mean": test_result.get("mean", {}) if test_result else {},
                "sample_count": split_counts["test"],
                "evaluation_mode": test_result.get(
                    "evaluation_mode", "pytorch_full_resolution_tiled_test_only"
                )
                if test_result
                else "not_evaluated",
                "per_image": test_result.get("per_image", {}) if test_result else {},
            },
        },
        "onnx": {
            "path": str(onnx_path) if onnx_path else None,
            "bytes": onnx_path.stat().st_size
            if onnx_path and onnx_path.is_file()
            else None,
            "sha256": _file_sha256(onnx_path)
            if onnx_path and onnx_path.is_file()
            else None,
        },
        "cpu": {
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "logical_cores": os.cpu_count(),
        },
        "weak_label_note": "Ground-truth masks are filled ellipses rasterized from COCO bounding boxes.",
    }


def _file_sha256(path: Path) -> str:
    """Вычислить SHA-256 артефакта потоково, не загружая его целиком в RAM."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
