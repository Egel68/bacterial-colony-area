"""End-to-end, reproducible COCO training/evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from .background_experiment import (
    combine_paired_evaluations,
    evaluate_paired_model,
    write_comparison_html,
)
from .config import TrainingConfig
from .dataset_adapters import CocoBboxImporter, load_manifest
from .evaluation import (
    build_training_report,
    compute_binary_metrics,
    evaluate_patch_records,
    evaluate_test_subset,
    verify_onnx_cpu_parity,
)

log = logging.getLogger(__name__)


def run_training(*args, **kwargs):
    """Загрузить PyTorch-цикл только при фактическом запуске обучения."""
    from .train import run_training as torch_run_training

    return torch_run_training(*args, **kwargs)


def _validate_black_background_dataset(
    manifest, dataset_root: Path, expected_source_ids: set[str]
) -> None:
    """Проверить полноту пар, общий mask и отсутствие split leakage."""
    source_records = [record for record in manifest.samples if record.kind == "source"]
    variant_records = [
        record for record in manifest.samples if record.kind == "augmented"
    ]
    sources = {record.id: record for record in source_records}
    variants = {}
    for record in variant_records:
        if record.source_id in variants:
            raise ValueError(
                f"Duplicate black-background variant for source '{record.source_id}'"
            )
        variants[record.source_id] = record

    if set(sources) != expected_source_ids:
        raise ValueError(
            "Prepared dataset source IDs do not exactly match the persisted split"
        )
    if set(variants) != expected_source_ids or len(variant_records) != len(sources):
        raise ValueError(
            "Prepared dataset must contain exactly one black-background variant per source"
        )
    if len(manifest.samples) != len(sources) + len(variants):
        raise ValueError(
            "Prepared black-background dataset contains unexpected records"
        )

    for source_id, source in sources.items():
        variant = variants[source_id]
        source_dish = source.dish
        variant_dish = variant.dish
        source_geometry = (
            (source_dish.cx, source_dish.cy, source_dish.r) if source_dish else None
        )
        variant_geometry = (
            (variant_dish.cx, variant_dish.cy, variant_dish.r) if variant_dish else None
        )
        if (
            variant.id != f"{source_id}__black_background"
            or source.subset not in {"train", "val", "test"}
            or variant.subset != source.subset
            or variant.mask != source.mask
            or source_geometry is None
            or variant_geometry != source_geometry
        ):
            raise ValueError(
                f"Prepared black-background pair is inconsistent for source '{source_id}'"
            )
        for record in (source, variant):
            for relative_path in (record.image, record.mask):
                if not (dataset_root / relative_path).is_file():
                    raise FileNotFoundError(
                        f"Prepared dataset file is missing for '{source_id}': {relative_path}"
                    )


def _tree_signature(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(":Zone.Identifier"):
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _cpu_model_name() -> str:
    """Вернуть модель CPU даже на Linux, где platform.processor() пустой."""
    processor = platform.processor()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.machine()


def _cpu_full_image_benchmark(
    model_path: Path,
    data_root: Path,
    *,
    tile_size: int = 512,
    stride: int = 384,
    image_limit: int | None = None,
) -> dict[str, object]:
    from testing.tiled_onnx_algorithm import TiledOnnxModelAlgorithm, tile_positions

    manifest = load_manifest(data_root)
    records = [record for record in manifest.samples if record.subset == "test"]
    if image_limit is not None:
        records = records[:image_limit]
    if not records:
        return {
            "images": [],
            "median_seconds": None,
            "p95_seconds": None,
            "metrics": {"mean": {}, "per_image": {}, "n_images": 0},
        }

    algo = TiledOnnxModelAlgorithm(
        model_path=str(model_path), img_size=tile_size, stride=stride
    )
    warmup_times = []
    measurements = []
    rows = []
    # Оставить хотя бы один снимок для измерения даже на маленьких наборах.
    warmup_count = min(3, max(0, len(records) - 1))
    metrics_per_image = {}
    for index, record in enumerate(records):
        image = cv2.imread(str(data_root / record.image), cv2.IMREAD_COLOR)
        target = cv2.imread(str(data_root / record.mask), cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise ValueError(f"Could not read benchmark image/mask pair '{record.id}'")
        started = time.perf_counter()
        mask = algo.detect(image, is_cropped=record.kind == "cropped")
        elapsed = time.perf_counter() - started
        if mask.shape != image.shape[:2]:
            raise RuntimeError(f"Tiled model returned invalid mask for '{record.id}'")
        metrics_per_image[record.id] = {
            **compute_binary_metrics(mask, target),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "subset": record.subset,
        }
        warmup = index < warmup_count
        (warmup_times if warmup else measurements).append(elapsed)
        rows.append(
            {
                "id": record.id,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "seconds": elapsed,
                "warmup": warmup,
                "tiles": len(tile_positions(image.shape[0], tile_size, stride))
                * len(tile_positions(image.shape[1], tile_size, stride)),
            }
        )
    metric_names = ("iou", "dice", "precision", "recall", "f1", "accuracy")
    metrics = {
        "mean": {
            name: float(np.mean([row[name] for row in metrics_per_image.values()]))
            for name in metric_names
        },
        "per_image": metrics_per_image,
        "n_images": len(metrics_per_image),
        "evaluation_mode": "onnx_cpu_full_resolution_tiled_test_once",
    }
    return {
        "images": rows,
        "median_seconds": statistics.median(measurements),
        "p95_seconds": float(np.percentile(measurements, 95)),
        "cpu_model": _cpu_model_name(),
        "logical_cores": os.cpu_count(),
        "onnxruntime_intra_op_num_threads": 0,
        "opencv_threads": cv2.getNumThreads(),
        "warmup_images": warmup_count,
        "warmup_seconds": warmup_times,
        "measured_images": len(measurements),
        "metrics": metrics,
        "note": "Warmup images are excluded; session setup is included in the first warmup.",
    }


def _standard_test_result_from_paired_model(
    model_result: dict[str, object],
) -> dict[str, object]:
    original = model_result["backgrounds"]["original"]
    per_sample = original["per_sample"]
    summary = original["summary"]
    metric_names = ("iou", "dice", "precision", "recall", "f1", "accuracy")
    return {
        "mean": {name: summary[name] for name in metric_names},
        "per_image": {sample["record_id"]: sample for sample in per_sample.values()},
        "n_images": len(per_sample),
        "evaluation_mode": "onnx_cpu_full_resolution_tiled_test_once",
    }


def run_colony_training(
    *,
    data_root: Path,
    output_root: Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    epochs: int = 100,
    batch_size: int = 8,
    patches_per_image: int = 4,
    img_size: int = 512,
    pretrained: bool = True,
    model_promotion_path: Path | None = None,
    allow_overwrite: bool = False,
    cpu_benchmark: bool = False,
    resume_run: Path | None = None,
    reuse_prepared_dataset: bool = False,
    model_name: str = "mobilenet_v3_small_unet",
    black_background: bool = False,
) -> dict[str, object]:
    if model_name not in {"mobilenet_v3_small_unet", "unet"}:
        raise ValueError(f"Unsupported train-colony model: {model_name}")
    if black_background and model_name != "unet":
        raise ValueError("Black-background experiment requires model_name='unet'")
    if black_background and resume_run is not None:
        raise ValueError(
            "Black-background experiment must start from random initialization"
        )
    if reuse_prepared_dataset and not black_background:
        raise ValueError(
            "Prepared-dataset reuse is only supported for black-background runs"
        )
    if reuse_prepared_dataset and resume_run is not None:
        raise ValueError("Cannot combine prepared-dataset reuse with --resume-run")
    if black_background:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "CUDA training requires the full PyTorch training environment"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This experiment requires a CUDA GPU; CPU fallback is disabled."
            )

    data_root = Path(data_root)
    output_root = Path(output_root)
    source_resolved = data_root.resolve()
    output_resolved = output_root.resolve()
    if (
        source_resolved == output_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError("Output root must be disjoint from the read-only COCO source")
    resume_run = Path(resume_run).resolve() if resume_run is not None else None
    dataset_root = output_root / "dataset"
    if output_root.exists() and any(output_root.iterdir()) and resume_run is None:
        existing_entries = {entry.name for entry in output_root.iterdir()}
        reusable_only = reuse_prepared_dataset and existing_entries == {"dataset"}
        if not reusable_only:
            raise FileExistsError(
                f"Refusing to use non-empty output directory: {output_root}; choose a new path"
            )
    if reuse_prepared_dataset and not (
        (dataset_root / "dataset.json").is_file()
        and (dataset_root / "split.json").is_file()
    ):
        raise FileNotFoundError(
            f"Prepared dataset reuse requires dataset.json and split.json in {dataset_root}"
        )
    if resume_run is not None:
        runs_root = (output_root / "runs").resolve()
        if runs_root not in resume_run.parents:
            raise ValueError("Resume run must be a child of <output>/runs")
        _resume_checkpoint(resume_run)
        if not (dataset_root / "dataset.json").is_file():
            raise FileNotFoundError(f"No prepared dataset found at {dataset_root}")
        if not (dataset_root / "split.json").is_file():
            raise FileNotFoundError(f"No persisted split found at {dataset_root}")
    if model_promotion_path is not None:
        model_promotion_path = Path(model_promotion_path)
        promotion_resolved = model_promotion_path.resolve()
        if (
            source_resolved == promotion_resolved
            or source_resolved in promotion_resolved.parents
        ):
            raise ValueError(
                "Promoted model path must not be inside the read-only COCO source"
            )
        if model_promotion_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing model: {model_promotion_path}; "
                "pass --allow-overwrite to confirm"
            )
        if (
            promotion_resolved == output_resolved
            or output_resolved in promotion_resolved.parents
        ):
            raise ValueError(
                "Promoted model path must not be inside the training output root"
            )
        legacy_model = (
            Path(__file__).resolve().parents[1] / "models" / "colony_seg.onnx"
        )
        if promotion_resolved == legacy_model.resolve():
            raise ValueError("Refusing to overwrite the legacy models/colony_seg.onnx")
    baseline_model_path = (
        Path(__file__).resolve().parents[1] / "models" / "colony_seg.onnx"
    )
    if black_background and not baseline_model_path.is_file():
        raise FileNotFoundError(
            f"Bundled baseline model not found: {baseline_model_path}"
        )
    baseline_evaluation = None
    source_signature = _tree_signature(data_root)

    if resume_run is None and not reuse_prepared_dataset:
        CocoBboxImporter().build(
            data_root=data_root,
            output_dir=dataset_root,
            split=True,
            seed=seed,
            split_ratios=ratios,
            black_background=black_background,
        )
    elif (
        not (dataset_root / "dataset.json").is_file()
        or not (dataset_root / "split.json").is_file()
    ):
        raise FileNotFoundError(f"No complete prepared dataset found at {dataset_root}")

    source_signature_after_import = _tree_signature(data_root)
    if source_signature_after_import != source_signature:
        raise RuntimeError(
            "Read-only COCO source changed during dataset preparation; "
            "training was stopped"
        )
    split = json.loads((dataset_root / "split.json").read_text(encoding="utf-8"))
    expected_ratios = dict(zip(("train", "val", "test"), ratios, strict=True))
    if split.get("seed") != seed or split.get("ratios") != expected_ratios:
        raise ValueError("Existing split seed/ratios do not match run arguments")
    coco = json.loads((data_root / "annot_COCO.json").read_text(encoding="utf-8"))
    annotated_ids = {
        str(annotation["image_id"]) for annotation in coco.get("annotations", [])
    }
    split_groups = split.get("groups", {})
    split_ids = {
        str(group["image_id"]) for groups in split_groups.values() for group in groups
    }
    if split_ids != annotated_ids:
        raise ValueError("Persisted split does not account for current COCO image IDs")
    assignments = {
        str(group["sample_id"]): subset
        for subset, groups in split_groups.items()
        for group in groups
    }
    if len(assignments) != len(split_ids):
        raise ValueError("Persisted split contains duplicate source sample IDs")

    manifest = load_manifest(dataset_root)
    source_records = [record for record in manifest.samples if record.kind == "source"]
    actual_assignments = {record.id: record.subset for record in source_records}
    if actual_assignments != assignments:
        raise ValueError(
            "Prepared source manifest assignments do not match persisted split"
        )
    if black_background:
        _validate_black_background_dataset(manifest, dataset_root, set(assignments))
    train_records = [record for record in manifest.samples if record.subset == "train"]
    val_records = [record for record in manifest.samples if record.subset == "val"]
    test_records = [record for record in manifest.samples if record.subset == "test"]
    if not train_records or not val_records or not test_records:
        raise ValueError(
            f"Split must have non-empty train/val/test: "
            f"{len(train_records)}/{len(val_records)}/{len(test_records)}"
        )
    if black_background:
        baseline_evaluation = evaluate_paired_model(
            baseline_model_path,
            manifest.samples,
            dataset_root,
            threshold=0.5,
            tile_size=img_size,
            stride=384,
        )

    cfg = TrainingConfig(
        data_root=dataset_root,
        model_name=model_name,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        patience=max(5, min(20, epochs // 5)),
        augment=True,
        run_dir=output_root / "runs",
        device="cuda",
        require_cuda=black_background,
        patch_training=True,
        patches_per_image=patches_per_image,
        pretrained=(pretrained if model_name == "mobilenet_v3_small_unet" else False),
        resume_checkpoint=_resume_checkpoint(resume_run),
    )
    summary, train_history, val_history, run_dir = run_training(cfg)
    source_signature_after_training = _tree_signature(data_root)
    if source_signature_after_training != source_signature:
        raise RuntimeError(
            "Read-only COCO source changed before checkpoint evaluation; "
            "training report was stopped"
        )
    summary["training_source_sha256"] = source_signature

    # Test evaluation is deliberately after run_training has frozen best_epoch.
    import torch

    from .models import get_model

    model = (
        get_model(model_name, pretrained=False)
        if model_name == "mobilenet_v3_small_unet"
        else get_model(model_name)
    )
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    manifest = load_manifest(dataset_root)
    train_records = [record for record in manifest.samples if record.subset == "train"]
    val_records = [record for record in manifest.samples if record.subset == "val"]
    train_result = evaluate_patch_records(
        model,
        train_records,
        dataset_root,
        patch_size=img_size,
        patches_per_image=patches_per_image,
        batch_size=batch_size,
        seed=seed,
    )
    validation_result = evaluate_patch_records(
        model,
        val_records,
        dataset_root,
        patch_size=img_size,
        patches_per_image=max(2, patches_per_image // 2),
        batch_size=batch_size,
        seed=seed,
    )

    onnx_path = run_dir / "checkpoints" / "best.onnx"
    retrained_evaluation = None
    paired_comparison = None
    if black_background:
        if baseline_evaluation is None:
            raise RuntimeError("Baseline evaluation was not completed before training")
        retrained_evaluation = evaluate_paired_model(
            onnx_path,
            manifest.samples,
            dataset_root,
            threshold=0.5,
            tile_size=img_size,
            stride=384,
        )
        paired_comparison = combine_paired_evaluations(
            baseline_evaluation,
            retrained_evaluation,
            threshold=0.5,
            tile_size=img_size,
            stride=384,
        )
    onnx_parity = verify_onnx_cpu_parity(
        model,
        onnx_path,
        val_records,
        dataset_root,
        patch_size=img_size,
        patches_per_image=max(2, patches_per_image // 2),
        max_patches=4,
        seed=seed,
    )
    benchmark_result = (
        _cpu_full_image_benchmark(
            onnx_path, dataset_root, tile_size=img_size, stride=384
        )
        if cpu_benchmark
        else None
    )
    if black_background:
        if paired_comparison is None:
            raise RuntimeError("Paired background comparison was not generated")
        test_result = _standard_test_result_from_paired_model(
            paired_comparison["models"]["retrained"]
        )
    else:
        test_result = (
            benchmark_result["metrics"]
            if benchmark_result is not None
            else evaluate_test_subset(model, dataset_root, tile_size=img_size)
        )
    report = build_training_report(
        data_root=dataset_root,
        cfg=cfg,
        summary=summary,
        train_history=train_history,
        val_history=val_history,
        test_result=test_result,
        onnx_path=onnx_path,
        train_result=train_result,
        validation_result=validation_result,
        onnx_parity=onnx_parity,
    )
    report["source_signature"] = {
        "before_import": source_signature,
        "after_import": source_signature_after_import,
        "after_run": None,
        "unchanged": None,
    }
    report["prepared_dataset_reused"] = reuse_prepared_dataset
    report["cpu"].update(
        {
            "model": _cpu_model_name(),
            "logical_cores": os.cpu_count(),
        }
    )

    report["cpu_benchmark"] = benchmark_result
    if resume_run is not None:
        report["resumed_from"] = str(_resume_checkpoint(resume_run))

    report_path = run_dir / "report.json"
    source_signature_after_run = _tree_signature(data_root)
    if source_signature_after_run != source_signature:
        raise RuntimeError(
            "Read-only COCO source changed during training/evaluation; "
            "report promotion was stopped"
        )
    report["source_signature"] = {
        "before_import": source_signature,
        "after_import": source_signature_after_import,
        "after_run": source_signature_after_run,
        "unchanged": True,
    }
    if paired_comparison is not None:
        split_ids = {
            subset: sorted(
                record.id
                for record in manifest.samples
                if record.kind == "source" and record.subset == subset
            )
            for subset in ("train", "val", "test")
        }
        if paired_comparison["test_source_ids"] != split_ids["test"]:
            raise RuntimeError("Paired inference IDs differ from persisted test split")
        paired_comparison.update(
            {
                "seed": seed,
                "ratios": dict(zip(("train", "val", "test"), ratios, strict=True)),
                "split_ids": split_ids,
                "configuration": {
                    "architecture": model_name,
                    "initialization": "random",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "patches_per_image": patches_per_image,
                    "img_size": img_size,
                    "seed": seed,
                    "pretrained": False,
                    "resume_checkpoint": None,
                    "online_augmentation": cfg.augment,
                },
                "source_provenance": {
                    "path": str(data_root.resolve()),
                    "sha256": source_signature_after_run,
                    "hash_algorithm": "sha256",
                    "unchanged": True,
                },
            }
        )
        paired_comparison["models"]["baseline"].update(
            {
                "label": "Bundled baseline",
                "training_membership": "unknown",
            }
        )
        paired_comparison["models"]["retrained"].update(
            {
                "label": "Scratch U-Net",
                "architecture": model_name,
                "initialization": "random",
                "checkpoint": str(checkpoint_path),
                "initialization_provenance": summary.get(
                    "initialization_provenance", {}
                ),
                "best_epoch": summary.get("best_epoch"),
                "best_val_iou": summary.get("best_iou"),
            }
        )
        comparison_json_path = run_dir / "comparison.json"
        comparison_html_path = run_dir / "comparison.html"
        comparison_json_path.write_text(
            json.dumps(paired_comparison, indent=2, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        write_comparison_html(paired_comparison, comparison_html_path)
        report["paired_background_experiment"] = paired_comparison
        report["comparison_json"] = str(comparison_json_path)
        report["comparison_html"] = str(comparison_html_path)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    if model_promotion_path is not None:
        if model_promotion_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing model: {model_promotion_path}; "
                "pass --allow-overwrite to confirm"
            )
        model_promotion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, model_promotion_path)
        report["promoted_model"] = str(model_promotion_path)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log.info("Run report: %s", report_path)
    log.info("Test metrics: %s", test_result["mean"])
    return report


def _resume_checkpoint(resume_run: Path | None) -> Path | None:
    if resume_run is None:
        return None
    checkpoints = resume_run / "checkpoints"
    for filename in ("resume_state.pt", "last.pt", "best.pt"):
        checkpoint = checkpoints / filename
        if checkpoint.is_file():
            return checkpoint
    raise FileNotFoundError(f"No resumable checkpoint found in {checkpoints}")
