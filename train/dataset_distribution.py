"""Проверка и локальная подготовка self-contained training snapshots."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2

from .dataset_adapters import CocoBboxImporter, load_manifest, validate_coco_inputs
from .dataset_manifest import DatasetManifest, SampleRecord

SUBSETS = ("train", "val", "test")


def validate_prepared_dataset(
    data_root: Path, *, require_black_background: bool = False
) -> tuple[DatasetManifest, dict[str, Any]]:
    """Проверить split, связи манифеста и читаемость всех подготовленных пар."""
    root = Path(data_root).resolve()
    split_path = root / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Prepared dataset is missing split.json: {split_path}")

    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read prepared split: {split_path}") from error
    if not isinstance(split, dict):
        raise ValueError("Prepared split.json must contain a JSON object")

    seed = split.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Prepared split.json must contain an integer seed")
    ratios = split.get("ratios")
    if not isinstance(ratios, dict) or set(ratios) != set(SUBSETS):
        raise ValueError("Prepared split.json must contain train/val/test ratios")
    try:
        ratio_values = tuple(float(ratios[subset]) for subset in SUBSETS)
    except (TypeError, ValueError) as error:
        raise ValueError("Prepared split ratios must be numeric") from error
    if any(not math.isfinite(value) or value <= 0 for value in ratio_values) or not math.isclose(sum(ratio_values), 1.0):
        raise ValueError("Prepared split ratios must be positive and sum to 1")
    if split.get("strategy") != "stratified_source_image_id":
        raise ValueError("Prepared split.json has an unsupported grouping strategy")

    groups = split.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(SUBSETS):
        raise ValueError("Prepared split.json must include train, val and test groups")
    assignments: dict[str, str] = {}
    image_ids: set[str] = set()
    for subset in SUBSETS:
        subset_groups = groups[subset]
        if not isinstance(subset_groups, list):
            raise ValueError(f"Prepared split group '{subset}' must be a list")
        for group in subset_groups:
            if not isinstance(group, dict) or not {"image_id", "sample_id"} <= set(group):
                raise ValueError(f"Prepared split contains an invalid '{subset}' group")
            image_id = str(group["image_id"])
            sample_id = group["sample_id"]
            if not image_id or not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Prepared split contains an invalid '{subset}' group")
            if image_id in image_ids:
                raise ValueError(f"Prepared split duplicates COCO image id '{image_id}'")
            if sample_id in assignments:
                raise ValueError(f"Prepared split duplicates source id '{sample_id}'")
            image_ids.add(image_id)
            assignments[sample_id] = subset

    manifest = load_manifest(root)
    source_records = [record for record in manifest.samples if record.kind == "source"]
    sources_by_id = {record.id: record for record in source_records}
    if len(sources_by_id) != len(source_records):
        raise ValueError("Prepared source manifest contains duplicate sample IDs")
    if set(sources_by_id) != set(assignments):
        raise ValueError("Prepared source IDs do not exactly match split.json groups")
    if any(record.subset != assignments[record.id] for record in source_records):
        raise ValueError("Prepared source manifest assignments do not match split.json")
    if any(not any(record.subset == subset for record in source_records) for subset in SUBSETS):
        raise ValueError("Prepared split must have non-empty train/val/test groups")

    variants_by_source: dict[str, SampleRecord] = {}
    for record in manifest.samples:
        if record.kind == "source":
            continue
        source_id = record.source_id
        if source_id is None and record.kind == "cropped" and record.id.endswith("_cropped"):
            source_id = record.id.removesuffix("_cropped")
        if source_id not in sources_by_id:
            raise ValueError(f"Derived record '{record.id}' has no matching source ID")
        if record.subset != assignments[source_id]:
            raise ValueError(f"Derived record '{record.id}' leaks across split groups")
        if record.source_id is not None:
            if source_id in variants_by_source:
                raise ValueError(f"Duplicate derived variant for source '{source_id}'")
            variants_by_source[source_id] = record

    if require_black_background:
        if set(variants_by_source) != set(sources_by_id):
            raise ValueError("Prepared dataset requires one black-background variant per source")
        for source_id, source in sources_by_id.items():
            variant = variants_by_source[source_id]
            source_geometry = (
                (source.dish.cx, source.dish.cy, source.dish.r) if source.dish else None
            )
            variant_geometry = (
                (variant.dish.cx, variant.dish.cy, variant.dish.r)
                if variant.dish
                else None
            )
            if (
                variant.kind != "augmented"
                or variant.id != f"{source_id}__black_background"
                or variant.mask != source.mask
                or variant.subset != source.subset
                or source_geometry is None
                or variant_geometry != source_geometry
            ):
                raise ValueError(f"Black-background pair is inconsistent for '{source_id}'")
        if len(manifest.samples) != len(sources_by_id) * 2:
            raise ValueError("Prepared black-background dataset contains unexpected records")

    root_resolved = root.resolve()
    image_shapes: dict[Path, tuple[int, int]] = {}
    mask_shapes: dict[Path, tuple[int, int]] = {}
    for record in manifest.samples:
        pair_shapes = []
        for relative_path, cache, flag, label in (
            (record.image, image_shapes, cv2.IMREAD_COLOR, "image"),
            (record.mask, mask_shapes, cv2.IMREAD_GRAYSCALE, "mask"),
        ):
            relative_path = Path(relative_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe {label} path for sample '{record.id}': {relative_path}")
            full_path = (root / relative_path).resolve()
            if not full_path.is_relative_to(root_resolved):
                raise ValueError(f"{label.capitalize()} path escapes prepared dataset: {relative_path}")
            if full_path not in cache:
                decoded = cv2.imread(str(full_path), flag)
                if decoded is None:
                    raise ValueError(f"Could not decode prepared {label} for '{record.id}': {relative_path}")
                cache[full_path] = decoded.shape[:2]
            pair_shapes.append(cache[full_path])
        if pair_shapes[0] != pair_shapes[1]:
            raise ValueError(f"Prepared image/mask dimensions differ for '{record.id}'")

    return manifest, split


def prepare_training_dataset(
    data_root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    black_background: bool = True,
) -> dict[str, Any]:
    """Атомарно создать проверенный training-ready snapshot из COCO-источника."""
    source = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Prepared dataset output must be disjoint from the read-only COCO source")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing prepared dataset: {output}")

    preflight = validate_coco_inputs(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    importer = CocoBboxImporter()
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.prepare-", dir=output.parent
    ) as workspace:
        staged_root = Path(workspace) / output.name
        manifest = importer.build(
            data_root=source,
            output_dir=staged_root,
            split=True,
            seed=seed,
            split_ratios=ratios,
            black_background=black_background,
        )
        shutil.copyfile(source / "annot_COCO.json", staged_root / "annot_COCO.json")
        manifest.name = output.name
        importer._write_dataset_json(staged_root, manifest)
        checked_manifest, split = validate_prepared_dataset(
            staged_root, require_black_background=black_background
        )
        os.replace(staged_root, output)

    counts = {
        subset: len(split["groups"][subset])
        for subset in SUBSETS
    }
    return {
        "output": str(output),
        "annotated_images": preflight["annotated_images"],
        "annotations": preflight["annotations"],
        "samples": len(checked_manifest.samples),
        "sources": sum(record.kind == "source" for record in checked_manifest.samples),
        "black_background_variants": sum(
            record.kind == "augmented" for record in checked_manifest.samples
        ),
        "split_counts": counts,
        "seed": seed,
        "ratios": dict(zip(SUBSETS, ratios, strict=True)),
    }
