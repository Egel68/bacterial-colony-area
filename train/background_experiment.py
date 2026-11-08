"""Сопоставимая оценка baseline и scratch-модели на парных фонах."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import cv2
import numpy as np

from .dataset_manifest import SampleRecord
from .evaluation import binary_confusion_counts, summarize_confusion_counts

DEFAULT_THRESHOLD = 0.5
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_STRIDE = 384
BACKGROUND_NAMES = ("original", "black")


def _test_pairs(
    records: list[SampleRecord],
) -> tuple[dict[str, SampleRecord], dict[str, SampleRecord]]:
    test_records = [record for record in records if record.subset == "test"]
    sources: dict[str, SampleRecord] = {}
    variants: dict[str, SampleRecord] = {}

    for record in test_records:
        if record.kind == "source":
            if record.id in sources:
                raise ValueError(f"Duplicate test source ID: {record.id}")
            sources[record.id] = record
        elif record.kind == "augmented":
            if not record.source_id:
                raise ValueError(
                    f"Test background variant '{record.id}' has no source_id"
                )
            if record.source_id in variants:
                raise ValueError(
                    f"Multiple test background variants for '{record.source_id}'"
                )
            variants[record.source_id] = record

    if not sources:
        raise ValueError("No held-out source records found in subset=test")
    if set(sources) != set(variants):
        missing = sorted(set(sources) - set(variants))
        unknown = sorted(set(variants) - set(sources))
        raise ValueError(
            "Test source/black-background IDs do not match "
            f"(missing variants={missing}, orphan variants={unknown})"
        )

    for source_id, source in sources.items():
        variant = variants[source_id]
        if variant.subset != source.subset or variant.mask != source.mask:
            raise ValueError(
                f"Test source/variant metadata differs for source ID '{source_id}'"
            )
        expected_id = f"{source_id}__black_background"
        if variant.id != expected_id:
            raise ValueError(
                f"Unexpected black-background ID '{variant.id}' for '{source_id}'"
            )
    return sources, variants


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_paired_model(
    model_path: Path,
    records: list[SampleRecord],
    data_root: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int = DEFAULT_TILE_STRIDE,
) -> dict[str, object]:
    """Оценить один ONNX-модельный файл на тех же source/black test-парах."""
    from testing.tiled_onnx_algorithm import TiledOnnxModelAlgorithm

    model_path = Path(model_path)
    data_root = Path(data_root)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    sources, variants = _test_pairs(records)
    source_ids = sorted(sources)
    algorithm = TiledOnnxModelAlgorithm(
        model_path=str(model_path),
        img_size=tile_size,
        stride=stride,
        threshold=threshold,
        output_is_logits=True,
    )

    backgrounds: dict[str, dict[str, object]] = {}
    for background, record_map in (("original", sources), ("black", variants)):
        per_sample: dict[str, dict[str, object]] = {}
        totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for source_id in source_ids:
            record = record_map[source_id]
            image = cv2.imread(str(data_root / record.image), cv2.IMREAD_COLOR)
            target = cv2.imread(str(data_root / record.mask), cv2.IMREAD_GRAYSCALE)
            if image is None or target is None:
                raise ValueError(f"Could not read test pair '{record.id}'")
            if image.shape[:2] != target.shape:
                raise ValueError(
                    f"Image/mask dimensions differ for test sample '{record.id}'"
                )
            prediction = algorithm.detect(image, is_cropped=False)
            if np.asarray(prediction).shape != target.shape:
                raise ValueError(
                    f"Model returned an invalid mask for test sample '{record.id}'"
                )
            counts = binary_confusion_counts(prediction, target)
            metrics = summarize_confusion_counts(counts)
            per_sample[source_id] = {
                "source_id": source_id,
                "record_id": record.id,
                "image": str(record.image),
                "mask": str(record.mask),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                **metrics,
            }
            for name in totals:
                totals[name] += counts[name]

        backgrounds[background] = {
            "source_ids": source_ids,
            "per_sample": per_sample,
            "summary": summarize_confusion_counts(totals),
        }

    return {
        "path": str(model_path),
        "sha256": _sha256(model_path),
        "source_ids": source_ids,
        "backgrounds": backgrounds,
    }


def combine_paired_evaluations(
    baseline: dict[str, object],
    retrained: dict[str, object],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int = DEFAULT_TILE_STRIDE,
) -> dict[str, object]:
    """Собрать два прогона только при полном совпадении held-out source ID."""
    test_source_ids = baseline["source_ids"]
    if test_source_ids != retrained["source_ids"]:
        raise ValueError("Baseline and retrained model test source IDs differ")
    for model_name, result in (("baseline", baseline), ("retrained", retrained)):
        for background in BACKGROUND_NAMES:
            condition_ids = result["backgrounds"][background]["source_ids"]
            if condition_ids != test_source_ids:
                raise ValueError(
                    f"{model_name} {background} results do not match held-out IDs"
                )
            if set(result["backgrounds"][background]["per_sample"]) != set(
                test_source_ids
            ):
                raise ValueError(
                    f"{model_name} {background} per-sample metrics have mismatched IDs"
                )

    return {
        "test_source_ids": test_source_ids,
        "baseline_training_overlap": "unknown",
        "threshold": threshold,
        "inference": {
            "mode": "full_resolution_tiled_cpu_onnx",
            "tile_size": tile_size,
            "stride": stride,
            "output_is_logits": True,
        },
        "models": {"baseline": baseline, "retrained": retrained},
    }


def compare_paired_models(
    baseline_path: Path,
    retrained_path: Path,
    records: list[SampleRecord],
    data_root: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int = DEFAULT_TILE_STRIDE,
) -> dict[str, object]:
    """Сравнить модели, принудительно проверив равенство held-out sample ID."""
    baseline = evaluate_paired_model(
        baseline_path,
        records,
        data_root,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
    )
    retrained = evaluate_paired_model(
        retrained_path,
        records,
        data_root,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
    )
    return combine_paired_evaluations(
        baseline,
        retrained,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
    )


def write_comparison_html(report: dict[str, object], output_path: Path) -> Path:
    """Записать компактный HTML со сводками, матрицами ошибок и ID выборок."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = report["models"]
    summary_rows = []
    sample_rows = []
    for model_name in ("baseline", "retrained"):
        model = models[model_name]
        label = html.escape(str(model.get("label", model_name)))
        for background in BACKGROUND_NAMES:
            result = model["backgrounds"][background]
            summary = result["summary"]
            matrix = summary["confusion_matrix"]
            matrix_text = (
                f"[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]"
            )
            summary_rows.append(
                "<tr>"
                f"<td>{label}</td><td>{html.escape(background)}</td>"
                f"<td>{len(result['source_ids'])}</td>"
                f"<td>{summary['tp']}</td><td>{summary['fp']}</td>"
                f"<td>{summary['fn']}</td><td>{summary['tn']}</td>"
                f"<td>{summary['precision']:.6f}</td>"
                f"<td>{summary['recall']:.6f}</td>"
                f"<td><code>{matrix_text}</code></td></tr>"
            )
            for source_id, sample in sorted(result["per_sample"].items()):
                sample_rows.append(
                    "<tr>"
                    f"<td>{html.escape(label)}</td>"
                    f"<td>{html.escape(background)}</td>"
                    f"<td>{html.escape(source_id)}</td>"
                    f"<td>{html.escape(str(sample['record_id']))}</td>"
                    f"<td>{sample['tp']}</td><td>{sample['fp']}</td>"
                    f"<td>{sample['fn']}</td><td>{sample['tn']}</td>"
                    f"<td>{sample['precision']:.6f}</td>"
                    f"<td>{sample['recall']:.6f}</td></tr>"
                )

    split_ids = report.get("split_ids", {})
    split_sections = []
    for subset in ("train", "val", "test"):
        ids = split_ids.get(subset, [])
        split_sections.append(
            f"<h3>{html.escape(subset)} ({len(ids)})</h3><p>"
            + html.escape(", ".join(ids))
            + "</p>"
        )
    test_ids = html.escape(", ".join(report.get("test_source_ids", [])))
    config = html.escape(str(report.get("configuration", {})))
    source_provenance = html.escape(str(report.get("source_provenance", {})))
    baseline_overlap = html.escape(
        str(report.get("baseline_training_overlap", "unknown"))
    )
    threshold = html.escape(str(report.get("threshold", DEFAULT_THRESHOLD)))
    inference = html.escape(str(report.get("inference", {})))
    body = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Сравнение фонов колоний</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}th{{background:#eee}}
code{{white-space:nowrap}}.meta{{background:#f5f5f5;padding:1rem;overflow-wrap:anywhere}}
</style></head><body>
<h1>Сравнение baseline и scratch U-Net</h1>
<div class="meta"><p>Threshold: {threshold}; inference: {inference}</p>
<p>Baseline training overlap: <strong>{baseline_overlap}</strong></p>
<p>Configuration: {config}</p><p>Source provenance: {source_provenance}</p>
<p>Held-out source IDs ({len(report.get("test_source_ids", []))}): {test_ids}</p></div>
<h2>Агрегированная пиксельная матрица ошибок</h2>
<p>Матрица задаётся как [[TN, FP], [FN, TP]]. Precision/recall рассчитаны по
сумме пикселей; если знаменатель равен нулю, метрика равна 0.</p>
<table><thead><tr><th>Модель</th><th>Фон</th><th>Sample IDs</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Precision</th><th>Recall</th><th>Confusion matrix</th></tr></thead>
<tbody>{"".join(summary_rows)}</tbody></table>
<h2>Source-level split IDs (seed {html.escape(str(report.get("seed", "unknown")))})</h2>
{"".join(split_sections)}
<h2>Метрики по каждому образцу и фону</h2>
<table><thead><tr><th>Модель</th><th>Фон</th><th>Source ID</th><th>Record ID</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Precision</th><th>Recall</th></tr></thead>
<tbody>{"".join(sample_rows)}</tbody></table>
</body></html>
"""
    output_path.write_text(body, encoding="utf-8")
    return output_path
