"""Конвейер обучения и оценки нескольких архитектур (train-compare).

Для каждой архитектуры из списка запускается `run_training`, затем `best.onnx`
оценивается на тестовых парах через `OnnxModelAlgorithm` + `run_algorithm`
(фреймворк `testing/`). Итог — сводная таблица средних метрик и HTML/JSON-отчёт.
"""

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import TrainingConfig
from .train import run_training

log = logging.getLogger(__name__)

# Метрики, попадающие в сводный отчёт (все вычисляются testing/metrics.py).
REPORT_METRICS = ("iou", "dice", "precision", "recall", "f1")


def _mean_per_variant(metrics_rows: list[dict]) -> dict[str, float]:
    """Усреднение метрик по наборам прогнозов (source и cropped)."""
    if not metrics_rows:
        return {}
    keys = [k for k in REPORT_METRICS]
    result = {}
    for key in keys:
        values = [row[key] for row in metrics_rows]
        result[key] = sum(values) / len(values)
    return result


def evaluate_onnx_on_testset(
    onnx_path: str, eval_root: Path
) -> dict[str, object]:
    """Оценка одной ONNX-модели на тестовых парах и усреднение метрик.

    Возвращает `{"mean": {...}, "n_snapshots": int}`.
    """
    from testing.dataset import TestDataset
    from testing.onnx_algorithm import OnnxModelAlgorithm
    from testing.runner import run_algorithm

    dataset = TestDataset(root=str(eval_root))
    algo = OnnxModelAlgorithm(model_path=onnx_path)
    results = run_algorithm(algo, dataset)

    rows = []
    for sample_key in results:
        for variant in ("source", "cropped"):
            if variant in results[sample_key]:
                rows.append(results[sample_key][variant])

    mean = _mean_per_variant(rows)
    mean["n_snapshots"] = len(rows)
    return mean


def run_train_compare(
    architectures: list[str],
    cfg: TrainingConfig,
    data_root: Path,
    eval_root: Path,
    progress_callback=None,
) -> dict[str, object]:
    """Обучает каждую архитектуру и сводит метрики оценки `best.onnx`.

    Возвращает словарь `{arch: {...метрики...}}` и пишет отчёты в общую
    директорию `cfg.run_dir / compare_{timestamp}/`.
    """
    compare_dir = cfg.run_dir / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for i, arch in enumerate(architectures):
        log.info("[%d/%d] Обучение архитектуры '%s'", i + 1, len(architectures), arch)
        arch_cfg = replace(cfg, model_name=arch, data_root=data_root)

        def _cb(epoch, total, train_metrics, val_metrics, elapsed):
            # Сохраняем сигнатуру колбэка ядра; имя архитектуры уже закодировано
            # в run-директории ({model}_{timestamp}), а здесь удобно логировать.
            if progress_callback:
                progress_callback(epoch, total, train_metrics, val_metrics, elapsed)

        _, _, _, run_dir = run_training(arch_cfg, progress_callback=_cb)
        best_onnx = run_dir / "checkpoints" / "best.onnx"

        if not best_onnx.is_file():
            summary[arch] = {"error": f"missing {best_onnx}"}
            continue

        metrics = evaluate_onnx_on_testset(str(best_onnx), eval_root)
        summary[arch] = metrics

    _write_reports(summary, compare_dir)
    return summary


def _write_reports(summary: dict[str, object], compare_dir: Path) -> None:
    """Сохраняет compare.json и человекочитаемый compare.html."""
    compare_dir.mkdir(parents=True, exist_ok=True)

    json_path = compare_dir / "compare.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    html_path = compare_dir / "compare.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    log.info("Compare report: %s", html_path)


def _format(value: float | object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _render_html(summary: dict[str, object]) -> str:
    columns = (*REPORT_METRICS, "n_snapshots")
    rows = "\n".join(
        "<tr><td><b>{}</b></td>{}</tr>".format(
            arch,
            "".join(f"<td>{_format(row.get(col, '—'))}</td>" for col in columns),
        )
        for arch, row in summary.items()
    )
    headers = "".join(f"<th>{col}</th>" for col in columns)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"><title>Сравнение архитектур</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; background: #1e1e2e; color: #cdd6f4; }}
  h1 {{ color: #89b4fa; }}
  table {{ border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ border: 1px solid #45475a; padding: .4rem .8rem; text-align: left; }}
  th {{ background: #313244; color: #a6e3a1; }}
</style>
</head>
<body>
<h1>Сравнение архитектур (train-compare)</h1>
<table>
<tr><th>Архитектура</th>{headers}</tr>
{rows}
</table>
</body>
</html>"""