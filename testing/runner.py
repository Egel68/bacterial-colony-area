import logging
import threading
from typing import Dict, List, Optional

import numpy as np

from .dataset import TestDataset
from .interface import BaseDetectionAlgorithm
from .metrics import compute_segmentation_metrics
from .pipeline_observer import (
    PHASE_COMPARISON,
    PHASE_RESULT_PREPARATION,
    PipelineObserver,
)
from .registry import list_algorithms, get_algorithm_descriptions
from .scheduler import PipelineDiagnostics, execute_pipeline
from .telemetry import TelemetryCollector
from .statistics import (
    compute_descriptive,
    wilcoxon_signed_rank,
    compute_winner_fractions,
)

log = logging.getLogger(__name__)

SampleResult = Dict[str, float]
AlgorithmResults = Dict[str, Dict[str, Dict[str, float]]]
AllResults = Dict[str, AlgorithmResults]


def _mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0] if k not in ("tp", "fp", "fn", "tn")]
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        mean = sum(values) / len(values)
        result[f"mean_{key}"] = mean
        if len(values) > 1:
            result[f"std_{key}"] = (
                sum((v - mean) ** 2 for v in values) / len(values)
            ) ** 0.5
        else:
            result[f"std_{key}"] = 0.0
    return result


def compute_detailed_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Расширенная агрегация: mean+std + median, Q1, Q3, P5, P95, min, max."""
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0] if k not in ("tp", "fp", "fn", "tn")]
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        # mean + std (existing)
        mean = sum(values) / len(values)
        result[f"mean_{key}"] = mean
        if len(values) > 1:
            result[f"std_{key}"] = (
                sum((v - mean) ** 2 for v in values) / len(values)
            ) ** 0.5
        else:
            result[f"std_{key}"] = 0.0
        # descriptive statistics
        desc = compute_descriptive(values)
        result[f"median_{key}"] = desc["median"]
        result[f"q1_{key}"] = desc["q1"]
        result[f"q3_{key}"] = desc["q3"]
        result[f"p5_{key}"] = desc["p5"]
        result[f"p95_{key}"] = desc["p95"]
        result[f"min_{key}"] = desc["min"]
        result[f"max_{key}"] = desc["max"]
    return result


def _run_algorithm_seq(
    algo: BaseDetectionAlgorithm,
    dataset: TestDataset,
) -> AlgorithmResults:
    """Sequential run for one algorithm (kept for backward compat / compare_algorithms)."""
    results: AlgorithmResults = {}
    for sample in dataset:
        sample_key = sample.name
        results[sample_key] = {}
        source_mask_pred = algo.detect(sample.source_image, is_cropped=False)
        source_metrics = compute_segmentation_metrics(
            source_mask_pred, sample.source_mask
        )
        results[sample_key]["source"] = source_metrics
        if sample.cropped_image is not None and sample.cropped_mask is not None:
            cropped_mask_pred = algo.detect(sample.cropped_image, is_cropped=True)
            cropped_metrics = compute_segmentation_metrics(
                cropped_mask_pred, sample.cropped_mask
            )
            results[sample_key]["cropped"] = cropped_metrics
    return results


def run_algorithm(
    algo: BaseDetectionAlgorithm,
    dataset: TestDataset,
) -> AlgorithmResults:
    """Run one algorithm sequentially for compatibility and comparisons."""
    return _run_algorithm_seq(algo, dataset)


def run_all(
    dataset: TestDataset,
    algorithms: Optional[List[str]] = None,
    workers: Optional[int] = None,
    use_cache: bool = False,
    batch_size: int = 8,
    memory_budget: int | None = None,
    telemetry: TelemetryCollector | None = None,
    diagnostics: PipelineDiagnostics | None = None,
) -> AllResults:
    """Прогоняет выбранные алгоритмы по bounded in-memory batch-ам."""
    if algorithms is None:
        names = list_algorithms()
    else:
        names = algorithms
    if not names:
        return {}
    log.info(
        "Running %d algorithm(s) with workers=%s, batch_size=%d",
        len(names),
        workers or "auto",
        batch_size,
    )
    return execute_pipeline(
        dataset,
        names,
        workers=workers,
        batch_size=batch_size,
        memory_budget=memory_budget,
        use_cache=use_cache,
        telemetry=telemetry,
        diagnostics=diagnostics,
    )


def build_comparison(
    results_a: Dict[str, Dict[str, Dict[str, float]]],
    results_b: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Строит парное сравнение A vs B по уже вычисленным метрикам.

    Возвращает вложенный словарь:
      comparison[metric][sample_key][variant] = "a" | "b" | "tie"
    """
    comparison: Dict[str, Dict[str, Dict[str, str]]] = {}
    metric_names = ["iou", "dice", "f1", "precision", "recall", "accuracy"]

    for sample_key in results_a:
        variants = set(results_a[sample_key]) & set(results_b.get(sample_key, {}))
        for variant in variants:
            metrics_a = results_a[sample_key][variant]
            metrics_b = results_b[sample_key][variant]
            for metric in metric_names:
                va = metrics_a.get(metric, 0.0)
                vb = metrics_b.get(metric, 0.0)
                comparison.setdefault(metric, {}).setdefault(sample_key, {})[
                    variant
                ] = "a" if va > vb else ("b" if vb > va else "tie")

    return comparison


def run_all_with_comparison(
    dataset: TestDataset,
    algorithms: Optional[List[str]] = None,
    compare_pair: Optional[tuple[str, str]] = None,
    workers: Optional[int] = None,
    use_cache: bool = False,
    batch_size: int = 8,
    memory_budget: int | None = None,
    telemetry: TelemetryCollector | None = None,
    diagnostics: PipelineDiagnostics | None = None,
    observer: Optional[PipelineObserver] = None,
    cancel_event: Optional["threading.Event"] = None,
) -> tuple[AllResults, Optional[Dict[str, Dict[str, Dict[str, str]]]]]:
    """Прогон + парное сравнение без повторных детекций (задача 3.4).

    Каждая требуемая пара «алгоритм × sample × variant» вычисляется не более
    одного раза: сравнение строится из результатов основного прогона. Если
    сравниваемый алгоритм не входит в отображаемое подмножество, он
    вычисляется дополнительно один раз и НЕ попадает в `all_results`.

    `observer` (задача 7.1) получает фазы конвейера (dataset_scan,
    read_decode, algorithm, comparison, result_preparation) и контекстные
    ошибки отдельных задач.
    """
    if algorithms is None:
        names = list_algorithms()
    else:
        names = list(algorithms)

    run_names = list(names)
    missing_compare: list[str] = []
    if compare_pair:
        for compare_name in compare_pair:
            if compare_name not in run_names:
                run_names.append(compare_name)
                missing_compare.append(compare_name)

    if not run_names:
        return {}, None

    log.info(
        "Running %d algorithm(s) with workers=%s, batch_size=%d "
        "(compare=%s, extra for comparison=%s)",
        len(run_names),
        workers or "auto",
        batch_size,
        compare_pair or "-",
        missing_compare or "-",
    )
    full_results = execute_pipeline(
        dataset,
        run_names,
        workers=workers,
        batch_size=batch_size,
        memory_budget=memory_budget,
        use_cache=use_cache,
        telemetry=telemetry,
        diagnostics=diagnostics,
        observer=observer,
        cancel_event=cancel_event,
    )

    # Отображаемый набор алгоритмов сохраняется: лишние результаты скрыты.
    all_results: AllResults = {name: full_results.get(name, {}) for name in names}

    comparison = None
    if compare_pair:
        name_a, name_b = compare_pair
        if observer is not None:
            observer.on_phase(PHASE_COMPARISON)
        comparison = build_comparison(
            full_results.get(name_a, {}), full_results.get(name_b, {})
        )
    if observer is not None:
        observer.on_phase(PHASE_RESULT_PREPARATION)
    return all_results, comparison


def compare_algorithms(
    dataset: TestDataset, name_a: str, name_b: str
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Парное сравнение алгоритмов A и B по снимкам (отдельный прогон).

    Возвращает вложенный словарь:
      comparison[metric][sample_key][variant] = "a" | "b" | "tie"

    Для сравнения внутри общего прогона используйте
    `run_all_with_comparison` — он не повторяет детекции.
    """
    compared = execute_pipeline(dataset, [name_a, name_b], workers=1)
    return build_comparison(compared.get(name_a, {}), compared.get(name_b, {}))


def compute_summary(all_results: AllResults) -> List[Dict]:
    desc_map = dict(get_algorithm_descriptions())
    summary = []
    for algo_name, algo_results in all_results.items():
        all_metrics = []
        for sample_key, variants in algo_results.items():
            for variant_key, metrics in variants.items():
                all_metrics.append(metrics)

        entry = {
            "name": algo_name,
            "description": desc_map.get(algo_name, ""),
            "num_images": len(algo_results),
        }
        entry.update(compute_detailed_metrics(all_metrics))
        summary.append(entry)
    return summary


def compute_wilcoxon_table(all_results: AllResults) -> Dict:
    """Матрица p-value парного Wilcoxon signed-rank test для каждой метрики.

    Возвращает { metric: { (algo_a, algo_b): { p_value, interpretation } } }
    где interpretation — 'significant' (p<0.05) или 'not_significant'.
    """
    algo_names = list(all_results.keys())
    metric_names = ["iou", "dice", "f1", "precision", "recall", "accuracy"]
    table: Dict = {}
    for metric in metric_names:
        table[metric] = {}
        for i, name_a in enumerate(algo_names):
            for name_b in algo_names[i + 1 :]:
                pair_key = f"{name_a} vs {name_b}"
                values_a = []
                values_b = []
                for sample in all_results[name_a]:
                    for variant in all_results[name_a][sample]:
                        if (
                            sample in all_results[name_b]
                            and variant in all_results[name_b][sample]
                        ):
                            values_a.append(
                                all_results[name_a][sample][variant].get(metric, 0.0)
                            )
                            values_b.append(
                                all_results[name_b][sample][variant].get(metric, 0.0)
                            )
                result = wilcoxon_signed_rank(values_a, values_b)
                if result["message"] == "identical":
                    table[metric][pair_key] = {
                        "p_value": None,
                        "interpretation": "identical",
                    }
                else:
                    p = result["p_value"]
                    table[metric][pair_key] = {
                        "p_value": round(p, 6),
                        "interpretation": "significant"
                        if p < 0.05
                        else "not_significant",
                    }
    return table


def compute_outlier_table(all_results: AllResults) -> List[Dict]:
    """Список выбросов (|z| > 3) по каждому алгоритму и метрике.

    Возвращает список словарей с ключами:
      algorithm, metric, sample, variant, value, z_score
    """
    metric_names = ["iou", "dice", "f1", "precision", "recall", "accuracy"]
    outliers = []
    for algo_name, algo_results in all_results.items():
        for metric in metric_names:
            samples = []
            vals = []
            variants = []
            for sample_key, variants_dict in algo_results.items():
                for variant_key, m in variants_dict.items():
                    samples.append(sample_key)
                    variants.append(variant_key)
                    vals.append(m.get(metric, 0.0))
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr))
            for idx, val in enumerate(vals):
                if std_v == 0:
                    continue
                z = (val - mean_v) / std_v
                if abs(z) > 3.0:
                    outliers.append(
                        {
                            "algorithm": algo_name,
                            "metric": metric,
                            "sample": samples[idx],
                            "variant": variants[idx],
                            "value": round(val, 6),
                            "z_score": round(z, 3),
                        }
                    )
    return outliers


def compute_all_winners(all_results: AllResults) -> Dict:
    """Доля побед каждого алгоритма по каждой метрике.

    Возвращает { metric: { (a, b): { a_wins, b_wins, ties } } }
    """
    algo_names = list(all_results.keys())
    metric_names = ["iou", "dice", "f1", "precision", "recall", "accuracy"]
    winners: Dict = {}
    for metric in metric_names:
        winners[metric] = {}
        for i, name_a in enumerate(algo_names):
            for name_b in algo_names[i + 1 :]:
                pair_key = f"{name_a} vs {name_b}"
                metrics_a = []
                metrics_b = []
                for sample in all_results[name_a]:
                    for variant in all_results[name_a][sample]:
                        if (
                            sample in all_results[name_b]
                            and variant in all_results[name_b][sample]
                        ):
                            metrics_a.append(all_results[name_a][sample][variant])
                            metrics_b.append(all_results[name_b][sample][variant])
                winners[metric][pair_key] = compute_winner_fractions(
                    metrics_a, metrics_b, metric
                )
    return winners
