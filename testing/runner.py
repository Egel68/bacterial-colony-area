import logging
from typing import Dict, List, Optional

import numpy as np

from .dataset import TestDataset
from .interface import BaseDetectionAlgorithm
from .metrics import compute_segmentation_metrics
from .registry import list_algorithms, get_algorithm, get_algorithm_descriptions
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


def run_algorithm(
    algo: BaseDetectionAlgorithm,
    dataset: TestDataset,
) -> AlgorithmResults:
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


def run_all(dataset: TestDataset, algorithms: Optional[List[str]] = None) -> AllResults:
    """Прогоняет весь реестр (или указанное подмножество) по датасету."""
    all_results: AllResults = {}

    if algorithms is None:
        names = list_algorithms()
    else:
        names = algorithms

    for algo_name in names:
        algo = get_algorithm(algo_name)
        log.info("Running: %s", algo_name)
        results = run_algorithm(algo, dataset)
        all_results[algo_name] = results

    return all_results


def compare_algorithms(
    dataset: TestDataset, name_a: str, name_b: str
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Парное сравнение алгоритмов A и B по снимкам.

    Возвращает вложенный словарь:
      comparison[metric][sample_key][variant] = "a" | "b" | "tie"
    """
    algo_a = get_algorithm(name_a)
    algo_b = get_algorithm(name_b)

    results_a = run_algorithm(algo_a, dataset)
    results_b = run_algorithm(algo_b, dataset)

    comparison: Dict[str, Dict[str, Dict[str, str]]] = {}
    metric_names = ["iou", "dice", "f1", "precision", "recall", "accuracy"]

    for sample_key in results_a:
        variants = set(results_a[sample_key]) & set(results_b[sample_key])
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
                        "interpretation": "significant" if p < 0.05 else "not_significant",
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
