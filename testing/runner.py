import logging
from typing import Dict, List

from .dataset import TestDataset
from .interface import BaseDetectionAlgorithm
from .metrics import compute_segmentation_metrics
from .registry import list_algorithms, get_algorithm, get_algorithm_descriptions

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
            result[f"std_{key}"] = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        else:
            result[f"std_{key}"] = 0.0
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
        source_metrics = compute_segmentation_metrics(source_mask_pred, sample.source_mask)
        results[sample_key]["source"] = source_metrics

        if sample.cropped_image is not None and sample.cropped_mask is not None:
            cropped_mask_pred = algo.detect(sample.cropped_image, is_cropped=True)
            cropped_metrics = compute_segmentation_metrics(cropped_mask_pred, sample.cropped_mask)
            results[sample_key]["cropped"] = cropped_metrics

    return results


def run_all(dataset: TestDataset) -> AllResults:
    all_results: AllResults = {}

    for algo_name in list_algorithms():
        algo = get_algorithm(algo_name)
        log.info("Running: %s", algo_name)
        results = run_algorithm(algo, dataset)
        all_results[algo_name] = results

    return all_results


def compute_summary(all_results: AllResults) -> List[Dict]:
    desc_map = dict(get_algorithm_descriptions())
    summary = []
    for algo_name, algo_results in all_results.items():
        all_metrics = []
        for sample_key, variants in algo_results.items():
            for variant_key, metrics in variants.items():
                all_metrics.append(metrics)

        means = _mean_metrics(all_metrics)
        entry = {
            "name": algo_name,
            "description": desc_map.get(algo_name, ""),
            "num_images": len(algo_results),
        }
        entry.update(means)
        summary.append(entry)
    return summary
