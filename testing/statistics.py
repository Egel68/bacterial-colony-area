import math
from typing import Dict, List, Tuple

import numpy as np


def compute_descriptive(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def detect_outliers(
    values: List[float],
    z_thresh: float = 3.0,
) -> List[Dict]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0.0:
        return []
    outliers = []
    for val in values:
        z = (val - mean) / std
        if abs(z) > z_thresh:
            outliers.append({"value": val, "z_score": round(z, 3)})
    return outliers


def wilcoxon_signed_rank(
    a: List[float], b: List[float]
) -> Dict:
    if len(a) != len(b):
        raise ValueError("Arrays must have same length")
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    non_zero = diffs[diffs != 0]
    n = len(non_zero)
    if n == 0:
        return {"p_value": None, "message": "identical", "statistic": 0.0}

    ranks = np.argsort(np.abs(non_zero)) + 1
    signed_ranks = ranks * np.sign(non_zero)
    w = float(np.sum(signed_ranks[signed_ranks > 0]))

    if n <= 20:
        # Normal approximation with continuity correction
        pass  # fall through to approximation below
    # Normal approximation
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    # Handle ties correction
    tie_counts = np.unique(np.abs(non_zero), return_counts=True)[1]
    if len(tie_counts) > 0 and np.any(tie_counts > 1):
        tie_correction = sum(t**3 - t for t in tie_counts[tie_counts > 1]) / 48
        sigma = math.sqrt(
            n * (n + 1) * (2 * n + 1) / 24 - tie_correction
        )
    if sigma == 0:
        z = 0.0
    else:
        z = (w - mu) / sigma
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {"p_value": p, "message": "ok", "statistic": w}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_winner_fractions(
    metrics_a: List[Dict],
    metrics_b: List[Dict],
    metric_key: str,
) -> Dict[str, float]:
    n = len(metrics_a)
    if n == 0:
        return {"a_wins": 0.0, "b_wins": 0.0, "ties": 0.0}
    a_wins = 0
    b_wins = 0
    ties = 0
    for ma, mb in zip(metrics_a, metrics_b):
        va = ma.get(metric_key, 0.0)
        vb = mb.get(metric_key, 0.0)
        if abs(va - vb) < 1e-9:
            ties += 1
        elif va > vb:
            a_wins += 1
        else:
            b_wins += 1
    return {
        "a_wins": a_wins / n,
        "b_wins": b_wins / n,
        "ties": ties / n,
    }