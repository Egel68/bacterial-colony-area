"""Детерминированное разбиение исходных изображений по выборкам."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
SUBSETS = ("train", "val", "test")


def stratified_group_split(
    group_labels: Mapping[str, object],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, str]:
    """Распределить стабильные ID исходных изображений по train/val/test.

    Каждая группа относится ровно к одному label/category. Сортировка групп и
    фиксированный seed не дают порядку входной COCO-коллекции менять результат.
    Для страт от трёх элементов сохраняется хотя бы один элемент в каждом subset.
    """
    if not group_labels:
        raise ValueError("Cannot split an empty dataset")
    if len(group_labels) < len(SUBSETS):
        raise ValueError("Dataset must contain at least three groups")
    if len(ratios) != len(SUBSETS):
        raise ValueError("ratios must contain train, val and test proportions")
    if any(not np.isfinite(value) or value <= 0 for value in ratios) or not np.isclose(
        sum(ratios), 1.0
    ):
        raise ValueError("ratios must be positive and sum to 1")

    strata: dict[str, list[str]] = defaultdict(list)
    for group_id, label in group_labels.items():
        strata[str(label)].append(str(group_id))
    if sum(len(group_ids) for group_ids in strata.values()) != len(
        set(map(str, group_labels))
    ):
        raise ValueError("group IDs must be unique after string conversion")

    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for label in sorted(strata):
        group_ids = sorted(strata[label])
        shuffled = [group_ids[index] for index in rng.permutation(len(group_ids))]
        counts = _stratum_counts(len(shuffled), ratios)
        start = 0
        for subset, count in zip(SUBSETS, counts, strict=True):
            for group_id in shuffled[start : start + count]:
                assignments[group_id] = subset
            start += count

    # Rounding across many small strata can otherwise leave a global subset empty.
    for missing_subset in SUBSETS:
        if missing_subset in assignments.values():
            continue
        donors = [
            subset
            for subset in SUBSETS
            if sum(value == subset for value in assignments.values()) > 1
        ]
        if not donors:
            raise ValueError("Not enough groups to populate train, val and test")
        donor = max(
            donors, key=lambda subset: sum(v == subset for v in assignments.values())
        )
        group_id = min(key for key, value in assignments.items() if value == donor)
        assignments[group_id] = missing_subset

    return dict(sorted(assignments.items()))


def _stratum_counts(
    size: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Посчитать целые доли, оставив train непустым и деля редкие классы."""
    if size == 1:
        return (1, 0, 0)
    if size == 2:
        return (1, 0, 1)
    raw = np.asarray(ratios, dtype=np.float64) * size
    counts = np.floor(raw).astype(int)
    remainder = size - int(counts.sum())
    order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1

    if size >= 3:
        for index in range(3):
            if counts[index] > 0:
                continue
            donor = max(range(3), key=lambda candidate: counts[candidate])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[index] += 1
    return tuple(int(value) for value in counts)
