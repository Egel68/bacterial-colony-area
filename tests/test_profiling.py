"""Задача 3.1: профилирование стадий конвейера обработки изображений.

Покрывает: раздельные замеры стадий (поиск чашки, предобработка, порог/
морфология, фильтрация компонентов) на обычных и крупных (20–35 MP) кадрах
и неизменность эталонных масок при активном профилировании.
"""

import numpy as np
import pytest

import reference_fixtures as rf
from analysis.profiling import (
    profile_algorithm_run,
    profile_stages,
    measure_stage,
)

pytest.importorskip("cv2")

EXPECTED_STAGES = ("dish_search", "classic_preprocess", "threshold_morphology")


def test_profile_reports_per_stage_wall_time():
    """Профиль содержит отдельное время каждой стадии конвейера."""
    fixture = rf.build_fixture("ordinary_source")
    stages = profile_algorithm_run(
        fixture.image, is_cropped=False, algorithm_name="ClassicDefault"
    )

    for name in EXPECTED_STAGES:
        assert name in stages, f"стадия {name} не попала в профиль"
        assert stages[name]["total"] >= 0.0
        assert stages[name]["calls"] >= 1.0
    assert "component_filtering" in stages
    assert stages["dish_search"]["total"] > 0.0


def test_profile_captures_component_filtering():
    fixture = rf.build_fixture("boundary_cropped")
    stages = profile_algorithm_run(
        fixture.image, is_cropped=True, algorithm_name="ClassicDefault"
    )
    assert "component_filtering" in stages
    assert stages["component_filtering"]["calls"] >= 1.0


def test_profiling_does_not_change_reference_masks():
    """Активное профилирование не меняет эталонные маски побитово.

    Требование задачи 3.1: вывод идентифицирует время стадий, но не меняет
    reference-маски. Сверяем с записанным baseline из tests/baseline/.
    """
    record = rf.load_baseline()
    subset = {
        "fixtures": {
            name: record["fixtures"][name]
            for name in ("ordinary_source", "boundary_cropped")
        },
        "entries": {
            name: {algo: record["entries"][name][algo] for algo in ("ClassicDefault",)}
            for name in ("ordinary_source", "boundary_cropped")
        },
    }
    with profile_stages() as profiler:
        entries, masks = rf.capture_baseline(
            fixture_names=("ordinary_source", "boundary_cropped"),
            algorithms=("ClassicDefault",),
        )
    problems = rf.compare_with_baseline(entries, masks, subset)
    assert not problems, "профилирование изменило маски/метрики: " + "; ".join(problems)
    assert profiler.totals(), "профилировщик не собрал ни одной стадии"


def test_no_profiler_means_no_overhead_and_same_masks():
    """Без активного профилировщика результаты не изменяются."""
    fixture = rf.build_fixture("ordinary_source")
    from testing.registry import get_algorithm

    algo = get_algorithm("ClassicDefault")
    mask_direct = algo.detect(fixture.image, is_cropped=False)
    entries, masks = rf.capture_baseline(
        fixture_names=("ordinary_source",), algorithms=("ClassicDefault",)
    )
    stored = masks[("ordinary_source", "ClassicDefault")]
    normalized = ((mask_direct > 0).astype(np.uint8)) * 255
    assert np.array_equal(normalized, stored)


@pytest.mark.slow
def test_profile_on_large_image_reports_stages():
    """Крупный кадр (20 MP): стадии профилируются и суммарное время > 0."""
    fixture = rf.build_fixture("largest_source")
    assert fixture.image.shape[0] * fixture.image.shape[1] >= 20_000_000

    stages = profile_algorithm_run(
        fixture.image, is_cropped=False, algorithm_name="ClassicDefault"
    )
    total = sum(info["total"] for info in stages.values())
    assert total > 0.0, "профиль крупного кадра пуст"
    for name in EXPECTED_STAGES:
        assert name in stages, f"стадия {name} не профилирована на крупном кадре"


def test_nested_measurement_stacks_and_restores():
    """measure_stage корректно работает при вложенности и после выхода."""
    with profile_stages() as outer:
        with measure_stage("a"):
            with measure_stage("b"):
                pass
        with measure_stage("a"):
            pass
    assert set(outer.stages) == {"a", "b"}
    assert outer.stages["a"] and len(outer.stages["a"]) == 2

    # После завершения контекста новые замеры не попадают в собранный профиль.
    with measure_stage("c"):
        pass
    assert "c" not in outer.stages


def test_as_dict_shape():
    with profile_stages() as profiler:
        with measure_stage("only"):
            pass
    summary = profiler.as_dict()
    assert set(summary) == {"only"}
    assert set(summary["only"]) == {"total", "calls", "mean"}
    assert summary["only"]["calls"] == 1.0
