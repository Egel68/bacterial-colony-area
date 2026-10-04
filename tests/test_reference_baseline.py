"""Задача 1.2: baseline бинарных масок и метрик эталонных фикстур.

Проверяет:
- детерминированную генерацию эталонных фикстур (ordinary/boundary/largest,
  варианты source/cropped);
- повторные прогоны baseline дают побитово идентичные маски и метрики
  в допуске `1e-6`;
- соответствие текущего кода зафиксированному baseline в `tests/baseline/`
  (страховка для задач 3.2–3.3: оптимизации не должны менять маски/метрики).
"""

import json

import numpy as np
import pytest

import reference_fixtures as rf

pytestmark = pytest.mark.slow

FIXTURE_PARAMS = [pytest.param(n, id=n) for n in rf.FIXTURE_NAMES]


@pytest.fixture(scope="session")
def baseline_runs():
    """Два независимых прогона baseline по всем фикстурам (общий кэш сессии)."""
    first_entries, first_masks = rf.capture_baseline()
    second_entries, second_masks = rf.capture_baseline()
    return (first_entries, first_masks), (second_entries, second_masks)


@pytest.mark.parametrize("name", FIXTURE_PARAMS)
def test_fixture_generation_is_deterministic(name):
    a = rf.build_fixture(name, use_cache=False)
    b = rf.build_fixture(name, use_cache=False)
    assert np.array_equal(a.image, b.image), f"{name}: изображение недетерминировано"
    assert np.array_equal(a.ground_truth, b.ground_truth), (
        f"{name}: GT-маска недетерминирована"
    )
    assert a.variant in ("source", "cropped")


@pytest.mark.parametrize("name", FIXTURE_PARAMS)
def test_repeated_baseline_runs_identical(name, baseline_runs):
    (first_entries, first_masks), (second_entries, second_masks) = baseline_runs
    for algo_name in rf.ALGORITHMS:
        first = first_entries[name][algo_name]
        second = second_entries[name][algo_name]
        assert first["mask_sha256"] == second["mask_sha256"], (
            f"{name}/{algo_name}: повторный прогон изменил маску"
        )
        assert np.array_equal(
            first_masks[(name, algo_name)], second_masks[(name, algo_name)]
        ), f"{name}/{algo_name}: повторный прогон изменил пиксели маски"
        for metric_key, value in first["metrics"].items():
            delta = abs(value - second["metrics"][metric_key])
            assert delta <= 1e-6, (
                f"{name}/{algo_name}: метрика {metric_key} "
                f"дрейфует между прогонами (|Δ|={delta:.3g})"
            )


@pytest.mark.parametrize("name", FIXTURE_PARAMS)
def test_committed_baseline_matches(name, baseline_runs):
    (entries, masks), _ = baseline_runs
    record = rf.load_baseline()
    subset = {
        "fixtures": {name: record["fixtures"][name]},
        "entries": {name: record["entries"][name]},
    }
    problems = rf.compare_with_baseline(entries, masks, subset)
    assert not problems, (
        "Текущий код расходится с tests/baseline/reference_masks.json:\n  - "
        + "\n  - ".join(problems)
        + "\nЕсли изменение масок легитимно (например, смена версии OpenCV), "
        "перезапишите baseline:\n"
        "  UV_PROJECT_ENVIRONMENT=.venv-dev uv run python "
        "tests/reference_fixtures.py --write"
    )


def test_committed_record_is_well_formed():
    record = rf.load_baseline()
    assert record["schema_version"] == rf.SCHEMA_VERSION
    assert record["algorithms"] == list(rf.ALGORITHMS)
    assert set(record["fixtures"]) == set(rf.FIXTURE_NAMES)
    assert set(record["entries"]) == set(rf.FIXTURE_NAMES)

    categories = {meta["category"] for meta in record["fixtures"].values()}
    variants = {meta["variant"] for meta in record["fixtures"].values()}
    assert {"ordinary", "boundary", "largest"} <= categories
    assert {"source", "cropped"} <= variants

    ratio_metrics = ("iou", "dice", "f1", "precision", "recall", "accuracy")
    for name, per_algo in record["entries"].items():
        assert set(per_algo) == set(rf.ALGORITHMS)
        for algo_name, entry in per_algo.items():
            assert entry["mask_sha256"] and entry["mask_file"].endswith(".png")
            for metric_key in ratio_metrics:
                value = entry["metrics"][metric_key]
                assert 0.0 <= value <= 1.0, (
                    f"{name}/{algo_name}: метрика {metric_key} вне [0, 1]"
                )
            for count_key in ("tp", "fp", "fn", "tn"):
                assert entry["metrics"][count_key] >= 0, (
                    f"{name}/{algo_name}: счётчик {count_key} отрицательный"
                )


def test_fixture_ground_truth_is_binary_mask():
    fixture = rf.build_fixture("ordinary_source")
    assert fixture.ground_truth.dtype == np.uint8
    assert set(np.unique(fixture.ground_truth)) <= {0, 255}
    assert fixture.image.dtype == np.uint8


def test_cli_write_and_check_roundtrip(tmp_path, monkeypatch):
    """CLI захвата/проверки baseline работает на подмножестве фикстур."""
    subset = ("ordinary_source", "ordinary_cropped")
    monkeypatch.setattr(rf, "FIXTURE_NAMES", subset)
    monkeypatch.setattr(rf, "ALGORITHMS", ("ClassicDefault",))
    monkeypatch.setattr(rf, "BASELINE_JSON", tmp_path / "reference_masks.json")
    monkeypatch.setattr(rf, "BASELINE_DIR", tmp_path)
    monkeypatch.setattr(rf, "MASKS_DIR", tmp_path / "masks")

    assert rf.main(["--write"]) == 0
    assert (tmp_path / "reference_masks.json").is_file()
    assert (tmp_path / "masks").is_dir()
    assert rf.main(["--check"]) == 0

    # Порча baseline должна обнаруживаться.
    record = json.loads((tmp_path / "reference_masks.json").read_text("utf-8"))
    record["entries"]["ordinary_source"]["ClassicDefault"]["metrics"]["iou"] += 0.05
    (tmp_path / "reference_masks.json").write_text(json.dumps(record), encoding="utf-8")
    assert rf.main(["--check"]) == 1
