"""Задача 3.3: инвариантный контекст снимка (SampleContext).

Покрывает: побитовую эквивалентность независимому запуску для source/cropped,
корректную инвалидацию при несовместимых параметрах, отклонение устаревшего
контекста и потокобезопасность общего контекста.
"""

import cv2
import numpy as np
import pytest

import reference_fixtures as rf
from analysis.colony_detector import ColonyDetector
from analysis.geometry import PetriInfo
from analysis.params import AnalysisParams
from analysis.sample_context import SampleContext, SampleContextCache

pytest.importorskip("cv2")


def _detect_independent(detector, fixture, params):
    """Независимый запуск без контекста (эталон по умолчанию)."""
    image = fixture.image
    h, w = image.shape[:2]
    if fixture.is_cropped:
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        petri_info = PetriInfo(*center, radius, image_shape=(h, w))
        petri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(petri_mask, center, radius, 255, -1)
    else:
        petri_mask, petri_info = detector.detect_petri_dish(image)
    return detector.detect_colonies(
        image, petri_mask, params=params, petri_info=petri_info
    )


def _params_variant(contrast=1.0, margin=10.0, sensitivity=0.5, min_size=50):
    return AnalysisParams(
        sensitivity=sensitivity,
        contrast=contrast,
        margin_percent=margin,
        min_colony_size=min_size,
        solid_fill=False,
        fill_strength=15,
    )


class TestContextMatchesIndependentRun:
    """Контекст не меняет результаты побитово (по сравнению с независимым запуском)."""

    @pytest.mark.parametrize("fixture_name", ["ordinary_source", "ordinary_cropped"])
    def test_bitwise_identical_with_context(self, fixture_name):
        fixture = rf.build_fixture(fixture_name)
        detector = ColonyDetector()
        params = _params_variant()

        expected, _ = _detect_independent(detector, fixture, params)

        context = SampleContext(image_id=fixture_name, is_cropped=fixture.is_cropped)
        image = fixture.image
        h, w = image.shape[:2]
        if fixture.is_cropped:
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            petri_info = PetriInfo(*center, radius, image_shape=(h, w))
            petri_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(petri_mask, center, radius, 255, -1)
            context.set_geometry(petri_mask, petri_info)
        else:
            petri_mask, petri_info = detector.detect_petri_dish(image)
            context.set_geometry(petri_mask, petri_info)

        first, _ = detector.detect_colonies(
            image, petri_mask, params=params, petri_info=petri_info, context=context
        )
        assert np.array_equal(first, expected), (
            "прогон с контекстом разошёлся с независимым запуском"
        )

    def test_repeated_runs_with_context_are_bitwise_identical(self):
        fixture = rf.build_fixture("ordinary_source")
        detector = ColonyDetector()
        context = SampleContext(image_id="repeat", is_cropped=False)
        petri_mask, petri_info = detector.detect_petri_dish(fixture.image)
        context.set_geometry(petri_mask, petri_info)

        params = _params_variant()
        first, _ = detector.detect_colonies(
            fixture.image,
            petri_mask,
            params=params,
            petri_info=petri_info,
            context=context,
        )
        second, _ = detector.detect_colonies(
            fixture.image,
            petri_mask,
            params=params,
            petri_info=petri_info,
            context=context,
        )
        assert np.array_equal(first, second)


class TestContextInvalidation:
    """Инвалидация: несовместимые параметры отклоняют устаревший кеш."""

    def test_contrast_change_invalidates_preprocess(self):
        fixture = rf.build_fixture("boundary_source")
        detector = ColonyDetector()
        context = SampleContext(image_id="contrast", is_cropped=False)
        petri_mask, petri_info = detector.detect_petri_dish(fixture.image)
        context.set_geometry(petri_mask, petri_info)

        params_low = _params_variant(contrast=1.0)
        params_high = _params_variant(contrast=2.5)

        expected_low, _ = _detect_independent(detector, fixture, params_low)
        expected_high, _ = _detect_independent(detector, fixture, params_high)

        first, _ = detector.detect_colonies(
            fixture.image,
            petri_mask,
            params=params_low,
            petri_info=petri_info,
            context=context,
        )
        assert np.array_equal(first, expected_low)
        # Контекст с contrast=1.0 отклонён для contrast=2.5.
        assert not context.is_compatible_with(2.5, 5)
        second, _ = detector.detect_colonies(
            fixture.image,
            petri_mask,
            params=params_high,
            petri_info=petri_info,
            context=context,
        )
        assert np.array_equal(second, expected_high), (
            "несовместимый контекст изменил результат"
        )
        # После инвалидации preprocess пересчитан под новый contrast.
        assert context.is_compatible_with(2.5, 5)
        assert not context.is_compatible_with(1.0, 5)

    def test_margin_change_does_not_reuse_incompatible_masked_layer(self):
        """`masked_diff` зависит от margin — он пересчитывается всегда."""
        fixture = rf.build_fixture("boundary_source")
        detector = ColonyDetector()
        context = SampleContext(image_id="margin", is_cropped=False)
        petri_mask, petri_info = detector.detect_petri_dish(fixture.image)
        context.set_geometry(petri_mask, petri_info)

        params_a = _params_variant(margin=5.0)
        params_b = _params_variant(margin=25.0)

        expected_a, _ = _detect_independent(detector, fixture, params_a)
        expected_b, _ = _detect_independent(detector, fixture, params_b)

        result_a, _ = detector.detect_colonies(
            fixture.image,
            petri_mask,
            params=params_a,
            petri_info=petri_info,
            context=context,
        )
        result_b, _ = detector.detect_colonies(
            fixture.image,
            petri_mask,
            params=params_b,
            petri_info=petri_info,
            context=context,
        )
        assert np.array_equal(result_a, expected_a)
        assert np.array_equal(result_b, expected_b)

    def test_geometry_is_reused_for_source_context(self):
        detector = ColonyDetector()
        fixture = rf.build_fixture("ordinary_source")
        context = SampleContext(image_id="geo", is_cropped=False)
        petri_mask, petri_info = detector.detect_petri_dish(fixture.image)

        assert not context.has_geometry
        context.set_geometry(petri_mask, petri_info)
        assert context.has_geometry
        assert context.get_geometry() is not None

        # Повторный set_geometry не перезаписывает (write-once).
        context.set_geometry(None, None)
        assert context.get_geometry()[0] is petri_mask

    def test_cropped_and_source_contexts_are_separate(self):
        cache = SampleContextCache()
        source = cache.get_or_create("img", is_cropped=False)
        cropped = cache.get_or_create("img", is_cropped=True)
        assert source is not cropped
        assert len(cache) == 2
        assert cache.get("img", is_cropped=False) is source


class TestContextThreadSafety:
    """Общий контекст потокобезопасен (параллельные алгоритмы одного снимка)."""

    def test_parallel_algorithms_do_not_corrupt_shared_context(self):
        import threading

        fixture = rf.build_fixture("ordinary_source")
        detector = ColonyDetector()
        petri_mask, petri_info = detector.detect_petri_dish(fixture.image)
        context = SampleContext(image_id="shared", is_cropped=False)
        context.set_geometry(petri_mask, petri_info)

        params_a = _params_variant(contrast=1.0)
        expected, _ = _detect_independent(detector, fixture, params_a)

        errors = []

        def run():
            try:
                result, _ = detector.detect_colonies(
                    fixture.image,
                    petri_mask,
                    params=params_a,
                    petri_info=petri_info,
                    context=context,
                )
                assert np.array_equal(result, expected), (
                    "результат с общим контекстом изменился под нагрузкой"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors


class TestContextCacheLRU:
    def test_lru_eviction(self):
        cache = SampleContextCache(maxsize=3)
        for i in range(5):
            cache.get_or_create(f"img-{i}", is_cropped=False)
        assert len(cache) == 3
        assert cache.get("img-0", is_cropped=False) is None
        assert cache.get("img-4", is_cropped=False) is not None

    def test_lru_touch_prevents_eviction(self):
        cache = SampleContextCache(maxsize=2)
        first = cache.get_or_create("a", is_cropped=False)
        cache.get_or_create("b", is_cropped=False)
        cache.get_or_create("a", is_cropped=False)  # touch
        cache.get_or_create("c", is_cropped=False)  # evict "b"
        assert cache.get("a", is_cropped=False) is first
        assert cache.get("b", is_cropped=False) is None
