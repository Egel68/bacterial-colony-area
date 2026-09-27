import pytest

from testing.interface import BaseDetectionAlgorithm
from testing.registry import (
    get_algorithm,
    list_algorithms,
    get_algorithm_descriptions,
    register_algorithm_instance,
)


class _FakeInstance(BaseDetectionAlgorithm):
    name = "FakeInstance"
    description = "Фиктивный экземпляр алгоритма"
    detects = 0

    def detect(self, image, is_cropped=False):
        _FakeInstance.detects += 1
        import numpy as np

        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


def test_list_contains_expected():
    names = list_algorithms()
    assert "ClassicDefault" in names
    assert "ClassicHighSensitivity" in names
    assert "ClassicSolidFill" in names
    assert "ClassicLowSensitivity" in names


def test_get_algorithm_returns_instance():
    algo = get_algorithm("ClassicDefault")
    assert algo.name == "Classic (default)"
    assert hasattr(algo, "detect")


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_algorithm("NonExistent")


def test_get_algorithm_twice_different_instances():
    a1 = get_algorithm("ClassicDefault")
    a2 = get_algorithm("ClassicDefault")
    assert a1 is not a2


class TestDescriptions:
    def test_returns_list_of_tuples(self):
        descs = get_algorithm_descriptions()
        assert isinstance(descs, list)
        for item in descs:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, desc = item
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_includes_all_class_names(self):
        descs = get_algorithm_descriptions()
        names = {d[0] for d in descs}
        assert names == {
            "ClassicDefault",
            "ClassicHighSensitivity",
            "ClassicSolidFill",
            "ClassicLowSensitivity",
        }


class TestInstances:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from testing.registry import _INSTANCES

        _INSTANCES.pop("FakeInstance", None)

    def test_register_and_get_instance(self):
        instance = _FakeInstance()
        register_algorithm_instance("FakeInstance", instance)
        got = get_algorithm("FakeInstance")
        assert got is instance

    def test_registered_instance_is_listed(self):
        register_algorithm_instance("FakeInstance", _FakeInstance())
        names = list_algorithms()
        assert "FakeInstance" in names
        assert "ClassicDefault" in names

    def test_instance_descriptions_included(self):
        register_algorithm_instance("FakeInstance", _FakeInstance())
        descs = dict(get_algorithm_descriptions())
        assert "FakeInstance" in descs

    def test_unknown_after_instance_registration_raises(self):
        register_algorithm_instance("FakeInstance", _FakeInstance())
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm("StillMissing")
