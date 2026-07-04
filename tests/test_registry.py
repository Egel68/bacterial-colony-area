import pytest

from testing.registry import (
    get_algorithm,
    list_algorithms,
    get_algorithm_descriptions,
)


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

    def test_includes_all_algorithms(self):
        descs = get_algorithm_descriptions()
        names = {d[0] for d in descs}
        assert names == {
            "ClassicDefault",
            "ClassicHighSensitivity",
            "ClassicSolidFill",
            "ClassicLowSensitivity",
        }
