import pytest

from analysis.params import AnalysisParams


class TestAnalysisParamsValidation:
    def test_default_values_ok(self):
        params = AnalysisParams()
        assert params.sensitivity == 0.5
        assert params.contrast == 1.0
        assert params.margin_percent == 8.0
        assert params.min_colony_size == 50
        assert params.solid_fill is False
        assert params.fill_strength == 15

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"sensitivity": 0.0}, "sensitivity"),
            ({"sensitivity": 2.0}, "sensitivity"),
            ({"contrast": 0.4}, "contrast"),
            ({"contrast": 5.0}, "contrast"),
            ({"margin_percent": -1}, "margin_percent"),
            ({"margin_percent": 50}, "margin_percent"),
            ({"min_colony_size": 0}, "min_colony_size"),
            ({"min_colony_size": 2000}, "min_colony_size"),
            ({"fill_strength": 0}, "fill_strength"),
            ({"fill_strength": 200}, "fill_strength"),
        ],
    )
    def test_out_of_range_rejected(self, kwargs, field):
        with pytest.raises(AssertionError, match=field):
            AnalysisParams(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sensitivity": 0.01},
            {"sensitivity": 1.0},
            {"contrast": 0.5},
            {"contrast": 3.0},
            {"margin_percent": 0},
            {"margin_percent": 30},
            {"min_colony_size": 1},
            {"min_colony_size": 1000},
            {"fill_strength": 1},
            {"fill_strength": 100},
        ],
    )
    def test_boundary_values_accepted(self, kwargs):
        AnalysisParams(**kwargs)