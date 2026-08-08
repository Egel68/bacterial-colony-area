from dataclasses import dataclass


@dataclass
class AnalysisParams:
    sensitivity: float = 0.5
    contrast: float = 1.0
    margin_percent: float = 8.0
    min_colony_size: int = 50
    solid_fill: bool = False
    fill_strength: int = 15

    def __post_init__(self):
        assert 0.01 <= self.sensitivity <= 1.0, (
            f"sensitivity out of range [0.01, 1.0]: {self.sensitivity}"
        )
        assert 0.5 <= self.contrast <= 3.0, (
            f"contrast out of range [0.5, 3.0]: {self.contrast}"
        )
        assert 0 <= self.margin_percent <= 30, (
            f"margin_percent out of range [0, 30]: {self.margin_percent}"
        )
        assert 1 <= self.min_colony_size <= 1000, (
            f"min_colony_size out of range [1, 1000]: {self.min_colony_size}"
        )
        assert 1 <= self.fill_strength <= 100, (
            f"fill_strength out of range [1, 100]: {self.fill_strength}"
        )
