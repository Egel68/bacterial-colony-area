from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisResult:
    colony_count: int
    colony_area_px: int
    colony_area_mm2: float
    petri_area_px: int
    petri_area_mm2: float
    coverage_percent: float
    px_to_mm2: float
