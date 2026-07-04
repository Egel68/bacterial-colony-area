from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    petri_diameter_mm: float = 90.0
    supported_formats: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
    )


@dataclass
class AnalysisDefaults:
    sensitivity: int = 50
    contrast: float = 1.0
    margin_pct: float = 8.0
    min_colony_size_px: int = 50
    solid_fill: bool = False
    fill_strength: int = 15
