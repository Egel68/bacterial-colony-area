from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PetriInfo:
    cx: int
    cy: int
    radius: int
    image_shape: tuple[int, int]

    def __post_init__(self):
        assert self.radius > 0, f"radius must be positive, got {self.radius}"
        assert self.cx >= 0 and self.cy >= 0, "center must be non-negative"
        h, w = self.image_shape
        assert self.cy < h, f"center y ({self.cy}) >= image height ({h})"
        assert self.cx < w, f"center x ({self.cx}) >= image width ({w})"

    @property
    def center(self) -> tuple[int, int]:
        return (self.cx, self.cy)

    @property
    def area_px(self) -> int:
        return int(np.pi * self.radius**2)
