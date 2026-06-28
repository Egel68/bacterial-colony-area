from abc import ABC, abstractmethod

import numpy as np


class BaseDetectionAlgorithm(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def detect(self, image: np.ndarray, is_cropped: bool = False) -> np.ndarray:
        ...
