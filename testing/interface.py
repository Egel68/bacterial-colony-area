from abc import ABC, abstractmethod

import numpy as np


class BaseDetectionAlgorithm(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        is_cropped: bool = False,
        context=None,
    ) -> np.ndarray:
        """Возвращает маску детекции.

        `context` — необязательный `analysis.sample_context.SampleContext`
        для переиспользования инвариантных стадий (геометрия чашки,
        предобработка) между совместимыми запусками на одном снимке.
        Алгоритмы, которым контекст не нужен, игнорируют параметр.
        """
        ...
