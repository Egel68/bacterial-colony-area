"""Инвариантный контекст снимка для повторного использования стадий (задача 3.3).

Классические алгоритмы повторно вычисляют общие стадии для одного снимка
(поиск чашки, предобработку) даже при разных параметрах сегментации. Здесь
предусмотрен контекст на sample/variant, который переиспользует **только** те
стадии, чей результат не зависит от параметров алгоритма.

Инвариантные слои и их ключи совместимости:

- **геометрия чашки** (`petri_mask`, `petri_info`): не зависит от параметров
  алгоритма; инвариантна для одного снимка и варианта (source/cropped);
- **предобработка** (`diff = addWeighted(denoised, 1.5, bg, -0.5, 0)`):
  зависит только от входного изображения, варианта, `contrast` (через CLAHE)
  и `blur_size` (медианный фильтр). Слой `masked_diff = bitwise_and(diff,
  diff, mask=roi_mask)` пересчитывается под каждый `margin_percent`.

Несовместимые параметры автоматически отклоняются: если `contrast`/`blur_size`
отличаются от ключа кеша — слой пересчитывается, устаревшее значение не
используется.

Использование::

    ctx = SampleContext(image_id="s1", is_cropped=False)
    a = detector.detect_colonies(img, petri, params_a, petri_info, context=ctx)
    # повторные запуски переиспользуют геометрию и preprocess при совпадении ключа
    b = detector.detect_colonies(img, petri, params_b, petri_info, context=ctx)

Результат с контекстом побитово эквивалентен независимому запуску (см.
`tests/test_sample_context.py`).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Hashable, Optional, Tuple


class SampleContext:
    """Контекст одного снимка (sample + variant).

    Хранит инвариантные слои, ключ совместимости для предобработки —
    `(contrast, blur_size)`. Несовместимые обращения возвращают `None`
    и слой пересчитывается.

    Потокобезопасен: общий контекст для параллельно вычисляемых разных
    алгоритмов одного снимка сериализуется через внутренний `RLock`
    (write-once при совпадении ключа).
    """

    def __init__(self, image_id: Hashable = None, is_cropped: bool = False):
        self.image_id = image_id
        self.is_cropped = bool(is_cropped)
        self.lock = threading.RLock()
        self._geometry: Optional[Tuple[Any, Any]] = None
        self._preprocess_key: Optional[Tuple[float, int]] = None
        self._preprocess_diff: Optional[Any] = None

    # -- Геометрия чашки (инвариантна) ---------------------------------

    @property
    def has_geometry(self) -> bool:
        with self.lock:
            return self._geometry is not None

    def get_geometry(self) -> Optional[Tuple[Any, Any]]:
        with self.lock:
            return self._geometry

    def set_geometry(self, petri_mask, petri_info) -> None:
        with self.lock:
            if self._geometry is None:
                self._geometry = (petri_mask, petri_info)

    # -- Предобработка (зависит от contrast + blur_size) -----------------

    @property
    def preprocess_key(self) -> Optional[Tuple[float, int]]:
        with self.lock:
            return self._preprocess_key

    def is_compatible_with(self, contrast: float, blur_size: int) -> bool:
        """True, если сохранённый preprocess-слой подходит для этих параметров."""
        with self.lock:
            return self._preprocess_key is not None and self._preprocess_key == (
                float(contrast),
                int(blur_size),
            )

    def get_preprocessed(self, contrast: float, blur_size: int):
        """Возвращает `diff` или `None`, если ключ совместимости не совпал."""
        with self.lock:
            if self._preprocess_key == (float(contrast), int(blur_size)):
                return self._preprocess_diff
            return None

    def set_preprocessed(self, contrast: float, blur_size: int, diff) -> None:
        with self.lock:
            key = (float(contrast), int(blur_size))
            if self._preprocess_key != key:
                self._preprocess_key = key
                self._preprocess_diff = diff

    # -- Инвалидация ----------------------------------------------------

    def invalidate_preprocess(self) -> None:
        with self.lock:
            self._preprocess_key = None
            self._preprocess_diff = None

    def invalidate_geometry(self) -> None:
        with self.lock:
            self._geometry = None

    def invalidate_all(self) -> None:
        self.invalidate_preprocess()
        self.invalidate_geometry()

    def __repr__(self) -> str:  # pragma: no cover — для отладки
        with self.lock:
            return (
                f"SampleContext(image_id={self.image_id!r}, "
                f"is_cropped={self.is_cropped}, "
                f"has_geometry={self._geometry is not None}, "
                f"preprocess_key={self._preprocess_key!r})"
            )


_DEFAULT_CACHE_SIZE = 8


class SampleContextCache:
    """Ограниченный LRU-кэш контекстов по `(image_id, is_cropped)`.

    Один снимок может иметь до двух контекстов (source и cropped) — варианты
    обрабатываются независимо.
    """

    def __init__(self, maxsize: int = _DEFAULT_CACHE_SIZE):
        self.maxsize = int(maxsize)
        self._entries: Dict[Tuple[Hashable, bool], SampleContext] = {}

    def get_or_create(self, image_id: Hashable, is_cropped: bool) -> SampleContext:
        key = (image_id, bool(is_cropped))
        if key in self._entries:
            self._entries[key] = self._entries.pop(key)  # LRU touch
            return self._entries[key]
        context = SampleContext(image_id=image_id, is_cropped=is_cropped)
        self._entries[key] = context
        while len(self._entries) > self.maxsize:
            self._entries.pop(next(iter(self._entries)))
        return context

    def get(self, image_id: Hashable, is_cropped: bool) -> Optional[SampleContext]:
        return self._entries.get((image_id, bool(is_cropped)))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
