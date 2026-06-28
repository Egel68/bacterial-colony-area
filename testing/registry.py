from typing import Type

from .interface import BaseDetectionAlgorithm

_REGISTRY: dict[str, Type[BaseDetectionAlgorithm]] = {}


def register_algorithm(cls: Type[BaseDetectionAlgorithm]) -> Type[BaseDetectionAlgorithm]:
    _REGISTRY[cls.__name__] = cls
    return cls


def get_algorithm(name: str, **kwargs) -> BaseDetectionAlgorithm:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_algorithms() -> list[str]:
    return sorted(_REGISTRY)


def get_algorithm_descriptions() -> list[tuple[str, str]]:
    return [(name, cls.description) for name, cls in sorted(_REGISTRY.items())]
