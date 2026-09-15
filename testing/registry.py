from typing import Type

from .interface import BaseDetectionAlgorithm

# Классы алгоритмов, регистрируемые через @register_algorithm.
# Каждый запрос создаёт новый экземпляр.
_CLASSES: dict[str, Type[BaseDetectionAlgorithm]] = {}

# Готовые экземпляры алгоритмов (например, ONNX-модели с весами),
# регистрируемые через register_algorithm_instance.
_INSTANCES: dict[str, BaseDetectionAlgorithm] = {}


def register_algorithm(
    cls: Type[BaseDetectionAlgorithm],
) -> Type[BaseDetectionAlgorithm]:
    """Регистрирует класс алгоритма по имени класса."""
    _CLASSES[cls.__name__] = cls
    return cls


def register_algorithm_instance(
    name: str, instance: BaseDetectionAlgorithm
) -> BaseDetectionAlgorithm:
    """Регистрирует готовый экземпляр алгоритма (модель с весами)."""
    _INSTANCES[name] = instance
    return instance


def _all_names() -> list[str]:
    return sorted(set(_CLASSES) | set(_INSTANCES))


def get_algorithm(name: str, **kwargs) -> BaseDetectionAlgorithm:
    """Возвращает экземпляр алгоритма по имени (класс или зарегистрированный экземпляр)."""
    if name in _CLASSES:
        return _CLASSES[name](**kwargs)
    if name in _INSTANCES:
        return _INSTANCES[name]
    available = ", ".join(_all_names())
    raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")


def list_algorithms() -> list[str]:
    return _all_names()


def get_algorithm_descriptions() -> list[tuple[str, str]]:
    descriptions = []
    for name in _all_names():
        if name in _CLASSES:
            cls = _CLASSES[name]
            description = getattr(cls, "description", "")
        else:
            description = getattr(_INSTANCES[name], "description", "")
        descriptions.append((name, description))
    return descriptions
