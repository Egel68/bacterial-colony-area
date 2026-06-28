_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper


def get_model(name: str, **kwargs):
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY)


from .unet import UNet  # noqa: E402 — triggers @register_model
