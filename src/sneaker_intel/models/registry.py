"""Model registry for creating models by name."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sneaker_intel.models.base import PredictionModel

_REGISTRY: dict[str, tuple[str, str]] = {
    "linear": ("sneaker_intel.models.linear", "LinearRegressionModel"),
    "random_forest": ("sneaker_intel.models.random_forest", "RandomForestModel"),
    "xgboost": ("sneaker_intel.models.xgboost_model", "XGBoostModel"),
    "lightgbm": ("sneaker_intel.models.lightgbm_model", "LightGBMModel"),
    "ensemble": ("sneaker_intel.models.ensemble", "EnsembleModel"),
}


def _resolve_model_class(name: str) -> type:
    module_path, class_name = _REGISTRY[name]
    module = import_module(module_path)
    return getattr(module, class_name)


def get_model(name: str, **kwargs: Any) -> PredictionModel:
    """Create a model instance by name.

    Args:
        name: Model name (linear, random_forest, xgboost, lightgbm, ensemble).
        **kwargs: Passed to the model constructor.

    Returns:
        Instantiated model matching the PredictionModel protocol.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}")
    try:
        model_cls = _resolve_model_class(name)
    except Exception as exc:
        raise RuntimeError(f"Model '{name}' is not available in this environment: {exc}") from exc
    return model_cls(**kwargs)


def available_models() -> list[str]:
    """Return list of registered model names."""
    return list(_REGISTRY.keys())
