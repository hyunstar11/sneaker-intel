"""Tests for the model registry."""

from __future__ import annotations

import pytest

from sneaker_intel.models.registry import available_models, get_model


def test_get_model_linear() -> None:
    model = get_model("linear")
    assert model.name == "Linear Regression"


def test_get_model_random_forest() -> None:
    model = get_model("random_forest")
    assert model.name == "Random Forest"


def test_get_model_xgboost() -> None:
    try:
        model = get_model("xgboost")
    except RuntimeError as exc:
        pytest.skip(f"XGBoost unavailable in this environment: {exc}")
    assert model.name == "XGBoost"


def test_get_model_lightgbm() -> None:
    try:
        model = get_model("lightgbm")
    except RuntimeError as exc:
        pytest.skip(f"LightGBM unavailable in this environment: {exc}")
    assert model.name == "LightGBM"


def test_get_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent")


def test_available_models() -> None:
    models = available_models()
    assert "linear" in models
    assert "random_forest" in models
    assert "ensemble" in models
