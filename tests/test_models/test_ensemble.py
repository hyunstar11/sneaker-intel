"""Tests for the ensemble model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sneaker_intel.models.ensemble import EnsembleModel
from sneaker_intel.models.linear import LinearRegressionModel
from sneaker_intel.models.random_forest import RandomForestModel


@pytest.fixture
def simple_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(100), "b": rng.randn(100)})
    y = pd.Series(3 * X["a"] + 2 * X["b"] + rng.randn(100) * 0.1)
    return X, y


def test_ensemble_fit_predict(simple_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = simple_data
    ensemble = EnsembleModel(models=[LinearRegressionModel(), RandomForestModel()])
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    assert len(predictions) == len(X)
    assert predictions.dtype == np.float64


def test_ensemble_weighted(simple_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = simple_data
    ensemble = EnsembleModel(
        models=[LinearRegressionModel(), RandomForestModel()],
        weights=[0.3, 0.7],
    )
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    assert len(predictions) == len(X)


def test_ensemble_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights length"):
        EnsembleModel(
            models=[LinearRegressionModel()],
            weights=[0.3, 0.7],
        )


def test_ensemble_feature_importances(simple_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = simple_data
    ensemble = EnsembleModel(models=[RandomForestModel()])
    ensemble.fit(X, y)
    imp = ensemble.feature_importances
    assert imp is not None
    assert len(imp) == 2
