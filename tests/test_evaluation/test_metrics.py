"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sneaker_intel.evaluation.metrics import ModelMetrics, evaluate_model
from sneaker_intel.models.linear import LinearRegressionModel


def test_model_metrics_to_dict() -> None:
    m = ModelMetrics(
        model_name="test",
        mae=10.123,
        mse=200.456,
        rmse=14.158,
        r2=0.9512,
        cv_mae=12.0,
    )
    d = m.to_dict()
    assert d["Model"] == "test"
    assert d["MAE"] == 10.12
    assert d["R2"] == 0.9512


def test_evaluate_model() -> None:
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame({"a": rng.randn(80), "b": rng.randn(80)})
    y_train = pd.Series(3 * X_train["a"] + 2 * X_train["b"] + rng.randn(80) * 0.1)

    X_test = pd.DataFrame({"a": rng.randn(20), "b": rng.randn(20)})
    y_test = pd.Series(3 * X_test["a"] + 2 * X_test["b"] + rng.randn(20) * 0.1)

    model = LinearRegressionModel()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    assert metrics.model_name == "Linear Regression"
    assert metrics.mae >= 0
    assert metrics.r2 > 0.5  # should be a decent fit
