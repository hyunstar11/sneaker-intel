"""Linear Regression model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class LinearRegressionModel:
    """Wrapper around sklearn LinearRegression with unified interface."""

    _model: LinearRegression = field(default_factory=LinearRegression, repr=False)

    @property
    def name(self) -> str:
        return "Linear Regression"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict[str, Any]:
        return self._model.get_params()

    @property
    def feature_importances(self) -> np.ndarray | None:
        if hasattr(self._model, "coef_"):
            return np.abs(self._model.coef_)
        return None
