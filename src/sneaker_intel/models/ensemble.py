"""Ensemble model combining multiple base models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sneaker_intel.models.base import PredictionModel


@dataclass
class EnsembleModel:
    """Average or weighted-average ensemble of multiple models."""

    models: list[PredictionModel]
    weights: list[float] | None = None
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError("weights length must match models length")
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    @property
    def name(self) -> str:
        return "Ensemble"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for model in self.models:
            model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        if self.weights is not None:
            return np.average(predictions, axis=0, weights=self.weights)
        return predictions.mean(axis=0)

    def get_params(self) -> dict[str, Any]:
        return {
            "models": [m.name for m in self.models],
            "weights": self.weights,
        }

    @property
    def feature_importances(self) -> np.ndarray | None:
        importances = []
        for model in self.models:
            imp = model.feature_importances
            if imp is not None:
                normalized = imp / imp.sum() if imp.sum() > 0 else imp
                importances.append(normalized)
        if not importances:
            return None
        return np.mean(importances, axis=0)
