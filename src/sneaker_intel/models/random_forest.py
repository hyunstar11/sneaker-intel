"""Random Forest model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sneaker_intel.config import settings


@dataclass
class RandomForestModel:
    """Wrapper around sklearn RandomForestRegressor."""

    n_estimators: int = settings.models.rf_n_estimators
    random_state: int = settings.models.rf_random_state
    _model: RandomForestRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict[str, Any]:
        return self._model.get_params()

    @property
    def feature_importances(self) -> np.ndarray | None:
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_
        return None
