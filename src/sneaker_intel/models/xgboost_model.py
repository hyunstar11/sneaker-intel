"""XGBoost model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sneaker_intel.config import settings


@dataclass
class XGBoostModel:
    """Wrapper around XGBRegressor."""

    max_depth: int = settings.models.xgb_max_depth
    learning_rate: float = settings.models.xgb_learning_rate
    n_estimators: int = settings.models.xgb_n_estimators
    objective: str = settings.models.xgb_objective
    random_state: int = settings.models.rf_random_state
    _model: XGBRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective=self.objective,
            random_state=self.random_state,
        )

    @property
    def name(self) -> str:
        return "XGBoost"

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
