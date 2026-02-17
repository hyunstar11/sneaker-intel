"""LightGBM model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from sneaker_intel.config import settings


@dataclass
class LightGBMModel:
    """Wrapper around LGBMRegressor."""

    boosting_type: str = settings.models.lgbm_boosting_type
    num_leaves: int = settings.models.lgbm_num_leaves
    learning_rate: float = settings.models.lgbm_learning_rate
    n_estimators: int = settings.models.lgbm_n_estimators
    feature_fraction: float = settings.models.lgbm_feature_fraction
    bagging_fraction: float = settings.models.lgbm_bagging_fraction
    bagging_freq: int = settings.models.lgbm_bagging_freq
    random_state: int = settings.models.rf_random_state
    _model: LGBMRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = LGBMRegressor(
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            random_state=self.random_state,
            verbose=-1,
        )

    @property
    def name(self) -> str:
        return "LightGBM"

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
