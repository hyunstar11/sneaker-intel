"""Base protocol for prediction models."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class PredictionModel(Protocol):
    """Protocol defining the interface for all prediction models."""

    @property
    def name(self) -> str: ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def get_params(self) -> dict[str, Any]: ...

    @property
    def feature_importances(self) -> np.ndarray | None: ...
