"""Model evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from sneaker_intel.models.base import PredictionModel


@dataclass
class ModelMetrics:
    """Container for regression evaluation metrics."""

    model_name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    cv_mae: float | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "Model": self.model_name,
            "MAE": round(self.mae, 2),
            "MSE": round(self.mse, 2),
            "RMSE": round(self.rmse, 2),
            "R2": round(self.r2, 4),
            "CV MAE": round(self.cv_mae, 2) if self.cv_mae is not None else None,
        }


def evaluate_model(
    model: PredictionModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    cv_folds: int = 0,
) -> ModelMetrics:
    """Evaluate a fitted model on test data.

    Args:
        model: Fitted PredictionModel.
        X_test: Test features.
        y_test: Test target.
        X_train: Training features (needed for CV).
        y_train: Training target (needed for CV).
        cv_folds: Number of CV folds. 0 to skip CV.

    Returns:
        ModelMetrics with computed scores.
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, predictions)

    cv_mae = None
    if cv_folds > 0 and X_train is not None and y_train is not None:
        # For CV we need the underlying sklearn model
        from sneaker_intel.models.ensemble import EnsembleModel

        if not isinstance(model, EnsembleModel) and hasattr(model, "_model"):
            scores = cross_val_score(
                model._model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="neg_mean_absolute_error",
            )
            cv_mae = float(-scores.mean())

    return ModelMetrics(
        model_name=model.name,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        cv_mae=cv_mae,
    )
