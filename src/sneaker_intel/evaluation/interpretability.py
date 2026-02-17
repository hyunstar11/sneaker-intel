"""Feature importance and interpretability utilities."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.models.base import PredictionModel


def compute_ensemble_importances(
    models: list[PredictionModel],
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute and aggregate feature importances across multiple models.

    Args:
        models: List of fitted models.
        feature_names: Column names for the features.

    Returns:
        DataFrame with feature importances per model and average.
    """
    data: dict[str, list[float]] = {"Feature": feature_names}

    for model in models:
        imp = model.feature_importances
        if imp is not None:
            normalized = imp / imp.sum() if imp.sum() > 0 else imp
            data[f"Importance_{model.name}"] = normalized.tolist()

    df = pd.DataFrame(data)

    importance_cols = [c for c in df.columns if c.startswith("Importance_")]
    if importance_cols:
        df["Average_Importance"] = df[importance_cols].mean(axis=1)
        df = df.sort_values("Average_Importance", ascending=False).reset_index(drop=True)

    return df


def get_top_features(
    model: PredictionModel,
    feature_names: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Get top N features by importance.

    Args:
        model: Fitted model.
        feature_names: Column names for features.
        top_n: Number of top features to return.

    Returns:
        List of (feature_name, importance) tuples sorted descending.
    """
    imp = model.feature_importances
    if imp is None:
        return []

    pairs = sorted(
        zip(feature_names, imp.tolist(), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return pairs[:top_n]
