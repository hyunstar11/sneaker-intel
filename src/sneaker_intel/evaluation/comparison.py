"""Multi-model comparison utilities."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.evaluation.metrics import ModelMetrics


def compare_models(metrics_list: list[ModelMetrics]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple model metrics.

    Args:
        metrics_list: List of ModelMetrics from different models.

    Returns:
        DataFrame with one row per model, sorted by MAE ascending.
    """
    rows = [m.to_dict() for m in metrics_list]
    df = pd.DataFrame(rows)
    return df.sort_values("MAE").reset_index(drop=True)
