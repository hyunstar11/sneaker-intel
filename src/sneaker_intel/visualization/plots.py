"""Reusable plot functions for the Sneaker Intel platform."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sneaker_intel.visualization.style import NIKE_PALETTE


def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of actual vs predicted values with regression line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color=NIKE_PALETTE[0])

    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    ax.plot(lims, lims, "--", color=NIKE_PALETTE[1], linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Sale Price")
    ax.set_ylabel("Predicted Sale Price")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Feature Importance",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar chart of feature importances."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    indices = np.argsort(importances)[-top_n:]
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=NIKE_PALETTE[0],
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return ax


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "MAE",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart comparing models on a given metric."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df[metric],
        color=NIKE_PALETTE[: len(comparison_df)],
    )
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison — {metric}")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    figsize: tuple[int, int] = (14, 10),
) -> plt.Axes:
    """Heatmap of feature correlations."""
    fig, ax = plt.subplots(figsize=figsize)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)
    ax.set_title(title)
    return ax


def plot_sales_over_time(
    df: pd.DataFrame,
    date_col: str = "Order Date",
    title: str = "Sales Over Time",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line chart of sales volume over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    daily_sales = df.groupby(date_col).size()
    ax.plot(daily_sales.index, daily_sales.values, color=NIKE_PALETTE[0], linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Sales")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    return ax
