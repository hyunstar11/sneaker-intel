"""Shared StockX-specific feature helpers."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median

import pandas as pd


def build_number_of_sales_reference(
    df: pd.DataFrame,
    date_col: str = "Order Date",
) -> tuple[dict[pd.Timestamp, float], float]:
    """Build a lookup of daily sales counts and a default fallback value."""
    if date_col not in df.columns:
        return {}, 0.0

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    counts = parsed_dates.value_counts(dropna=True)
    sales_lookup = {pd.Timestamp(date): float(count) for date, count in counts.items()}
    default_sales = float(median(sales_lookup.values())) if sales_lookup else 0.0
    return sales_lookup, default_sales


def add_number_of_sales_feature(
    df: pd.DataFrame,
    sales_lookup: Mapping[pd.Timestamp, float],
    default_sales: float,
    date_col: str = "Order Date",
) -> pd.DataFrame:
    """Add Number of Sales using a shared lookup built from training data."""
    df = df.copy()
    if date_col not in df.columns:
        df["Number of Sales"] = float(default_sales)
        return df

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    df["Number of Sales"] = parsed_dates.map(sales_lookup).fillna(default_sales).astype(float)
    return df
