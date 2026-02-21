"""Launch strategy analysis — timing and pricing insights."""

from __future__ import annotations

import pandas as pd


class ReleaseStrategyAnalyzer:
    """Analyze launch timing and pricing strategies."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        if "release" in self._df.columns:
            self._df["release"] = pd.to_datetime(self._df["release"], errors="coerce")
            self._df["release_month"] = self._df["release"].dt.month
            self._df["release_dow"] = self._df["release"].dt.day_name()

    def timing_analysis(self) -> pd.DataFrame:
        """Analyze premium by release month."""
        if "release_month" not in self._df.columns:
            return pd.DataFrame()

        return (
            self._df.groupby("release_month")
            .agg(
                avg_premium=("pricePremium", "mean"),
                avg_sales=("salesThisPeriod", "mean"),
                count=("item", "count"),
            )
            .round(3)
            .sort_index()
        )

    def pricing_sensitivity(self) -> pd.DataFrame:
        """Analyze how retail price relates to premium."""
        bins = [0, 100, 150, 200, 250, 300, 500, float("inf")]
        labels = ["<$100", "$100-150", "$150-200", "$200-250", "$250-300", "$300-500", "$500+"]

        df = self._df.copy()
        df["price_bin"] = pd.cut(df["retail"], bins=bins, labels=labels)

        return (
            df.groupby("price_bin", observed=True)
            .agg(
                avg_premium=("pricePremium", "mean"),
                avg_sales=("salesThisPeriod", "mean"),
                count=("item", "count"),
            )
            .round(3)
        )

    def brand_comparison(self) -> pd.DataFrame:
        """Compare key metrics across brands."""
        return (
            self._df.groupby("brand")
            .agg(
                avg_premium=("pricePremium", "mean"),
                avg_sales=("salesThisPeriod", "mean"),
                avg_volatility=("volatility", "mean"),
                count=("item", "count"),
            )
            .sort_values("avg_premium", ascending=False)
            .round(3)
        )
