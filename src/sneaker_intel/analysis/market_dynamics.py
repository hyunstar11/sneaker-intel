"""Market dynamics analysis — liquidity, volatility, pricing opportunities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketDynamicsAnalyzer:
    """Analyze market dynamics from the sneakers2023 dataset."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        if "pricePremium" in self._df.columns:
            premium = pd.to_numeric(self._df["pricePremium"], errors="coerce")
            self._df["pricePremium"] = premium.where(np.isfinite(premium), pd.NA)

    def liquidity_analysis(self) -> pd.DataFrame:
        """Analyze market liquidity by brand."""
        return (
            self._df.groupby("brand")
            .agg(
                avg_bids=("numberOfBids", "mean"),
                avg_asks=("numberOfAsks", "mean"),
                avg_deadstock=("deadstockSold", "mean"),
                avg_sales=("salesThisPeriod", "mean"),
            )
            .sort_values("avg_deadstock", ascending=False)
            .round(1)
        )

    def volatility_drivers(self, top_n: int = 20) -> pd.DataFrame:
        """Find most volatile sneakers and their characteristics."""
        return self._df.nlargest(top_n, "volatility")[
            ["item", "brand", "retail", "volatility", "pricePremium", "deadstockSold"]
        ].reset_index(drop=True)

    def market_inefficiencies(self) -> pd.DataFrame:
        """Detect pricing opportunities where highestBid > lowestAsk."""
        inefficient = self._df[
            (self._df["highestBid"] > self._df["lowestAsk"]) & (self._df["lowestAsk"] > 0)
        ].copy()
        if inefficient.empty:
            return pd.DataFrame(
                columns=["item", "brand", "lowestAsk", "highestBid", "arbitrage_pct"]
            )

        inefficient["arbitrage_pct"] = (
            (inefficient["highestBid"] - inefficient["lowestAsk"]) / inefficient["lowestAsk"]
        ).round(4)
        return (
            inefficient[["item", "brand", "lowestAsk", "highestBid", "arbitrage_pct"]]
            .sort_values("arbitrage_pct", ascending=False)
            .reset_index(drop=True)
        )

    def overview(self) -> dict[str, float]:
        """High-level market overview stats."""
        df = self._df
        premium = df["pricePremium"].dropna() if "pricePremium" in df.columns else pd.Series()
        avg_premium = round(float(premium.mean()), 3) if not premium.empty else 0.0
        median_premium = round(float(premium.median()), 3) if not premium.empty else 0.0
        return {
            "total_products": len(df),
            "total_brands": df["brand"].nunique(),
            "avg_premium": avg_premium,
            "median_premium": median_premium,
            "avg_volatility": round(df["volatility"].mean(), 4),
            "total_deadstock_sold": int(df["deadstockSold"].sum()),
        }
