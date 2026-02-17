"""Market dynamics analysis — liquidity, volatility, inefficiencies."""

from __future__ import annotations

import pandas as pd


class MarketDynamicsAnalyzer:
    """Analyze market dynamics from the sneakers2023 dataset."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

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
        """Detect market inefficiencies where highestBid > lowestAsk."""
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
        return {
            "total_products": len(df),
            "total_brands": df["brand"].nunique(),
            "avg_premium": round(df["pricePremium"].mean(), 3),
            "median_premium": round(df["pricePremium"].median(), 3),
            "avg_volatility": round(df["volatility"].mean(), 4),
            "total_deadstock_sold": int(df["deadstockSold"].sum()),
        }
