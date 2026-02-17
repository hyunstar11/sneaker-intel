"""Tests for market dynamics analysis."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.analysis.market_dynamics import MarketDynamicsAnalyzer


def test_market_inefficiencies_skips_zero_lowest_ask() -> None:
    df = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "brand": ["Nike", "Nike", "Jordan"],
            "lowestAsk": [0.0, 100.0, 150.0],
            "highestBid": [10.0, 120.0, 140.0],
            "pricePremium": [0.1, 0.2, 0.3],
            "volatility": [0.1, 0.2, 0.3],
            "deadstockSold": [10, 20, 30],
            "numberOfBids": [1, 2, 3],
            "numberOfAsks": [1, 2, 3],
            "salesThisPeriod": [1, 2, 3],
        }
    )

    analyzer = MarketDynamicsAnalyzer(df)
    result = analyzer.market_inefficiencies()

    assert len(result) == 1
    assert result.iloc[0]["item"] == "B"
