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


def test_overview_ignores_infinite_price_premium() -> None:
    df = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "brand": ["Nike", "Nike", "Jordan"],
            "pricePremium": [0.5, float("inf"), 1.5],
            "volatility": [0.1, 0.2, 0.3],
            "deadstockSold": [10, 20, 30],
            "highestBid": [10, 20, 30],
            "lowestAsk": [10, 20, 30],
            "numberOfBids": [1, 2, 3],
            "numberOfAsks": [1, 2, 3],
            "salesThisPeriod": [1, 2, 3],
        }
    )

    analyzer = MarketDynamicsAnalyzer(df)
    overview = analyzer.overview()

    # Mean/median should be computed from finite values only: [0.5, 1.5]
    assert overview["avg_premium"] == 1.0
    assert overview["median_premium"] == 1.0
