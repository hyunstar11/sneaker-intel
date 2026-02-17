"""Market dynamics feature extraction for the sneakers2023 dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sneaker_intel.features.base import BaseFeatureExtractor


class MarketDynamicsExtractor(BaseFeatureExtractor):
    """Derive market intelligence features from the 2023 dataset.

    Creates:
        - bid_ask_spread: lowestAsk - highestBid
        - bid_ask_spread_pct: spread / lowestAsk
        - demand_supply_ratio: numberOfBids / numberOfAsks
        - price_range_pct: (annualHigh - annualLow) / retail
        - sell_through_ratio: salesThisPeriod / deadstockSold
        - premium_sustainability: lastSale / averageDeadstockPrice
    """

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["bid_ask_spread"] = df["lowestAsk"] - df["highestBid"]

        df["bid_ask_spread_pct"] = np.where(
            df["lowestAsk"] > 0,
            df["bid_ask_spread"] / df["lowestAsk"],
            0,
        )

        df["demand_supply_ratio"] = np.where(
            df["numberOfAsks"] > 0,
            df["numberOfBids"] / df["numberOfAsks"],
            0,
        )

        df["price_range_pct"] = np.where(
            df["retail"] > 0,
            (df["annualHigh"] - df["annualLow"]) / df["retail"],
            0,
        )

        df["sell_through_ratio"] = np.where(
            df["deadstockSold"] > 0,
            df["salesThisPeriod"] / df["deadstockSold"],
            0,
        )

        df["premium_sustainability"] = np.where(
            df["averageDeadstockPrice"] > 0,
            df["lastSale"] / df["averageDeadstockPrice"],
            0,
        )

        return df

    @property
    def feature_names(self) -> list[str]:
        return [
            "bid_ask_spread",
            "bid_ask_spread_pct",
            "demand_supply_ratio",
            "price_range_pct",
            "sell_through_ratio",
            "premium_sustainability",
        ]
