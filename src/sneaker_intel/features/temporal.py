"""Temporal feature extraction from date columns."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.features.base import BaseFeatureExtractor


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Decompose Order Date and Release Date into year/month/day + days_since_release."""

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure datetime
        for col in ("Order Date", "Release Date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed")

        df["Order Year"] = df["Order Date"].dt.year
        df["Order Month"] = df["Order Date"].dt.month
        df["Order Day"] = df["Order Date"].dt.day
        df["Release Year"] = df["Release Date"].dt.year
        df["Release Month"] = df["Release Date"].dt.month
        df["Release Day"] = df["Release Date"].dt.day

        df["days_since_release"] = (df["Order Date"] - df["Release Date"]).dt.days

        return df

    @property
    def feature_names(self) -> list[str]:
        return [
            "Order Year",
            "Order Month",
            "Order Day",
            "Release Year",
            "Release Month",
            "Release Day",
            "days_since_release",
        ]
