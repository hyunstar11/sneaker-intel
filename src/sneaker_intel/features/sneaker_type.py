"""Sneaker type feature extraction."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.config import settings
from sneaker_intel.features.base import BaseFeatureExtractor


class SneakerTypeExtractor(BaseFeatureExtractor):
    """Extract binary sneaker type features from Sneaker Name."""

    def __init__(self, sneaker_type_keywords: dict[str, str] | None = None):
        self._types = sneaker_type_keywords or settings.features.sneaker_type_keywords

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature_name, keyword in self._types.items():
            df[feature_name] = (
                df["Sneaker Name"].str.contains(keyword, case=False, na=False).astype(int)
            )
        return df

    @property
    def feature_names(self) -> list[str]:
        return list(self._types.keys())
