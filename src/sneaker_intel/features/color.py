"""Color-based feature extraction from sneaker names."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.config import settings
from sneaker_intel.features.base import BaseFeatureExtractor


class ColorFeatureExtractor(BaseFeatureExtractor):
    """Extract binary color features + composite 'Colorful' count from Sneaker Name."""

    def __init__(self, color_keywords: list[str] | None = None):
        self._colors = color_keywords or settings.features.color_keywords

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for color in self._colors:
            df[color] = df["Sneaker Name"].str.contains(color, case=False, na=False).astype(int)
        df["Colorful"] = df[self._colors].sum(axis=1)
        return df

    @property
    def feature_names(self) -> list[str]:
        return [*self._colors, "Colorful"]
