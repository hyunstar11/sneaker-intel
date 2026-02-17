"""Region-based feature extraction."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.config import settings
from sneaker_intel.features.base import BaseFeatureExtractor


class RegionFeatureExtractor(BaseFeatureExtractor):
    """Extract binary region features from Buyer Region with 'Other States' bucket."""

    def __init__(self, region_states: list[str] | None = None):
        self._states = region_states or settings.features.region_states

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for state in self._states:
            df[state] = (df["Buyer Region"] == state).astype(int)
        df["Other States"] = (~df["Buyer Region"].isin(self._states)).astype(int)
        return df

    @property
    def feature_names(self) -> list[str]:
        return [*self._states, "Other States"]
