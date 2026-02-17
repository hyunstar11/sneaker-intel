"""Shoe size normalization feature."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sneaker_intel.features.base import BaseFeatureExtractor


@dataclass
class SizeNormalizer(BaseFeatureExtractor):
    """Compute size frequency using training distribution and reuse at inference."""

    unknown_frequency: float = 0.0
    _size_frequencies: dict[float, float] | None = field(default=None, init=False, repr=False)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Learn once from a non-trivial batch (training flow), then reuse for inference.
        if self._size_frequencies is None and len(df) > 1:
            self._size_frequencies = df["Shoe Size"].value_counts(normalize=True).to_dict()

        if self._size_frequencies is None:
            df["size_freq"] = float(self.unknown_frequency)
            return df

        df["size_freq"] = (
            df["Shoe Size"]
            .map(self._size_frequencies)
            .fillna(self.unknown_frequency)
            .astype(float)
        )
        return df

    @property
    def feature_names(self) -> list[str]:
        return ["size_freq"]
